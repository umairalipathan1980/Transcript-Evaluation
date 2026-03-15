# Command: python enhance_transcript_no_lexicon.py --input-dir <hypotheses_root_dir> --output-dir <enhanced_root_dir> [--no-azure]
"""
Enhance Finnish transcripts using GPT-5.4 (no domain-specific dictionary).

Strategy:
- Read transcripts from a directory
- Use GPT-5.4 to correct spelling, capitalization, and inflection
- CONSTRAINT: Do NOT add or remove words
- Save enhanced versions to an output directory
"""

import argparse
import os
from pathlib import Path

from config import create_openai_client

USE_AZURE_DEFAULT = True

DEFAULT_MODEL_AZURE = "gpt-5.4"
DEFAULT_MODEL_OPENAI = "gpt-5.4-2026-03-05"

##Focuses on spelling corrections

PASS1_SYSTEM_PROMPT = """You are a Finnish transcript editor.

PRIMARY GOAL: Maximize spelling correctness and spelling consistency while preserving the original meaning and style.

What to do (high priority):
1) Spelling consistency is TOP PRIORITY.
   - If the same content term appears with multiple spellings in this transcript, choose the best Finnish spelling/canonical form and normalize ALL occurrences to that form everywhere.
   - This includes technical terms, proper nouns, abbreviations, and loanwords.

2) Finnish vocabulary:
   - Prefer valid Finnish words and standard Finnish orthography.
   - If a token looks malformed or non-Finnish but the intended Finnish word is obvious from immediate context, correct it into a valid Finnish word.
   - Preserve common loanwords/brand names when they are clearly intended.

3) Technical terms and names:
   - Correct capitalization of proper nouns/brands when clearly identifiable.
   - Do NOT change a person’s name to a different person. Only correct spelling/casing for the same name (or dictionary-mapped variant).

4) Hyphenation / compounds (consistency):
   - Normalize consistent hyphenation and compound forms when it is clearly the same intended term.
   - Normalize common compound terms consistently across the transcript.

Forbidden:
- Do NOT summarize, rewrite, paraphrase, or reorder sentences.
- Do NOT add new facts or explanations.
- Do NOT invent new names, brands, roles, or titles.
- Avoid inserting or deleting words unless it is required to fix a clear tokenization artifact (e.g., accidental split/merge that keeps the same meaning).
- Avoid merging two separate words into one or removing tokens. Prefer minimal spelling fixes that keep word boundaries stable.

Output:
Return ONLY the corrected transcript text with no commentary.
"""

## Focuses on context based repair
PASS2_SYSTEM_PROMPT = """You are a Finnish transcript repair editor.

GOAL: Reduce transcription errors using context, while staying faithful to SPOKEN Finnish. This is a transcript of speech, so preserve colloquial forms.

Allowed repairs (ONLY when confident):
1) Insert short Finnish function/filler words ONLY from this set:
   että, ja, niin, se, on, eli, siis, sitten, kun, mutta, myös, et, niinku, joo
   - Insert only if the surrounding grammar strongly requires it and the insertion is extremely likely.
   - Do NOT insert content words (nouns/verbs/adjectives) unless it is clearly a split/merge artifact.

2) Fix split/merge and compounds:
   - Merge compound words that ASR incorrectly split: "lauantai töiksi" → "lauantaitöiksi", "reaali maailmassa" → "reaalimaailmassa"
   - Fix broken hyphenation consistently (e.g., peri implantiitti ↔ peri-implantiitti).
   - Fix malformed loanwords/terms consistently, but do not invent new terms.

3) Finish remaining spelling/casing consistency:
   - Ensure the same term is spelled the same way throughout the transcript.
   - Ensure malformed/non-Finnish tokens are corrected when the intended word is obvious from immediate context.

4) Convert numeric digits to Finnish word numbers WITH CORRECT INFLECTION:
   CRITICAL: Use correct Finnish grammatical case for numbers!
   - Genitive case (possessive): "20 prosentin" → "kahdenkymmenen prosentin" (NOT "kaksikymmenen")
   - Nominative: "20 prosenttia" → "kaksikymmentä prosenttia"
   - Common genitive forms: yhden, kahden, kolmen, neljän, viiden, kuuden, seitsemän, kahdeksan, yhdeksän, kymmenen
   - "11" genitive → "yhdentoista", "20" genitive → "kahdenkymmenen", "55" genitive → "viidenkymmenenviiden"
   - Decimals: "37,5" → "kolmekymmentäseitsemän ja puoli"
   - Years/decades: "70-luvulta" → "seitsemänkymmentäluvulta"
   - Keep numbers in proper nouns/codes unchanged (e.g., "COVID-19", "ISO 9001")
   - Keep date and time exactly in the same format (DO NOT change)

5) PRESERVE COLLOQUIAL FINNISH (spoken language):
   - Keep colloquial forms if present: "tän", "tää", "et", "sitte", "sit", "oo", "mä", "sä", "niinku", "elikkä"
   - Do NOT "correct" colloquial forms to formal Finnish
   - This is a transcript of natural speech, not formal written text

Hard constraints (must follow):
- Do NOT delete any words in Pass 2 (number conversion may change word count).
- Do NOT introduce any new names, brands, roles, or titles.
- Do NOT replace one person's name with another.
- Do NOT rewrite or paraphrase sentences.
- Do NOT add new sentences or remove entire phrases.
- Do NOT convert colloquial Finnish to formal Finnish.

Insertion budget:
- At most 4 inserted words per 100 words of transcript (excluding number conversions).
- If you are near the budget, prioritize the most grammar-critical insertions only.

If uncertain about a change, leave the original text unchanged.

Output:
Return ONLY the repaired transcript text with no commentary.
"""

def get_client(use_azure: bool = True):
    if use_azure:
        config = {
            "use_azure": True,
            "api_key": os.getenv("AZURE_API_KEY"),
            "azure_endpoint": "https://haagahelia-poc-gaik.openai.azure.com/",
            "api_version": "2025-03-01-preview",
            "model": DEFAULT_MODEL_AZURE,
        }
    else:
        config = {
            "use_azure": False,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": DEFAULT_MODEL_OPENAI,
        }

    if not config.get("api_key"):
        key_name = "AZURE_API_KEY" if use_azure else "OPENAI_API_KEY"
        raise SystemExit(f"{key_name} not found in environment")

    return create_openai_client(config), config

def enhance_transcript_pass1(client, transcript_text: str, model: str) -> str:
    """
    Pass 1: Fix spelling consistency, capitalization, and Finnish vocabulary.
    Focus on making terms consistent and correctly spelled.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PASS1_SYSTEM_PROMPT},
            {"role": "user", "content": f"Edit this Finnish transcript for spelling consistency:\n\n{transcript_text}"}
        ],
        temperature=0.0,
    )
    return _extract_response_text(response, fallback_text=transcript_text, stage_name="Pass 1")

def enhance_transcript_pass2(client, transcript_text: str, model: str) -> str:
    """
    Pass 2: Context-based repair with limited insertions/deletions allowed.
    Fix ASR-specific errors like dropped filler words and compound splitting.
    Also converts numeric digits to Finnish word numbers.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PASS2_SYSTEM_PROMPT},
            {"role": "user", "content": f"Repair remaining ASR errors in this Finnish transcript:\n\n{transcript_text}"}
        ],
        temperature=0.0,
    )
    return _extract_response_text(response, fallback_text=transcript_text, stage_name="Pass 2")


def _extract_response_text(response, fallback_text: str, stage_name: str) -> str:
    """
    OpenAI responses may occasionally include None content (e.g., filtered/refusal-like outputs).
    In that case, keep pipeline running by returning the input text unchanged.
    """
    if not response or not getattr(response, "choices", None):
        print(f"  WARNING: {stage_name} returned no choices; using original text unchanged.")
        return fallback_text

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None

    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return stripped
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
            else:
                text_part = getattr(item, "text", None)
            if isinstance(text_part, str) and text_part.strip():
                parts.append(text_part.strip())
        if parts:
            return "\n".join(parts)

    finish_reason = getattr(choice, "finish_reason", None)
    refusal = getattr(message, "refusal", None) if message is not None else None
    print(
        f"  WARNING: {stage_name} returned empty content "
        f"(finish_reason={finish_reason}, refusal={bool(refusal)}); using original text unchanged."
    )
    return fallback_text

def _process_one_directory(client, transcripts_path: Path, output_path: Path, model: str):
    output_path.mkdir(parents=True, exist_ok=True)
    transcript_files = sorted(transcripts_path.glob("*.txt"))
    if not transcript_files:
        print(f"No .txt files found in {transcripts_path}")
        return 0

    print(f"Found {len(transcript_files)} transcripts in {transcripts_path.name}")
    success_count = 0
    for transcript_file in transcript_files:
        try:
            print(f"Processing: {transcripts_path.name}/{transcript_file.name}")
            original_text = transcript_file.read_text(encoding="utf-8")
            original_word_count = len(original_text.split())
            print(f"  Original: {original_word_count} words")

            print("  Pass 1: Spelling consistency...")
            pass1_text = enhance_transcript_pass1(client, original_text, model=model)
            pass1_word_count = len(pass1_text.split())
            print(f"    -> {pass1_word_count} words (delta: {pass1_word_count - original_word_count:+d})")

            print("  Pass 2: Context repair + number conversion...")
            pass2_text = enhance_transcript_pass2(client, pass1_text, model=model)
            pass2_word_count = len(pass2_text.split())
            print(f"    -> {pass2_word_count} words (delta: {pass2_word_count - pass1_word_count:+d})")
            print(f"  Total change: {original_word_count} -> {pass2_word_count} words ({pass2_word_count - original_word_count:+d})")

            output_file = output_path / transcript_file.name
            output_file.write_text(pass2_text, encoding="utf-8")
            print(f"  Saved to: {output_file}")
            print()
            success_count += 1
        except Exception as exc:
            print(f"  ERROR: Failed to process {transcript_file.name}: {exc}")
            print("  Skipping file and continuing.\n")

    return success_count


def process_all_hypothesis_directories(
    hypothesis_root: str | Path,
    enhanced_root_dir: str | Path,
    use_azure: bool = True,
):
    hypothesis_root = Path(hypothesis_root)
    enhanced_root_dir = Path(enhanced_root_dir)
    if not hypothesis_root.exists():
        raise FileNotFoundError(f"Hypothesis root directory not found: {hypothesis_root}")

    enhanced_root_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted([p for p in hypothesis_root.iterdir() if p.is_dir()])
    if not model_dirs:
        print(f"No model directories found in {hypothesis_root}")
        return

    client, config = get_client(use_azure=use_azure)
    model = config["model"]

    provider = "Azure OpenAI" if use_azure else "OpenAI"
    print(f"Using hypothesis root: {hypothesis_root}")
    print(f"Writing enhanced outputs to: {enhanced_root_dir}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print()

    total_dirs = 0
    total_files = 0
    for model_dir in model_dirs:
        output_subdir = enhanced_root_dir / f"{model_dir.name}_enhanced"
        print("=" * 80)
        print(f"Enhancing model folder: {model_dir.name}")
        print(f"Output folder: {output_subdir}")
        print("=" * 80)
        processed = _process_one_directory(client, model_dir, output_subdir, model=model)
        if processed:
            total_dirs += 1
            total_files += processed

    print()
    print(f"Done! Enhanced {total_files} transcript files across {total_dirs} model folders.")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance transcripts without a domain lexicon."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root folder containing model hypothesis subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root folder where enhanced model subdirectories will be written.",
    )
    parser.add_argument(
        "--no-azure",
        action="store_true",
        help="Use standard OpenAI API instead of Azure OpenAI.",
    )
    args = parser.parse_args()

    process_all_hypothesis_directories(
        hypothesis_root=args.input_dir,
        enhanced_root_dir=args.output_dir,
        use_azure=not args.no_azure,
    )

if __name__ == "__main__":
    main()
