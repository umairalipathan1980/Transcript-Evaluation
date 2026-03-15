
# Command: python eval_ref_hyp.py <reference_dir> <hypotheses_root_dir> [output_xlsx]
"""
Evaluate reference transcripts against all hypothesis model subfolders.

Usage:
    python eval_ref_hyp.py <reference_dir> <hypotheses_root_dir> [output_xlsx]

- <reference_dir> contains reference .txt files.
- <hypotheses_root_dir> contains one subdirectory per model, each with hypothesis .txt files.
- [output_xlsx] is optional (default: eval_ref_hyp_output.xlsx).
"""

import sys
from pathlib import Path

import jiwer
from openpyxl import Workbook
from rapidfuzz.distance import Levenshtein

TABLE_HEADERS = [
    "Model",
    "WER",
    "Spelling Mistake Rate",
    "Substitution Rate",
    "Insertion Rate",
    "Deletion Rate",
]

# Right-align numeric columns for readability.
NUMERIC_COLS = {1, 2, 3, 4, 5}


def _build_transform():
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def _evaluate_texts(reference_text, hypothesis_text):
    word_transform = _build_transform()
    return jiwer.process_words(
        reference_text,
        hypothesis_text,
        reference_transform=word_transform,
        hypothesis_transform=word_transform,
    )


def _is_spelling_error(ref_word, hyp_word, max_normalized_distance=0.4):
    max_len = max(len(ref_word), len(hyp_word))
    if max_len == 0:
        return False
    distance = Levenshtein.distance(ref_word, hyp_word)
    normalized_distance = distance / max_len
    return normalized_distance <= max_normalized_distance


def _spelling_error_count(output, max_normalized_distance=0.4):
    spelling_errors = 0
    for ref_words_sent, hyp_words_sent, chunks in zip(
        output.references, output.hypotheses, output.alignments
    ):
        for chunk in chunks:
            if chunk.type != "substitute":
                continue
            ref_span = ref_words_sent[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_span = hyp_words_sent[chunk.hyp_start_idx : chunk.hyp_end_idx]
            pair_len = min(len(ref_span), len(hyp_span))
            for i in range(pair_len):
                ref_word = ref_span[i]
                hyp_word = hyp_span[i]
                if _is_spelling_error(
                    ref_word, hyp_word, max_normalized_distance=max_normalized_distance
                ):
                    spelling_errors += 1
    return spelling_errors


def _init_totals():
    return {
        "subs": 0,
        "dels": 0,
        "ins": 0,
        "ref_words": 0,
        "spelling_errors": 0,
    }


def _update_totals(totals, output):
    totals["subs"] += output.substitutions
    totals["dels"] += output.deletions
    totals["ins"] += output.insertions
    totals["ref_words"] += output.hits + output.substitutions + output.deletions
    totals["spelling_errors"] += _spelling_error_count(output)


def _rates(totals):
    ref_words = totals["ref_words"]
    if ref_words == 0:
        return {
            "wer": 0.0,
            "spelling": 0.0,
            "substitution": 0.0,
            "insertion": 0.0,
            "deletion": 0.0,
        }

    return {
        "wer": (totals["subs"] + totals["dels"] + totals["ins"]) / ref_words,
        "spelling": totals["spelling_errors"] / ref_words,
        "substitution": totals["subs"] / ref_words,
        "insertion": totals["ins"] / ref_words,
        "deletion": totals["dels"] / ref_words,
    }


def _fmt_percent(value: float) -> str:
    return f"{value:.2%}"


def _evaluate_model(reference_dir: Path, model_dir: Path):
    totals = _init_totals()
    evaluated = 0

    for ref_file in sorted(reference_dir.glob("*.txt")):
        hyp_file = model_dir / ref_file.name
        if not hyp_file.exists():
            continue

        ref_text = ref_file.read_text(encoding="utf-8")
        hyp_text = hyp_file.read_text(encoding="utf-8")
        output = _evaluate_texts(ref_text, hyp_text)
        _update_totals(totals, output)
        evaluated += 1

    if evaluated == 0:
        return None

    return {"rates": _rates(totals)}


def _compute_col_widths(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    return widths


def _format_row(row, widths, numeric_cols=None):
    numeric_cols = numeric_cols or set()
    cells = []
    for i, (value, width) in enumerate(zip(row, widths)):
        text = str(value)
        cells.append(text.rjust(width) if i in numeric_cols else text.ljust(width))
    return "| " + " | ".join(cells) + " |"


def _print_table(headers, rows):
    widths = _compute_col_widths(headers, rows)
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print(sep)
    print(_format_row(headers, widths, numeric_cols=set()))
    print(sep)
    for row in rows:
        print(_format_row(row, widths, numeric_cols=NUMERIC_COLS))
    print(sep)


def _save_xlsx(rows, output_xlsx: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "evaluation"
    ws.append(TABLE_HEADERS)
    for row in rows:
        ws.append(row)
    wb.save(output_xlsx)


def main():
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python eval_ref_hyp.py "
            "<reference_dir> <hypotheses_root_dir> [output_xlsx]"
        )
        sys.exit(1)

    reference_dir = Path(sys.argv[1])
    hypotheses_root_dir = Path(sys.argv[2])
    script_dir = Path(__file__).resolve().parent
    output_xlsx = (
        Path(sys.argv[3]) if len(sys.argv) == 4 else script_dir / "eval_ref_hyp_output.xlsx"
    )
    if not output_xlsx.is_absolute():
        output_xlsx = script_dir / output_xlsx

    if not reference_dir.exists() or not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    if not hypotheses_root_dir.exists() or not hypotheses_root_dir.is_dir():
        raise FileNotFoundError(f"Hypotheses root directory not found: {hypotheses_root_dir}")

    model_dirs = sorted([p for p in hypotheses_root_dir.iterdir() if p.is_dir()])
    if not model_dirs:
        print(f"No model subdirectories found in {hypotheses_root_dir}")
        return

    rows = []

    print(f"Reference directory:   {reference_dir}")
    print(f"Hypotheses root dir:  {hypotheses_root_dir}")
    print()

    for model_dir in model_dirs:
        metrics = _evaluate_model(reference_dir, model_dir)
        if metrics is None:
            print(f"Skipping {model_dir.name}: no matching .txt files with references")
            continue

        rates = metrics["rates"]
        rows.append(
            [
                model_dir.name,
                _fmt_percent(rates["wer"]),
                _fmt_percent(rates["spelling"]),
                _fmt_percent(rates["substitution"]),
                _fmt_percent(rates["insertion"]),
                _fmt_percent(rates["deletion"]),
            ]
        )

    if not rows:
        print("\nNo model rows were produced.")
        return

    _print_table(TABLE_HEADERS, rows)
    _save_xlsx(rows, output_xlsx)
    print(f"\nSaved: {output_xlsx}")


if __name__ == "__main__":
    main()

