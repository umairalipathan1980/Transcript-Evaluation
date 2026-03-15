# Command: python eval_enhanced.py <reference_dir> <hypotheses_root_dir> <enhanced_root_dir> [output_xlsx]
"""
Batch evaluation comparing original and enhanced transcripts for all model folders.

Reads:
- reference transcripts from a CLI-provided reference directory
- original hypotheses from each subfolder in a CLI-provided hypotheses root
- enhanced transcripts from a CLI-provided enhanced root under "<model_name>_enhanced/"

Outputs:
- CLI table, row printed after each model evaluation
- Excel file with the same table
"""

import sys
from pathlib import Path

import jiwer
from openpyxl import Workbook
from rapidfuzz.distance import Levenshtein

DEFAULT_OUTPUT_XLSX = Path(__file__).resolve().parent / "output_general.xlsx"

TABLE_HEADERS = [
    "Model",
    "Original WER",
    "Enhanced WER",
    "Original Spelling",
    "Enhanced Spelling",
    "Original Substitution",
    "Enhanced Substitution",
    "Original Deletion",
    "Enhanced Deletion",
    "Original Insertion",
    "Enhanced Insertion",
]


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


def _evaluate_texts(reference_text, hypothesis_text):
    word_transform = _build_transform()
    return jiwer.process_words(
        reference_text,
        hypothesis_text,
        reference_transform=word_transform,
        hypothesis_transform=word_transform,
    )


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
            "deletion": 0.0,
            "insertion": 0.0,
        }
    return {
        "wer": (totals["subs"] + totals["dels"] + totals["ins"]) / ref_words,
        "spelling": totals["spelling_errors"] / ref_words,
        "substitution": totals["subs"] / ref_words,
        "deletion": totals["dels"] / ref_words,
        "insertion": totals["ins"] / ref_words,
    }


def _format_model_name(folder_name: str) -> str:
    if folder_name.startswith("WhisperX-"):
        return f"WhisperX ({folder_name.split('-', 1)[1]})"
    if folder_name.startswith("whisperx-"):
        return f"WhisperX ({folder_name.split('-', 1)[1]})"
    return folder_name


def _fmt_percent(value: float) -> str:
    return f"{value:.2%}"


def _evaluate_model(reference_dir: Path, original_dir: Path, enhanced_dir: Path):
    orig_totals = _init_totals()
    enh_totals = _init_totals()
    evaluated = 0

    for ref_file in sorted(reference_dir.glob("*.txt")):
        original_file = original_dir / ref_file.name
        enhanced_file = enhanced_dir / ref_file.name
        if not original_file.exists() or not enhanced_file.exists():
            continue

        ref_text = ref_file.read_text(encoding="utf-8")
        orig_text = original_file.read_text(encoding="utf-8")
        enh_text = enhanced_file.read_text(encoding="utf-8")

        orig_output = _evaluate_texts(ref_text, orig_text)
        enh_output = _evaluate_texts(ref_text, enh_text)
        _update_totals(orig_totals, orig_output)
        _update_totals(enh_totals, enh_output)
        evaluated += 1

    if evaluated == 0:
        return None

    orig_rates = _rates(orig_totals)
    enh_rates = _rates(enh_totals)
    return {
        "evaluated_files": evaluated,
        "original": orig_rates,
        "enhanced": enh_rates,
    }


def _build_row(model_name: str, metrics: dict):
    return [
        model_name,
        _fmt_percent(metrics["original"]["wer"]),
        _fmt_percent(metrics["enhanced"]["wer"]),
        _fmt_percent(metrics["original"]["spelling"]),
        _fmt_percent(metrics["enhanced"]["spelling"]),
        _fmt_percent(metrics["original"]["substitution"]),
        _fmt_percent(metrics["enhanced"]["substitution"]),
        _fmt_percent(metrics["original"]["deletion"]),
        _fmt_percent(metrics["enhanced"]["deletion"]),
        _fmt_percent(metrics["original"]["insertion"]),
        _fmt_percent(metrics["enhanced"]["insertion"]),
    ]


def _compute_col_widths(headers, rows):
    widths = [len(header) for header in headers]
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
    numeric_cols = set(range(1, len(headers)))
    widths = _compute_col_widths(headers, rows)
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(separator)
    print(_format_row(headers, widths))
    print(separator)
    for row in rows:
        print(_format_row(row, widths, numeric_cols=numeric_cols))
    print(separator)


def _save_xlsx(rows, output_xlsx: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "evaluation"
    ws.append(TABLE_HEADERS)
    for row in rows:
        ws.append(row)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)


def _average_enhancements(metrics_list):
    count = len(metrics_list)
    return {
        "wer": sum(m["original"]["wer"] - m["enhanced"]["wer"] for m in metrics_list) / count,
        "spelling": sum(m["original"]["spelling"] - m["enhanced"]["spelling"] for m in metrics_list) / count,
        "substitution": sum(m["original"]["substitution"] - m["enhanced"]["substitution"] for m in metrics_list) / count,
        "deletion": sum(m["original"]["deletion"] - m["enhanced"]["deletion"] for m in metrics_list) / count,
        "insertion": sum(m["original"]["insertion"] - m["enhanced"]["insertion"] for m in metrics_list) / count,
    }


def main():
    if len(sys.argv) not in (4, 5):
        print(
            "Usage: python eval_enhanced.py "
            "<reference_dir> <hypotheses_root_dir> <enhanced_root_dir> [output_xlsx]"
        )
        sys.exit(1)

    reference_dir = Path(sys.argv[1])
    hypothesis_root = Path(sys.argv[2])
    enhanced_root_dir = Path(sys.argv[3])
    output_xlsx = Path(sys.argv[4]) if len(sys.argv) == 5 else DEFAULT_OUTPUT_XLSX
    if not output_xlsx.is_absolute():
        output_xlsx = Path(__file__).resolve().parent / output_xlsx

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    if not hypothesis_root.exists():
        raise FileNotFoundError(f"Hypothesis root directory not found: {hypothesis_root}")
    if not enhanced_root_dir.exists():
        raise FileNotFoundError(f"Enhanced root directory not found: {enhanced_root_dir}")

    model_dirs = sorted([p for p in hypothesis_root.iterdir() if p.is_dir()])
    if not model_dirs:
        print(f"No model directories found in {hypothesis_root}")
        return

    rows = []
    metrics_list = []

    print(f"Reference directory: {reference_dir}")
    print(f"Hypothesis root:     {hypothesis_root}")
    print(f"Enhanced root:       {enhanced_root_dir}")
    print()

    for model_dir in model_dirs:
        enhanced_dir = enhanced_root_dir / f"{model_dir.name}_enhanced"
        if not enhanced_dir.exists():
            print(f"Skipping {model_dir.name}: missing {enhanced_dir}")
            continue

        metrics = _evaluate_model(reference_dir, model_dir, enhanced_dir)
        if metrics is None:
            print(f"Skipping {model_dir.name}: no matching reference/original/enhanced files")
            continue

        row = _build_row(_format_model_name(model_dir.name), metrics)
        rows.append(row)
        metrics_list.append(metrics)

    if not rows:
        print("\nNo model rows were produced.")
        return

    _print_table(TABLE_HEADERS, rows)

    avg = _average_enhancements(metrics_list)
    print()
    print("Average enhancement summary:")
    print(f"  Average WER enhancement:          {avg['wer']:+.2%}")
    print(f"  Average spelling enhancement:     {avg['spelling']:+.2%}")
    print(f"  Average substitution enhancement: {avg['substitution']:+.2%}")
    print(f"  Average deletion enhancement:     {avg['deletion']:+.2%}")
    print(f"  Average insertion enhancement:    {avg['insertion']:+.2%}")

    _save_xlsx(rows, output_xlsx)
    print(f"\nSaved: {output_xlsx}")


if __name__ == "__main__":
    main()

