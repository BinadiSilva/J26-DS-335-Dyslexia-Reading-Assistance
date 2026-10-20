"""Compare existing C1 passage-relative JSON alignments with MPS human miscue labels.

No Whisper inference, no modifications to existing code or results.
Run from the repository root:
    python backend/app/services/c1/evaluate_existing_outputs.py

This is NOT an ASR WER evaluation or diagnosis validation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from evaluation import ground_truth, labels_from_alignment, evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'datasets/c1/mps/selected_20.json')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'datasets/c1/test_audio')
    parser.add_argument('--save-csv', type=Path, default=None, help='Optional local evaluation CSV path')
    args = parser.parse_args()
    if not args.metadata.is_file():
        raise SystemExit(f'Metadata not found: {args.metadata}')
    if not args.outputs.is_dir():
        raise SystemExit(f'Output directory not found: {args.outputs}')
    records = json.loads(args.metadata.read_text(encoding='utf-8'))
    indexed = {r['audioID']: r for r in records}
    rows = []
    skipped = []
    paths = sorted(args.outputs.glob('*_output.json')) + sorted(args.outputs.glob('c1_session_output.json'))
    seen = set()
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        rid = data.get('recording_id')
        if rid in seen or rid not in indexed:
            continue
        seen.add(rid)
        gt = ground_truth(indexed[rid])
        alignment = data.get('alignment', [])
        if not alignment:
            skipped.append((rid, 'missing alignment')); continue
        predicted = labels_from_alignment(alignment)
        if len(predicted) != len(gt['labels']):
            skipped.append((rid, f'word-count mismatch {len(predicted)} vs {len(gt["labels"])}')); continue
        rows.append({
            'id': rid, 'gt': gt, 'pred_labels': predicted,
            'pred_subs': data['substitution_count'],
            'pred_dels': data['omission_count'],
            'pred_ins': data['insertion_count']
        })
    if not rows:
        raise SystemExit('No compatible stored outputs found; ' + str(skipped))
    result = evaluate_run(rows)
    print('C1 preliminary reading-miscue agreement vs MPS annotations')
    print('NOT Whisper ASR WER. Phonetic miscues and ASR errors affect these values.')
    print(f'Recordings evaluated: {result["n_recordings"]}')
    print(f'Precision: {result["precision"]:.3f}')
    print(f'Recall:    {result["recall"]:.3f}')
    print(f'F1 score:  {result["f1"]:.3f}')
    print(f'Word-label agreement: {result["word_level_agreement"]:.1%}')
    print(f'Accuracy MAE: {result["mae_accuracy_pct"]:.2f} percentage points')
    for per in result['per_recording']:
        print(f'  {per["id"]}: MPS {per["gt_acc"]:.1f}% / predicted {per["pred_acc"]:.1f}%')
    for rid,why in skipped:
        print(f'SKIPPED {rid}: {why}')
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp, fieldnames=list(result['per_recording'][0]))
            writer.writeheader(); writer.writerows(result['per_recording'])
        print('Saved local CSV:', args.save_csv)


if __name__ == '__main__':
    main()
