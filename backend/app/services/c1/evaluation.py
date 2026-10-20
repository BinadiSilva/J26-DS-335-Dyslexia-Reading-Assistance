"""
Evaluation of C1 against the human miscue labels shipped with the MPS dataset.

Ground truth
  Each record has `textAlignment` with one entry per reference word:
    c = read correctly, s = substitution, d = deletion (omitted)
  and entries whose promptTextWord is "<eps>" for insertions (label i).

What is measured (matches Table 3.4 of the proposal)
  * Word-level miscue detection: precision / recall / F1
    (positive class = word NOT read correctly, i.e. substitution or omission)
  * Count error: MAE of substitutions, omissions, insertions per recording
  * Reading accuracy and WER: MAE against the values implied by human labels
"""

import math


def ground_truth(record):
    labels, insertions = [], 0
    for a in record["textAlignment"]:
        if a["promptTextWord"] == "<eps>":
            insertions += 1
        else:
            labels.append(a["miscueLabel"])
    n = len(labels)
    subs, dels = labels.count("s"), labels.count("d")
    return {
        "labels": labels, "n": n, "substitutions": subs, "omissions": dels,
        "insertions": insertions,
        "accuracy": labels.count("c") / n * 100 if n else None,
        "wer": (subs + dels + insertions) / n if n else None,
    }


def labels_from_alignment(alignment):
    code = {"correct": "c", "substitution": "s", "deletion": "d"}
    return [code[a["operation"]] for a in alignment if a["reference"] is not None]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def evaluate_run(rows):
    """
    rows: list of dicts with keys
      id, gt (from ground_truth), pred_labels, pred_subs, pred_dels, pred_ins
    """
    tp = fp = fn = agree = total = 0
    per, abs_sub, abs_del, abs_ins, abs_acc, abs_wer = [], [], [], [], [], []
    pred_acc, gt_acc = [], []

    for r in rows:
        gt, pl = r["gt"], r["pred_labels"]
        if len(pl) != len(gt["labels"]):
            raise ValueError(f"{r['id']}: reference length mismatch "
                             f"({len(pl)} vs {len(gt['labels'])})")
        t = sum(1 for g, p in zip(gt["labels"], pl) if g != "c" and p != "c")
        f_p = sum(1 for g, p in zip(gt["labels"], pl) if g == "c" and p != "c")
        f_n = sum(1 for g, p in zip(gt["labels"], pl) if g != "c" and p == "c")
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        agree += sum(1 for g, p in zip(gt["labels"], pl) if (g == "c") == (p == "c"))
        total += len(pl)

        pn = len(pl)
        p_acc = pl.count("c") / pn * 100
        p_wer = (r["pred_subs"] + r["pred_dels"] + r["pred_ins"]) / pn
        abs_sub.append(abs(r["pred_subs"] - gt["substitutions"]))
        abs_del.append(abs(r["pred_dels"] - gt["omissions"]))
        abs_ins.append(abs(r["pred_ins"] - gt["insertions"]))
        abs_acc.append(abs(p_acc - gt["accuracy"]))
        abs_wer.append(abs(p_wer - gt["wer"]))
        pred_acc.append(p_acc); gt_acc.append(gt["accuracy"])
        p, rc, f1 = prf(t, f_p, f_n)
        per.append({"id": r["id"], "gt_acc": round(gt["accuracy"], 1),
                    "pred_acc": round(p_acc, 1),
                    "gt_s/d/i": f"{gt['substitutions']}/{gt['omissions']}/{gt['insertions']}",
                    "pred_s/d/i": f"{r['pred_subs']}/{r['pred_dels']}/{r['pred_ins']}",
                    "precision": round(p, 2), "recall": round(rc, 2), "f1": round(f1, 2)})

    mean = lambda v: sum(v) / len(v) if v else None
    p, rc, f1 = prf(tp, fp, fn)
    return {
        "n_recordings": len(rows),
        "precision": p, "recall": rc, "f1": f1,
        "word_level_agreement": agree / total if total else None,
        "mae_substitutions": mean(abs_sub), "mae_omissions": mean(abs_del),
        "mae_insertions": mean(abs_ins),
        "mae_accuracy_pct": mean(abs_acc), "mae_wer": mean(abs_wer),
        "accuracy_pearson_r": pearson(pred_acc, gt_acc),
        "per_recording": per,
    }
