"""
Task 3: score detection and discrimination of injected ground truth

Reads injected_truth.csv (injected faults) and candidate labels.csv (discriminator output) and reports:
    - Detection recall - did each injected fault produce a candidate?
    - Confusion matrix - for detected injections, was failure or event correctly labeled?
Optionally --map_csv reports the cluster detector's recall for comparison.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score Task 3 injections")
    p.add_argument("--truth_csv", type=str, default="downstream_tasks/task3_fault_injection/outputs/injected_truth.csv")
    p.add_argument("--labels_csv", type=str, default="downstream_tasks/task2_failure_vs_event/outputs/candidate_labels.csv")
    p.add_argument("--map_csv", type=str, default="", help="optional map_freq_hot clusters_all.csv for cluster-detector recall")
    p.add_argument("--out_json", type=str, default="downstream_tasks/task3_fault_injection/outputs/score.json")
    return p.parse_args()


def injection_cells(row) -> set[tuple[int, int]]:
    sensors = [int(x) for x in str(row.sensor_indices).split(";") if x != ""]
    return {(t, s) for s in sensors for t in range(int(row.t0), int(row.t1))}


def cluster_cells(map_csv: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(map_csv)
    cells: set[tuple[int, int]] = set()
    for r in df.itertuples(index=False):
        t = int(r.time_idx)
        for idx in str(r.sensor_indices).split(";"):
            if idx != "":
                cells.add((t, int(idx)))
    return cells


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth_csv)
    labels = pd.read_csv(args.labels_csv)

    cand_cells = {(int(r.time_idx), int(r.sensor_index)) for r in labels.itertuples(index=False)}
    label_by_cell = {(int(r.time_idx), int(r.sensor_index)): r.label for r in labels.itertuples(index=False)}
    map_cells = cluster_cells(Path(args.map_csv)) if args.map_csv else None

    true_to_pred = {"failure": "sensor_failure", "event": "event"}
    per_injection = []
    confusion: dict[tuple[str, str], int] = {}
    det_counts = {"failure": [0, 0], "event": [0, 0]}    # [detected, total]
    map_counts = {"failure": [0, 0], "event": [0, 0]}

    for row in truth.itertuples(index=False):
        cells = injection_cells(row)
        detected = bool(cells & cand_cells)
        det_counts[row.kind][1] += 1
        det_counts[row.kind][0] += int(detected)

        map_detected = None
        if map_cells is not None:
            map_detected = bool(cells & map_cells)
            map_counts[row.kind][1] += 1
            map_counts[row.kind][0] += int(map_detected)

        predicted = None
        if detected:
            hit_labels = [label_by_cell[c] for c in cells if c in label_by_cell]
            if hit_labels:
                predicted = pd.Series(hit_labels).value_counts().idxmax()  # majority vote
                key = (true_to_pred[row.kind], predicted)
                confusion[key] = confusion.get(key, 0) + 1

        per_injection.append({
            "injection_id": int(row.injection_id), "kind": row.kind, "subtype": row.subtype,
            "detected": detected, "predicted": predicted,
            "map_detected": map_detected,
        })

    def recall(c):
        return (c[0] / c[1]) if c[1] else float("nan")

    print("=== Detection recall (activation pool) ===")
    for k in ("failure", "event"):
        print(f"  {k:8s}: {det_counts[k][0]}/{det_counts[k][1]}  ({recall(det_counts[k]):.0%})")

    if map_cells is not None:
        print("\n=== Cluster detector (map_freq_hot) recall ===  [D8: needs >=2 linked sensors; lone failures only caught via a chance-active neighbor]")
        for k in ("failure", "event"):
            print(f"  {k:8s}: {map_counts[k][0]}/{map_counts[k][1]}  ({recall(map_counts[k]):.0%})")

    print("\n=== Discrimination confusion (detected injections) ===")
    preds = ["sensor_failure", "event", "ambiguous"]
    header = "true \\ pred"
    print(f"  {header:16s}" + "".join(f"{p:16s}" for p in preds))
    for tk in ("sensor_failure", "event"):
        print(f"  {tk:16s}" + "".join(f"{confusion.get((tk, p), 0):<16d}" for p in preds))

    print("\n=== Per-subtype (detection + predicted label) ===")
    subtypes: dict[str, list] = {}
    for r in per_injection:
        subtypes.setdefault(r["subtype"], []).append(r)
    for st, rs in subtypes.items():
        det = sum(r["detected"] for r in rs)
        dist = {lab: sum(r["predicted"] == lab for r in rs) for lab in ("sensor_failure", "event", "ambiguous")}
        print(f"  {st:14s}: detected {det}/{len(rs)}  predicted -> {dist}")

    missed = [r["injection_id"] for r in per_injection if not r["detected"]]
    if missed:
        print(f"\n  missed (not detected at all): {missed}")

    out = {
        "detection_recall": {k: recall(det_counts[k]) for k in det_counts},
        "cluster_detector_recall": ({k: recall(map_counts[k]) for k in map_counts} if map_cells is not None else None),
        "confusion": {f"{a}->{b}": v for (a, b), v in confusion.items()},
        "per_injection": per_injection,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
