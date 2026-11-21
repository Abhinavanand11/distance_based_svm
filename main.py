#!/usr/bin/env python
"""Distance-based kernel SVM (Amaya-Tejera et al., 2024) - reproduction + rigorous evaluation.

Examples
  python main.py --dataset banknote --mode A --n-jobs 4
  python main.py --dataset rice --mode both --n-jobs 4
  python main.py --dataset all --quick           # fast smoke test on a reduced grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import visualizations as viz
from data_loader import load_dataset
from distance_metrics import DISTANCES
from evaluation import (best_per_method, best_proposed, classification_metrics, comparison_table,
                        paper_comparison, save_table)
from kernel import STANDARD_KERNELS, kernel_matrix, scale_gamma
from preprocessing import EqualWidthBinarizer
from proposed_svm import (fit_precomputed_svm, predict_precomputed, random_subset_baseline,
                          run_grid, split_indices)

EXPECTED_BINARY_DIMS = {"banknote": 12, "rice": 21}       # for 3 bins


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["banknote", "rice", "all"], default="all")
    p.add_argument("--mode", choices=["A", "B", "both"], default="both",
                   help="A = paper reproduction, B = independent evaluation")
    p.add_argument("--bins", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-sizes", type=int, nargs="+", default=[60, 70, 80, 90], help="N in percent")
    p.add_argument("--c-values", type=float, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--patience", type=int, default=100, help="stop after this many non-improving iterations")
    p.add_argument("--max-iter", type=int, default=1000, help="hard iteration limit per configuration")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--final-test-size", type=float, default=0.3, help="Mode B: untouched final test fraction")
    p.add_argument("--select-by", choices=["mean", "best"], default="mean",
                   help="Mode B: pick (N, C) by mean or by best validation accuracy over the random splits")
    p.add_argument("--baseline-draws", type=int, default=30, help="Mode B diagnostic: random-subset draws")
    p.add_argument("--stratify-mode-a", action="store_true", help="stratified splits in Mode A (paper: plain random)")
    p.add_argument("--skip-standard", action="store_true", help="skip linear/RBF/poly/sigmoid baselines")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quick", action="store_true", help="reduced grid/iterations for a smoke test")
    args = p.parse_args()
    args.c_values = [int(c) if float(c).is_integer() else c for c in args.c_values]
    if args.quick:
        args.c_values, args.test_sizes = [10, 50, 100], [60, 80]
        args.patience, args.max_iter = min(args.patience, 20), min(args.max_iter, 100)
    return args


def methods(skip_standard):
    out = [dict(kernel="proposed", distance=d, name=d, method=f"proposed-{d}") for d in DISTANCES]
    if not skip_standard:
        out += [dict(kernel=k, distance="-", name=k, method=k) for k in STANDARD_KERNELS]
    return out


def report_dims(ds, Xb, bins, tag):
    print(f"[{ds.name}] {tag}: before binarization {ds.X.shape} -> after {Xb.shape}")
    exp = EXPECTED_BINARY_DIMS[ds.name]
    if bins == 3 and Xb.shape[1] != exp:
        print(f"  WARNING: paper reports {exp} binary features, got {Xb.shape[1]}")


def meta_for(ds, m):
    return dict(dataset=ds.name, kernel=m["kernel"], distance=m["distance"], method=m["method"])


# ------------------------------------------------------------------ Mode A
def run_mode_a(ds, args, out):
    print(f"\n===== {ds.name}: MODE A (paper reproduction; test-split selection => optimistic) =====")
    X, y = ds.X.to_numpy(float), ds.y
    # Paper protocol: bin edges come from the FULL dataset (documented leak; Mode B avoids it).
    binarizer = EqualWidthBinarizer(args.bins).fit(X, list(ds.X.columns))
    Xb = binarizer.transform(X)
    report_dims(ds, Xb, args.bins, "Mode A")
    if not args.no_plots:
        viz.plot_class_distribution(y, ds.class_names, out / "class_distribution.png", f"{ds.name}: class distribution")
        viz.plot_binary_matrix(Xb, y, binarizer.feature_names_out_, out / "binary_matrix.png", seed=args.seed)
        eig = viz.plot_kernel_heatmaps(Xb, y, out / "kernel_heatmaps.png", seed=args.seed)
        print("  min kernel eigenvalue on a 300-instance subsample:", {k: f"{v:.2e}" for k, v in eig.items()})
        viz.plot_mds(Xb, y, out / "mds_visualization.png", seed=args.seed)

    gamma, rows = scale_gamma(Xb), []
    for m in methods(args.skip_standard):
        K = kernel_matrix(Xb, Xb, m["name"], gamma)          # full n x n; sliced per split
        rows += run_grid(K, y, meta_for(ds, m), [n / 100 for n in args.test_sizes], args.c_values,
                         args.patience, args.max_iter, args.seed, args.stratify_mode_a, args.n_jobs)
        del K
    df = pd.DataFrame(rows)
    save_table(df, out / "grid_results.csv")
    save_table(comparison_table(df, "test_acc"), out / "comparison_table.csv")
    best = best_per_method(df, "test_acc")
    save_table(best, out / "best_per_method.csv")
    print("\nBest configuration per method (Mode A, best-split test accuracy):")
    print(best[["method", "test_size", "C", "train_acc", "test_acc", "mean_split_acc", "precision", "recall",
                "specificity", "f1"]].to_string(index=False, float_format="%.4f"))
    text = paper_comparison(df, ds.name)
    print("\n" + text)
    (out / "paper_comparison.txt").write_text(text + "\n")

    if not args.no_plots:
        items = [(f"{r['method']}\nN={r['test_size']:.0%} C={r['C']} acc={r['test_acc']:.3f}",
                  [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]) for _, r in best.iterrows() if r["kernel"] == "proposed"]
        viz.plot_confusion_matrices(items, ds.class_names, out / "confusion_matrices.png")
        viz.plot_accuracy_comparison(df, "test_acc", out / "accuracy_comparison.png",
                                     f"{ds.name} - Mode A best-split test accuracy (optimistic)")
        viz.plot_accuracy_vs_c(df, "test_acc", out / "accuracy_vs_C.png", f"{ds.name} - Mode A")
        viz.plot_accuracy_vs_test_size(df, "test_acc", out / "accuracy_vs_test_size.png", f"{ds.name} - Mode A")


# ------------------------------------------------------------------ Mode B
def run_mode_b(ds, args, out):
    print(f"\n===== {ds.name}: MODE B (independent evaluation, untouched final test set) =====")
    X, y = ds.X.to_numpy(float), ds.y
    rng = np.random.RandomState(args.seed)
    dev_idx, fin_idx = split_indices(y, args.final_test_size, rng, stratify=True)   # stratified
    y_dev, y_fin = y[dev_idx], y[fin_idx]
    # Preprocessing fitted on the development (training/validation) data only.
    binarizer = EqualWidthBinarizer(args.bins).fit(X[dev_idx], list(ds.X.columns))
    Xd, Xf = binarizer.transform(X[dev_idx]), binarizer.transform(X[fin_idx])
    report_dims(ds, Xd, args.bins, "Mode B (dev part)")
    print(f"  development set {Xd.shape}, untouched final test set {Xf.shape}")

    score_col = "val_mean_acc" if args.select_by == "mean" else "val_acc"
    gamma, grid_rows, final_rows = scale_gamma(Xd), [], []
    for m in methods(args.skip_standard):
        K_dev = kernel_matrix(Xd, Xd, m["name"], gamma)
        K_fin = kernel_matrix(Xf, Xd, m["name"], gamma)       # K(X_test, X_train)
        rows = run_grid(K_dev, y_dev, meta_for(ds, m), [n / 100 for n in args.test_sizes], args.c_values,
                        args.patience, args.max_iter, args.seed, True, args.n_jobs)
        for r in rows:                                         # these metrics are on VALIDATION splits
            r["val_acc"], r["val_mean_acc"] = r.pop("test_acc"), r.pop("mean_split_acc")
        sel = max(rows, key=lambda r: (r[score_col], r["val_acc"]))   # selection uses validation only
        tr = sel["_train_idx"]
        model = fit_precomputed_svm(K_dev, y_dev, tr, sel["C"])
        met = classification_metrics(y_fin, predict_precomputed(model, K_fin, np.arange(len(y_fin)), tr))
        b_mean, b_std = random_subset_baseline(K_dev, K_fin, y_dev, y_fin, len(tr), sel["C"],
                                               args.baseline_draws, args.seed)
        final_rows.append({**meta_for(ds, m), "test_size": sel["test_size"], "C": sel["C"], "n_train": len(tr),
                           "train_acc": sel["train_acc"], "val_acc": sel["val_acc"],
                           "val_mean_acc": sel["val_mean_acc"], "final_test_acc": met.pop("accuracy"), **met,
                           "diag_random_subset_test_acc_mean": b_mean, "diag_random_subset_test_acc_std": b_std})
        grid_rows += rows
        del K_dev, K_fin
    grid, final = pd.DataFrame(grid_rows), pd.DataFrame(final_rows)
    save_table(grid, out / "grid_results_validation.csv")
    save_table(comparison_table(grid, "val_acc"), out / "comparison_table_validation.csv")
    save_table(final, out / "final_test_results.csv")

    sel_prop = best_proposed(final.assign(test_acc=final["val_mean_acc" if args.select_by == "mean" else "val_acc"]),
                             "test_acc")
    print(f"\nFinal results on the untouched test set (each selected model evaluated once; selection by "
          f"validation '{score_col}'):")
    print(final[["method", "test_size", "C", "n_train", "val_acc", "val_mean_acc", "final_test_acc", "precision",
                 "recall", "specificity", "f1", "diag_random_subset_test_acc_mean"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"\nProposed kernel chosen by validation: {sel_prop['method']} (N={sel_prop['test_size']:.0%}, "
          f"C={sel_prop['C']}) -> untouched-test accuracy {sel_prop['final_test_acc']:.3f}")
    print("This number (not Mode A's) is the generalization estimate. 'diag_random_subset_test_acc_mean' shows "
          "what random training subsets of the same size achieve, i.e. whether best-subset selection helps.")

    if not args.no_plots:
        viz.plot_class_distribution(y_fin, ds.class_names, out / "final_test_class_distribution.png",
                                    f"{ds.name}: final test set")
        viz.plot_confusion_matrices(
            [(f"{r['method']}\nN={r['test_size']:.0%} C={r['C']} acc={r['final_test_acc']:.3f}",
              [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]) for _, r in final.iterrows() if r["kernel"] == "proposed"],
            ds.class_names, out / "confusion_matrices_final_test.png")
        viz.plot_accuracy_comparison(final, "final_test_acc", out / "accuracy_comparison_final_test.png",
                                     f"{ds.name} - Mode B untouched-test accuracy")
        viz.plot_accuracy_vs_c(grid, "val_acc", out / "accuracy_vs_C_validation.png", f"{ds.name} - Mode B (validation)")
        viz.plot_accuracy_vs_test_size(grid, "val_acc", out / "accuracy_vs_test_size_validation.png",
                                       f"{ds.name} - Mode B (validation)")


def main():
    args = parse_args()
    names = ["banknote", "rice"] if args.dataset == "all" else [args.dataset]
    for name in names:
        ds = load_dataset(name, args.data_dir)
        root = Path(args.output_dir) / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "run_config.json").write_text(json.dumps(vars(args), indent=2))
        if args.mode in ("A", "both"):
            run_mode_a(ds, args, root / "mode_A_paper_reproduction")
        if args.mode in ("B", "both"):
            run_mode_b(ds, args, root / "mode_B_independent")
    print(f"\nAll outputs written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
