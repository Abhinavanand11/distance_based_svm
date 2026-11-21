# Distance-based kernel SVM (Amaya-Tejera et al., 2024)

Reproduction of the paper's method (binary features -> Jaccard / Rogers-Tanimoto / Sokal-Michener
distance D -> kernel K = 1 - D -> `SVC(kernel="precomputed")` + iterative best-subset search) on the UCI
**Banknote Authentication** and **Rice (Cammeo/Osmancik)** datasets, plus a rigorous evaluation mode.

## Install & run
```bash
pip install -r requirements.txt
python main.py --dataset banknote --mode A --n-jobs 4      # paper reproduction
python main.py --dataset rice --mode both --n-jobs 4       # Mode A + Mode B
python main.py --dataset all --quick                       # reduced-grid smoke test
```
Datasets are downloaded to `data/` (or put `data_banknote_authentication.txt` /
`Rice_Cammeo_Osmancik.arff` there manually). Results go to `outputs/<dataset>/mode_A_*` and `mode_B_*`.
Useful flags: `--seed`, `--test-sizes 60 70`, `--c-values 10 50`, `--patience`, `--max-iter`,
`--skip-standard`, `--stratify-mode-a`, `--final-test-size 0.3`, `--select-by mean|best`, `--no-plots`.

Full grid cost: 40 (N, C) pairs x >=100 SVM fits x 7 kernels per dataset; Rice takes a while
(use `--n-jobs`). Same (N, C) uses the same stream of random splits for every kernel.

## Modes
* **Mode A (paper reproduction)** - bin edges fitted on the full data (as in the paper), plain random
  splits, N = test fraction (60% => 40% training), best split chosen by *its own test accuracy*.
  Reported numbers are maxima over many splits and are **optimistic**; `mean_split_acc` is reported next to them.
* **Mode B (independent)** - stratified untouched final test set (default 30%); binarizer fitted on the
  development part only; Algorithm 1 runs inside the development data with validation splits; (N, C) is chosen by
  mean validation accuracy; each selected model is scored **once** on the final test set. A diagnostic column
  (`diag_random_subset_test_acc_mean`) shows what random training subsets of the same size achieve (never used for selection).

## Files
`data_loader.py` loading/validation - `preprocessing.py` equal-width binarizer - `distance_metrics.py`
similarities & distances - `kernel.py` proposed + standard kernels - `proposed_svm.py` Algorithm 1 & grid -
`evaluation.py` metrics/tables/paper comparison - `visualizations.py` figures - `main.py` CLI.

## Notes & deviations
* Distances = 1 - similarity. The paper's Table 3 prints the Rogers-Tanimoto and Sokal-Michener *similarity*
  formulas under "D"; the text (1 - coefficient) is what is implemented. Hence K(x, y) equals the similarity coefficient.
* The paper says it used sklearn's implementations. The common scipy definition of `sokalmichener` was
  2(q+r)/(n+q+r), identical to Rogers-Tanimoto (newer scipy removed it). Here Sokal-Michener is (q+r)/n as specified.
* With one-hot binning every row has exactly one 1 per original feature, so Sokal-Michener K is an affine function of the
  linear kernel x.y; all three distances are monotone functions of p.
* Pseudo-code of Algorithm 1 never resets `wait`; the text does. The text is followed (reset on strict improvement).
* Rogers-Tanimoto / Jaccard kernels are not guaranteed positive semi-definite for general binary data; the
  minimum eigenvalue on a subsample is printed.
* The paper's Table 10 baseline (standard kernels paired with a "distance") is under-specified. Baselines here use
  the binarized features with the same splits; polynomial degree 3, gamma = 'scale'.
* 2-D MDS plots (classical MDS) are visualization only; SVMs use the full kernel matrix.
