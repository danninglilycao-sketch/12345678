> **Start here:** Download and extract `ECE1512_Project_A_GitHub.zip`, then open the extracted `ECE1512_Project_A_GitHub/` folder. The archive preserves the complete project directory structure. Root-level source files, the executed notebook, final comparison CSV and selected figures are also available for browsing.

# ECE1512 Project A: ABMIL with Feature Perturbation and Consistency

Whole-slide classification on CAMELYON16 (C16), CAMELYON17 (C17), and BRACS using pre-extracted patch features. The implemented research extension adds masked Gaussian feature noise and a symmetric KL prediction-consistency loss to ABMIL. A second, unimplemented proposal explores multi-branch attention with a diversity penalty.

## Scope and provenance

The baseline notebook, model helpers, dataset code and configs derive from the supplied course project package completed for this group. `research_extension.py` is the experiment implementation used by the project. Original baseline artifacts are preserved in `results/`; completed extension artifacts are preserved in `results_extension/consistency_v1/`. Method figures are in `figures/method/`.

The historical experiment implementation is intentionally preserved byte-for-byte. Its experiment manifest includes a hash of that file, and editing it would invalidate completed-run reuse. Additional inspection and plotting utilities are separate under `tools/`.

## Install

Use a Python environment compatible with your platform and install PyTorch for your hardware, then:

```bash
python -m pip install -r requirements.txt
```

Original result summaries record the actual Python/PyTorch versions and devices. C16/C17 extension summaries recorded Python 3.13.16 and PyTorch 2.11.0+cu130 on Tesla T4. The final BRACS execution recorded PyTorch 2.11.0+cpu; earlier completed BRACS configurations were reused, so that final summary does not prove all configurations ran on CPU. Current pip package resolution is not an exact lockfile for those historical environments.

## Obtain data

Download the course features from https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature or https://zenodo.org/records/23130543 and arrange them as:

```text
feature/C16/patch_feats_pretrain_medical_ssl.h5
feature/C17/patch_feats_pretrain_medical_ssl.h5
feature/BRACS/patch_feats_pretrain_medical_ssl.h5
```

Raw features and WSIs are external data, not included in Git. Metadata is in `dataset_csv/`. Follow the source datasets' access and usage terms. To reproduce optional C16 attention visualizations, obtain `test_067.tif`, `test_068.tif`, and `test_068.xml` and put them under `slide/`.

## Baseline

The executed cloud notebook is `notebooks/ECE1512_ProjectA_Student.ipynb`, preserved with its recorded outputs. It includes historical recovery cells; do not blindly Run All, since some cells back up incomplete folders or start training. For a baseline reference, open `notebooks/Baseline_Reference.ipynb`, set `PROJECT_DIR` to the repository root, and choose a dataset. The reference notebook contains Colab-specific setup/download cells; skip or adapt them outside Colab. Run the training/evaluation cells for the selected dataset. The standalone architecture/evaluation helpers are in `baseline_core.py`.

Preserve the submitted `results/` before launching a new baseline run, because the reference notebook writes there. The extension reads the saved baseline splits, including slide ordering.

## Extension: fresh experiment

Run from the repository root. Start with **a new run name** so original results are preserved:

```bash
python -u research_extension.py --project . --run reproduction_01 --datasets C16 --sigmas 0.01 0.05 --weights 0.1 1.0
python -u research_extension.py --project . --run reproduction_01 --datasets C17 --sigmas 0.01 0.05 --weights 0.1 1.0
python -u research_extension.py --project . --run reproduction_01 --datasets BRACS --sigmas 0.01 0.05 --weights 0.1 1.0
```

The historical runner automatically uses CUDA when available and otherwise CPU. It does not automatically select Apple MPS. Each dataset searches four configurations for 30 epochs each. Model selection uses validation macro-AUROC + macro-F1. Testing uses clean inputs after selection. Both prediction branches receive gradients. Classifier/model capacity is unchanged, but the extension uses two forward passes per batch and a larger hyperparameter-search budget than the baseline.

**Interrupted runs:** fully completed configurations with `selection.json` are reused. The original runner does not save optimizer/RNG state for epoch-level resumption. Do not simply change `exist_ok=False` to overwrite an incomplete folder. Preserve it separately after confirming no training process is active. Do not change manifest hashes to bypass a settings mismatch. Historical absolute paths in completed-run manifests may differ from a new checkout; use a new run name for reproduction.

For Colab, reading a large HDF5 directly through Drive caused I/O errors in this project. BRACS was completed with a local feature cache and results saved back to Drive. The actual launcher is preserved at `results_extension/consistency_v1/BRACS/run_with_local_features.py`.

## Verify and summarize saved results (no GPU or features required)

```bash
python tools/check_results.py --project .
python tools/report_results.py --project .
```

`check_results.py` verifies matching saved baseline/extension splits, disjoint slide IDs, the presence of result/checkpoint files, and 30 recorded epochs for each submitted configuration. It does not deserialize checkpoint files or establish patient-level separation, numerical checkpoint correctness, or independent reproducibility.

`report_results.py` reads the final metrics/history files to make a comparison table and plots without rerunning test evaluation. The sign of the accuracy delta is in percentage points; other deltas use raw metric units. AUC differences should be computed from original precision, not rounded labels.

## Synthetic implementation checks

```bash
python test_extension.py
```

These checks use synthetic data only: noise masking, symmetric KL, both-branch gradients, padding invariance, optimization, and a small train/select/test pipeline. They are not real-data reproduction.

## Interpretation and limitations

The reported extension improved all listed C16 test metrics, produced mixed changes on C17, and reduced BRACS accuracy/macro-F1 while leaving AUC approximately unchanged at displayed precision. There is one seed (42), no significance claim, and no perturbation-only (lambda=0) ablation. Gaussian feature noise is not a direct test of staining/scanner robustness. Attention visualizations are qualitative importance maps, not patch tumor probabilities or validated causal explanations.

See `docs/REPRODUCIBILITY_CHECKLIST.md`, `docs/EXPERIMENTS.md`, and the per-dataset metrics/manifests for details. The final group report and its repository link should be included once the group has finalized them.

## References and attribution

- Ilse, Tomczak, and Welling. Attention-based Deep Multiple Instance Learning. ICML 2018. https://proceedings.mlr.press/v80/ilse18a.html
- Lu et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering, 2021. https://doi.org/10.1038/s41551-020-00682-w
- The unimplemented multi-branch proposal is motivated by attention pooling and parallel attention branches, but does not reproduce CLAM's clustering objective. No novelty claim is made.

No new license is asserted over course-provided material or third-party datasets. Preserve their applicable attribution and terms.

## Archived results and validation

All three datasets include four completed extension configurations, each with 30 history rows, a selected checkpoint and a selection record. Their final test metrics, original manifests, saved splits, BRACS local-feature launcher, and report figures are included. Interrupted backup folders were excluded; see `docs/IMPORT_NOTES.md`. The original files in Downloads were left unchanged.

All three result bundles passed file, split-equality, slide-disjointness, history-length, validation-score and selected-candidate checks. The original extension source SHA-256 matches all three experiment manifests. Plots were regenerated from the saved JSON/CSV files without retraining. These checks do not establish independent real-data reproduction.

| Dataset | Method | Accuracy (%) | Macro-AUROC | Macro-F1 | Test CE |
|---|---|---:|---:|---:|---:|
| C16 | ABMIL | 82.95 | 0.8704 | 0.8159 | 0.4165 |
| C16 | Extension | 88.37 | 0.9227 | 0.8714 | 0.2700 |
| C17 | ABMIL | 79.00 | 0.8351 | 0.5258 | 0.7032 |
| C17 | Extension | 80.50 | 0.8260 | 0.5759 | 0.7532 |
| BRACS | ABMIL | 47.67 | 0.8155 | 0.4077 | 1.3560 |
| BRACS | Extension | 45.35 | 0.8155 | 0.3668 | 1.3731 |
