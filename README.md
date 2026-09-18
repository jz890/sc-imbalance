# Rethinking Class Imbalance for Single-Cell Foundation Models: A Systematic Benchmark Across Architectures and Long-Tail Loss Functions

Jiahui Zhong and Zeyu Dong

## Abstract

Single-cell foundation models (scGPT, scBERT, and Geneformer) achieve
cell-type classification accuracy up to 97.5% in our experiments, yet this
aggregate accuracy can mask systematic failure on rare, often disease-relevant cell
populations that long-tail loss functions are widely assumed to address. We
present a systematic benchmark of six long-tail loss functions
(cross-entropy, weighted CE, class-balanced loss, focal loss, LDAM,
logit-adjusted softmax) across three architectures and three datasets
(Multiple Sclerosis, Zheng68K, human Pancreas), totaling 162 controlled
training runs (3 backbones × 3 datasets × 6 losses × 3 seeds). The gap between
overall accuracy, Macro-F1, and rare-class recall under plain cross-entropy is
consistent across all nine (architecture, dataset) settings, driven by dataset
structure rather than pretraining. Rare-class failure itself splits into two
regimes with distinct embedding-geometry signatures, visible before any loss is
chosen: some classes are recoverable by the right loss, while others retain
linear separability yet are absorbed into unrelated classes' neighborhoods under
every evaluated loss and architecture. Among the recoverable classes, the
efficacy of reweighting is predicted by a class's absolute training-set size,
rather than its share of the dataset or the dataset's overall imbalance ratio.
Class-balanced loss and LDAM are the most consistent choices across all nine
settings, while logit adjustment trades rare-class precision for recall rather
than improving both. Our results give both a reusable benchmark and
mechanism-grounded practical guidelines for combining foundation models with
imbalanced biological data.

## Models

The benchmark evaluates three pretrained single-cell foundation models:

- [scGPT](https://github.com/bowang-lab/scGPT), which models discretized gene
  expression values with a transformer.
- [scBERT](https://github.com/TencentAILabHealthcare/scBERT), which uses a
  Performer-based architecture over a fixed gene panel.
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer), which represents
  each cell by ranked gene-expression tokens.

## Data and Preprocessing

The datasets and pretrained models used in this study can be obtained from the
sources cited in README and the manuscript. Set the
corresponding local paths in the YAML configuration files before training.

| Dataset | Source |
| --- | --- |
| Multiple Sclerosis | [Schirmer et al. (2019)](https://doi.org/10.1038/s41586-019-1404-z) |
| Zheng68K | [Zheng et al. (2017)](https://doi.org/10.1038/ncomms14049) |
| Human Pancreas | [Baron et al. (2016)](https://doi.org/10.1016/j.cels.2016.08.011) |

Dataset-specific preprocessing and fixed train/test partitions are described in
the manuscript and implemented in `scripts/`. The static PanglaoDB gene-panel
reference used for scBERT preprocessing is at `resources/panglao_var_names.json`.

## Installation

Experiments were run with Python 3.10 and PyTorch 2.3.0 on Linux/CUDA.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install the required backbone packages:

```bash
# scGPT
pip install --no-deps scgpt==0.2.4
pip install flash-attn --no-build-isolation

# scBERT
git clone https://github.com/TencentAILabHealthcare/scBERT.git

# Geneformer
git clone https://huggingface.co/ctheodoris/Geneformer.git
pip install --no-deps ./Geneformer
```

For Geneformer, obtain the `Geneformer-V1-10M` checkpoint from the
[official release](https://huggingface.co/ctheodoris/Geneformer) and set
`pretrained_model_dir` in the relevant YAML file.

## Reproduction

The canonical 162-run benchmark is specified in
`experiments/canonical_experiment_matrix.json`. Each YAML file in `configs/`
defines one backbone-dataset setting. Before launching a run, set the dataset
and pretrained-model paths in the selected configuration.

The following commands illustrate MS experiments with LDAM and seed 0:

```bash
python scripts/train_scgpt.py --config configs/scgpt_ms.yaml --loss ldam --seed 0
python scripts/train_scbert.py --config configs/scbert_ms.yaml --loss ldam --seed 0
python scripts/train_geneformer.py --config configs/geneformer_ms.yaml --loss ldam --seed 0
```

To reproduce the complete benchmark, execute each backbone-dataset
configuration for all six losses (`cross_entropy`, `weighted_ce`,
`class_balanced`, `focal`, `ldam`, and `logit_adjusted`) and seeds 0, 1, and
2, as defined by the experiment matrix.

## Repository Layout

- `configs/`: nine backbone-dataset configurations.
- `experiments/`: canonical experiment definition.
- `resources/`: static preprocessing resources.
- `scripts/`: training, loss functions, and preprocessing code.

## Citation

If you use this code, please cite the associated manuscript:

```bibtex
@misc{zhong2026rethinking,
  title={Rethinking Class Imbalance for Single-Cell Foundation Models: A Systematic Benchmark Across Architectures and Long-Tail Loss Functions},
  author={Zhong, Jiahui and Dong, Zeyu},
  year={2026},
  note={Manuscript}
}
```
