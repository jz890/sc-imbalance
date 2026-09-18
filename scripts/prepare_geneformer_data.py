"""Prepare AnnData files for Geneformer tokenization."""

import argparse
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parent.parent
ZHENG68K_RAW_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_raw_gene_bc_matrices.tar.gz"
)

DATASET_FILES = {
    "zheng68k": [
        (
            "data/zheng68k/zheng68k_default.h5ad",
            "data/zheng68k/zheng68k_default_geneformer.h5ad",
        ),
        (
            "data/zheng68k/zheng68k_test.h5ad",
            "data/zheng68k/zheng68k_test_geneformer.h5ad",
        ),
    ],
    "ms": [
        ("data/ms/ms_default.h5ad", "data/ms/ms_default_geneformer.h5ad"),
        ("data/ms/ms_test.h5ad", "data/ms/ms_test_geneformer.h5ad"),
    ],
    "hpancreas": [
        ("data/hpancreas/demo_train.h5ad", "data/hpancreas/demo_train_geneformer.h5ad"),
        ("data/hpancreas/demo_test.h5ad", "data/hpancreas/demo_test_geneformer.h5ad"),
    ],
}
GENE_NAME_COL = {"zheng68k": None, "ms": "gene_name", "hpancreas": None}


OVERSAMPLED_FILES = {
    "ms_oversampled": (
        "data/ms/ms_oversampled_fixed.h5ad",
        "data/ms/ms_oversampled_fixed_geneformer.h5ad",
        "gene_name",
    ),
    "zheng68k_oversampled": (
        "data/zheng68k/zheng68k_oversampled_fixed.h5ad",
        "data/zheng68k/zheng68k_oversampled_fixed_geneformer.h5ad",
        None,
    ),
}


def download_zheng68k_raw(cache_dir: Path) -> Path:
    mtx_dir = cache_dir / "matrices_mex" / "hg19"
    if (mtx_dir / "matrix.mtx").exists():
        return mtx_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "raw_gene_bc_matrices.tar.gz"
    if not tar_path.exists():
        print(f"  downloading {ZHENG68K_RAW_URL} ...")
        subprocess.run(
            ["curl", "-sL", ZHENG68K_RAW_URL, "-o", str(tar_path)], check=True
        )
    print("  extracting...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(cache_dir)
    return mtx_dir


def prepare_zheng68k():
    cache_dir = REPO_ROOT / "data" / "zheng68k" / "_raw_10x_cache"
    mtx_dir = download_zheng68k_raw(cache_dir)

    print(
        "  loading raw 10x matrix (this is ~5.9M barcodes x 32738 genes, takes a minute)..."
    )
    raw = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols")

    raw_barcode_to_row = {bc: i for i, bc in enumerate(raw.obs_names)}
    ensembl_by_symbol = dict(zip(raw.var_names, raw.var["gene_ids"]))
    raw_counts_per_cell = np.asarray(raw.X.sum(axis=1)).ravel()

    for src_rel, dst_rel in DATASET_FILES["zheng68k"]:
        src, dst = REPO_ROOT / src_rel, REPO_ROOT / dst_rel
        print(f"  processing {src_rel} -> {dst_rel}")
        adata = sc.read_h5ad(src)

        rows = np.array([raw_barcode_to_row.get(bc, -1) for bc in adata.obs_names])
        n_matched = int((rows >= 0).sum())
        print(
            f"    matched {n_matched}/{adata.n_obs} cells to the raw matrix by exact barcode+channel"
        )
        if n_matched != adata.n_obs:
            raise RuntimeError(
                f"{n_matched}/{adata.n_obs} matched -- expected 100% (verified earlier); "
                "investigate before trusting n_counts here."
            )
        adata.obs["n_counts"] = raw_counts_per_cell[rows]

        ensembl_ids = [ensembl_by_symbol.get(g, "") for g in adata.var_names]
        n_unmapped = sum(1 for e in ensembl_ids if not e)
        print(
            f"    mapped {adata.n_vars - n_unmapped}/{adata.n_vars} genes to Ensembl IDs "
            f"({n_unmapped} unmapped -- these will be dropped by the tokenizer, expected to be small)"
        )
        adata.var["ensembl_id"] = ensembl_ids

        adata.write_h5ad(dst)
        print(f"    wrote {dst}")


def get_or_build_symbol_to_ensembl_map(symbols: list[str], cache_path: Path) -> dict:
    """Resolve gene identifiers and cache the lookup result."""
    cache = {}
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cache = dict(zip(cached_df["symbol"], cached_df["ensembl_id"]))

    missing = [s for s in symbols if s not in cache]
    if missing:
        import mygene

        print(
            f"    querying mygene.info for {len(missing)} gene symbols (one-time, cached after)..."
        )
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            missing, scopes="symbol", fields="ensembl.gene", species="human"
        )
        for r in results:
            symbol = r.get("query")
            ens = r.get("ensembl")
            if isinstance(ens, list):
                ens = ens[0]
            cache[symbol] = (ens or {}).get("gene", "") if ens else ""

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"symbol": list(cache.keys()), "ensembl_id": list(cache.values())}
        ).to_csv(cache_path, index=False)
        print(f"    cached mapping to {cache_path}")

    return cache


def process_file_via_expm1(
    src_rel: str, dst_rel: str, gene_name_col: str | None, cache_path: Path
):
    src, dst = REPO_ROOT / src_rel, REPO_ROOT / dst_rel
    print(f"  processing {src_rel} -> {dst_rel}")
    adata = sc.read_h5ad(src)

    X = adata.X
    pseudo_raw = np.expm1(X.toarray() if hasattr(X, "toarray") else np.asarray(X))
    adata.obs["n_counts"] = pseudo_raw.sum(axis=1)

    if gene_name_col and gene_name_col not in adata.var.columns:
        print(
            f"    WARNING: expected var column {gene_name_col!r} not found in this file "
            f"(columns: {list(adata.var.columns)}) -- falling back to var_names"
        )
        gene_name_col = None
    symbols = (
        adata.var[gene_name_col].tolist() if gene_name_col else adata.var_names.tolist()
    )

    mapping = get_or_build_symbol_to_ensembl_map(symbols, cache_path)
    ensembl_ids = [mapping.get(s, "") for s in symbols]
    n_unmapped = sum(1 for e in ensembl_ids if not e)
    print(
        f"    mapped {adata.n_vars - n_unmapped}/{adata.n_vars} genes to Ensembl IDs "
        f"({n_unmapped} unmapped -- these will be dropped by the tokenizer, expected to be small)"
    )
    adata.var["ensembl_id"] = ensembl_ids

    adata.write_h5ad(dst)
    print(f"    wrote {dst}")


def prepare_via_expm1(dataset: str):
    gene_name_col = GENE_NAME_COL[dataset]
    cache_path = REPO_ROOT / "data" / "gene_symbol_to_ensembl_cache.csv"
    for src_rel, dst_rel in DATASET_FILES[dataset]:
        process_file_via_expm1(src_rel, dst_rel, gene_name_col, cache_path)


def prepare_oversampled(name: str):
    cache_path = REPO_ROOT / "data" / "gene_symbol_to_ensembl_cache.csv"
    src_rel, dst_rel, gene_name_col = OVERSAMPLED_FILES[name]
    process_file_via_expm1(src_rel, dst_rel, gene_name_col, cache_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "zheng68k",
            "ms",
            "hpancreas",
            "ms_oversampled",
            "zheng68k_oversampled",
        ],
        default=None,
    )
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.dataset and not args.all:
        raise SystemExit("pass --dataset <name> or --all")

    if args.all:
        datasets = [
            "zheng68k",
            "ms",
            "hpancreas",
            "ms_oversampled",
            "zheng68k_oversampled",
        ]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(f"=== {dataset} ===")
        if dataset in OVERSAMPLED_FILES:
            prepare_oversampled(dataset)
        elif dataset == "zheng68k":
            prepare_zheng68k()
        else:
            prepare_via_expm1(dataset)


if __name__ == "__main__":
    main()
