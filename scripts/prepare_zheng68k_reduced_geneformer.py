"""Prepare HVG-reduced Zheng68K data for Geneformer."""

from pathlib import Path

import numpy as np
import scanpy as sc

from prepare_geneformer_data import download_zheng68k_raw

REPO_ROOT = Path(__file__).resolve().parent.parent

FILES = [
    (
        "data/zheng68k/zheng68k_default_3000hvg.h5ad",
        "data/zheng68k/zheng68k_default_3000hvg_geneformer.h5ad",
    ),
    (
        "data/zheng68k/zheng68k_test_3000hvg.h5ad",
        "data/zheng68k/zheng68k_test_3000hvg_geneformer.h5ad",
    ),
]


def main():
    cache_dir = REPO_ROOT / "data" / "zheng68k" / "_raw_10x_cache"
    mtx_dir = download_zheng68k_raw(cache_dir)

    print(
        "  loading raw 10x matrix (this is ~5.9M barcodes x 32738 genes, takes a minute)..."
    )
    raw = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols")
    raw_barcode_to_row = {bc: i for i, bc in enumerate(raw.obs_names)}
    ensembl_by_symbol = dict(zip(raw.var_names, raw.var["gene_ids"]))
    raw_counts_per_cell = np.asarray(raw.X.sum(axis=1)).ravel()

    for src_rel, dst_rel in FILES:
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
                f"{n_matched}/{adata.n_obs} matched -- expected 100% (same cells as the "
                "full-gene file, which matched 100%); investigate before trusting n_counts here."
            )
        adata.obs["n_counts"] = raw_counts_per_cell[rows]

        ensembl_ids = [ensembl_by_symbol.get(g, "") for g in adata.var_names]
        n_unmapped = sum(1 for e in ensembl_ids if not e)
        print(
            f"    mapped {adata.n_vars - n_unmapped}/{adata.n_vars} genes to Ensembl IDs "
            f"({n_unmapped} unmapped -- dropped by the tokenizer)"
        )
        adata.var["ensembl_id"] = ensembl_ids

        adata.write_h5ad(dst)
        print(f"    wrote {dst}")


if __name__ == "__main__":
    main()
