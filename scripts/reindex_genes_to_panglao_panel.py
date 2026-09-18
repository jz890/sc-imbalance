"""Reindex a dataset to the scBERT reference gene panel."""

import argparse
import json

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def reindex(
    adata: ad.AnnData, gene_name_col: str | None, ref_genes: list[str]
) -> ad.AnnData:
    obj_genes = (
        adata.var[gene_name_col].tolist() if gene_name_col else adata.var.index.tolist()
    )
    obj_index = {g: i for i, g in enumerate(obj_genes)}

    X_src = adata.X
    if not sparse.issparse(X_src):
        X_src = sparse.csr_matrix(X_src)
    X_src = X_src.tocsc()

    n_obs = adata.n_obs
    n_ref = len(ref_genes)
    cols = []
    matched = 0
    for gene in ref_genes:
        j = obj_index.get(gene)
        if j is None:
            cols.append(sparse.csc_matrix((n_obs, 1), dtype=X_src.dtype))
        else:
            cols.append(X_src[:, j])
            matched += 1
    X_new = sparse.hstack(cols, format="csr")
    print(
        f"  matched {matched}/{n_ref} reference genes ({matched / n_ref:.1%}); "
        f"{len(obj_genes) - matched} of our {len(obj_genes)} genes had no reference match (dropped)"
    )

    new_adata = ad.AnnData(X=X_new, obs=adata.obs.copy())
    new_adata.var_names = ref_genes
    return new_adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-train", required=True)
    parser.add_argument("--in-test", required=True)
    parser.add_argument(
        "--gene-name-col",
        default=None,
        help="Set for MS (gene_name); omit for Zheng68k/hPancreas (var index IS the gene symbol).",
    )
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--panglao-var-names", required=True)
    args = parser.parse_args()

    with open(args.panglao_var_names) as f:
        ref_genes = json.load(f)
    print(f"reference panel: {len(ref_genes)} genes")

    for in_path, out_path, tag in [
        (args.in_train, args.out_train, "train"),
        (args.in_test, args.out_test, "test"),
    ]:
        print(f"[{tag}] loading {in_path}")
        adata = sc.read_h5ad(in_path)
        new_adata = reindex(adata, args.gene_name_col, ref_genes)
        new_adata.write(out_path)
        print(f"[{tag}] wrote {out_path} shape={new_adata.shape}")


if __name__ == "__main__":
    main()
