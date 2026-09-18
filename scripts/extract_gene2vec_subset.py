"""Create a dataset-aligned scBERT Gene2Vec embedding matrix."""

import argparse
import json

import numpy as np
import scanpy as sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        required=True,
        help="Any h5ad with this dataset's gene panel (train or test -- same panel either way).",
    )
    parser.add_argument(
        "--gene-name-col",
        default=None,
        help="Set for MS (gene_name); omit for Zheng68k/hPancreas (var index IS the gene symbol).",
    )
    parser.add_argument("--gene2vec-npy", required=True)
    parser.add_argument("--panglao-var-names", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    genes = (
        adata.var[args.gene_name_col].tolist()
        if args.gene_name_col
        else adata.var.index.tolist()
    )

    with open(args.panglao_var_names) as f:
        ref_genes = json.load(f)
    ref_index = {g: i for i, g in enumerate(ref_genes)}

    gene2vec = np.load(args.gene2vec_npy)
    assert gene2vec.shape == (
        len(ref_genes),
        200,
    ), f"unexpected gene2vec shape {gene2vec.shape}"

    subset = np.zeros((len(genes), gene2vec.shape[1]), dtype=gene2vec.dtype)
    matched = 0
    for i, gene in enumerate(genes):
        j = ref_index.get(gene)
        if j is not None:
            subset[i] = gene2vec[j]
            matched += 1

    print(
        f"matched {matched}/{len(genes)} genes ({matched / len(genes):.1%}) to the reference gene2vec panel"
    )
    np.save(args.out, subset)
    print(f"wrote {args.out} shape={subset.shape}")


if __name__ == "__main__":
    main()
