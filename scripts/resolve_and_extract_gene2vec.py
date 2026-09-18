"""Resolve gene symbols and create a Gene2Vec subset."""

import argparse
import json
import time

import numpy as np
import requests
import scanpy as sc


def resolve_aliases_mygene(unmatched_genes, batch_size=200):
    """Resolve unmatched symbols to current HGNC symbols."""
    resolved = {}
    for i in range(0, len(unmatched_genes), batch_size):
        batch = unmatched_genes[i : i + batch_size]
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                data={
                    "q": ",".join(batch),
                    "scopes": "symbol,alias,other_names",
                    "species": "human",
                    "fields": "symbol",
                },
                timeout=30,
            )
            resp.raise_for_status()
            hits = resp.json()
        except Exception as e:
            print(f"  mygene.info batch query failed ({e}), skipping this batch")
            continue
        for hit in hits:
            q = hit.get("query")
            sym = hit.get("symbol")
            if q and sym and not hit.get("notfound"):
                resolved.setdefault(q, sym)
        time.sleep(0.2)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--gene-name-col", default=None)
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

    direct_matched = 0
    unmatched = []
    gene_to_ref_idx = {}
    for gene in genes:
        j = ref_index.get(gene)
        if j is not None:
            gene_to_ref_idx[gene] = j
            direct_matched += 1
        else:
            unmatched.append(gene)
    print(
        f"pass 1 (direct): {direct_matched}/{len(genes)} matched, "
        f"{len(unmatched)} unmatched -- resolving via mygene.info..."
    )

    alias_map = resolve_aliases_mygene(unmatched)
    alias_matched = 0
    still_unmatched = []
    for gene in unmatched:
        resolved_sym = alias_map.get(gene)
        j = ref_index.get(resolved_sym) if resolved_sym else None
        if j is not None:
            gene_to_ref_idx[gene] = j
            alias_matched += 1
        else:
            still_unmatched.append(gene)

    total_matched = direct_matched + alias_matched
    print(f"pass 2 (alias):  +{alias_matched} matched via mygene.info alias resolution")
    print(
        f"final: {total_matched}/{len(genes)} matched ({total_matched / len(genes):.1%})"
    )
    if still_unmatched:
        preview = still_unmatched[:20]
        print(
            f"still unmatched ({len(still_unmatched)}): {preview}"
            f"{'...' if len(still_unmatched) > 20 else ''}"
        )

    subset = np.zeros((len(genes), gene2vec.shape[1]), dtype=gene2vec.dtype)
    for i, gene in enumerate(genes):
        j = gene_to_ref_idx.get(gene)
        if j is not None:
            subset[i] = gene2vec[j]

    np.save(args.out, subset)
    print(f"wrote {args.out} shape={subset.shape}")


if __name__ == "__main__":
    main()
