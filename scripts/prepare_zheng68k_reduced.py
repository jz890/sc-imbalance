"""Select highly variable genes for Zheng68K data."""

from pathlib import Path

import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SRC = REPO_ROOT / "data/zheng68k/zheng68k_default.h5ad"
TEST_SRC = REPO_ROOT / "data/zheng68k/zheng68k_test.h5ad"
TRAIN_DST = REPO_ROOT / "data/zheng68k/zheng68k_default_3000hvg.h5ad"
TEST_DST = REPO_ROOT / "data/zheng68k/zheng68k_test_3000hvg.h5ad"

N_TOP_GENES = 3000


def main():
    print(f"loading {TRAIN_SRC} ...")
    train = sc.read_h5ad(TRAIN_SRC)
    print(f"  train: {train.n_obs} cells x {train.n_vars} genes")

    print(f"loading {TEST_SRC} ...")
    test = sc.read_h5ad(TEST_SRC)
    print(f"  test: {test.n_obs} cells x {test.n_vars} genes")

    n_before = train.n_vars
    sc.pp.filter_genes(train, min_cells=10)
    print(
        f"  dropped {n_before - train.n_vars} genes expressed in <10 cells "
        f"({train.n_vars} remain)"
    )

    print(f"selecting top {N_TOP_GENES} highly variable genes from TRAIN only...")
    sc.pp.highly_variable_genes(train, n_top_genes=N_TOP_GENES, flavor="cell_ranger")
    hvg_genes = train.var_names[train.var["highly_variable"]]
    print(f"  selected {len(hvg_genes)} genes")

    train_reduced = train[:, hvg_genes].copy()
    test_reduced = test[:, hvg_genes].copy()

    print(
        f"writing {TRAIN_DST} ({train_reduced.n_obs} cells x {train_reduced.n_vars} genes)"
    )
    train_reduced.write_h5ad(TRAIN_DST)
    print(
        f"writing {TEST_DST} ({test_reduced.n_obs} cells x {test_reduced.n_vars} genes)"
    )
    test_reduced.write_h5ad(TEST_DST)

    print("done")


if __name__ == "__main__":
    main()
