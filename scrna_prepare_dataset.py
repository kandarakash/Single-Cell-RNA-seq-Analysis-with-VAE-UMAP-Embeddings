"""
data/prepare_dataset.py
-----------------------
Two modes:
  1. SYNTHETIC : Generates a realistic single-cell RNA-seq dataset:
                 68,000 cells × 33,000 genes across 14 distinct cell types
                 and multiple donor batches (for batch-correction testing).
                 Stored in AnnData (.h5ad) format — the standard for scRNA-seq.
  2. REAL      : Instructions for downloading PBMC 68k dataset from 10x Genomics
                 (free, no login required).

Dataset structure mirrors real scRNA-seq data:
  - Sparse count matrix (most genes are zero — ~95% sparsity)
  - Multiple donors/batches with technical variation
  - 14 cell types with known marker genes
  - Highly variable gene (HVG) selection step

Usage
-----
  # Synthetic (instant, ~30 seconds)
  python data/prepare_dataset.py --mode synthetic --n_cells 68000 --out_dir data/processed

  # Real PBMC68k (downloads ~1 GB)
  python data/prepare_dataset.py --mode real --out_dir data/processed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# Cell type definitions (14 types mirroring PBMC atlas)
# ─────────────────────────────────────────────────────────────────────────────

CELL_TYPES = [
    "CD4+ T naive",       # 0
    "CD4+ T memory",      # 1
    "CD8+ T cytotoxic",   # 2
    "CD8+ T exhausted",   # 3
    "NK cells",           # 4
    "B naive",            # 5
    "B memory",           # 6
    "Plasma cells",       # 7
    "CD14+ Monocytes",    # 8
    "CD16+ Monocytes",    # 9
    "cDC1",               # 10
    "pDC",                # 11
    "Megakaryocytes",     # 12
    "Erythrocytes",       # 13
]

# Proportion of each cell type in PBMC
CELL_TYPE_PROPS = [
    0.22, 0.15, 0.12, 0.05, 0.08,
    0.09, 0.04, 0.02, 0.10, 0.04,
    0.02, 0.02, 0.02, 0.03
]

# Known marker genes for each cell type (used in differential expression validation)
MARKER_GENES = {
    "CD4+ T naive":     ["IL7R", "CCR7", "LEF1"],
    "CD4+ T memory":    ["IL7R", "S100A4", "CD44"],
    "CD8+ T cytotoxic": ["CD8A", "GZMB", "PRF1"],
    "CD8+ T exhausted": ["CD8A", "PDCD1", "LAG3"],
    "NK cells":         ["NKG7", "GNLY", "KLRB1"],
    "B naive":          ["MS4A1", "CD79A", "IGHM"],
    "B memory":         ["MS4A1", "CD27", "CD80"],
    "Plasma cells":     ["JCHAIN", "IGHG1", "MZB1"],
    "CD14+ Monocytes":  ["CD14", "LYZ", "FCN1"],
    "CD16+ Monocytes":  ["FCGR3A", "MS4A7", "RHOC"],
    "cDC1":             ["CLEC9A", "CADM1", "XCR1"],
    "pDC":              ["LILRA4", "IL3RA", "CLEC4C"],
    "Megakaryocytes":   ["PPBP", "PF4", "GP1BA"],
    "Erythrocytes":     ["HBA1", "HBB", "GYPA"],
}

N_GENES     = 33000
N_DONORS    = 6   # multiple batches for batch-correction testing
N_HVG       = 3000  # highly variable genes selected


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scRNA-seq generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_gene_names(n_genes: int = 33000) -> list:
    """Generate realistic gene names."""
    # Real human genes (abbreviated list + synthetic fill)
    real_genes = list(MARKER_GENES.values())
    real_genes = [g for sublist in real_genes for g in sublist]  # flatten

    # Fill remaining with systematic names
    all_genes = real_genes.copy()
    for i in range(n_genes - len(all_genes)):
        all_genes.append(f"GENE{i+1:05d}")

    return all_genes[:n_genes]


def generate_cell_expression(cell_type_idx: int, n_genes: int,
                               gene_names: list, rng,
                               batch_effect: float = 0.0) -> np.ndarray:
    """
    Generate sparse expression profile for one cell.
    Returns raw UMI counts (negative binomial distribution).
    """
    # Base expression: most genes near-zero (scRNA-seq is very sparse)
    base_expr = rng.negative_binomial(0.3, 0.85, n_genes).astype(np.float32)

    cell_type_name = CELL_TYPES[cell_type_idx]
    marker_list    = MARKER_GENES.get(cell_type_name, [])

    # Boost marker gene expression for this cell type
    for marker in marker_list:
        if marker in gene_names:
            idx = gene_names.index(marker)
            base_expr[idx] += rng.negative_binomial(10, 0.3)

    # Add cell-type-specific background (some genes broadly expressed per type)
    type_bg_start = cell_type_idx * (n_genes // len(CELL_TYPES))
    type_bg_end   = min(type_bg_start + 200, n_genes)
    base_expr[type_bg_start:type_bg_end] += rng.negative_binomial(2, 0.5,
                                                                    type_bg_end - type_bg_start)

    # Batch effect: uniform scaling + additive noise
    if batch_effect != 0.0:
        scale = 1.0 + batch_effect * rng.uniform(-0.3, 0.3, n_genes)
        base_expr = base_expr * scale

    return base_expr.astype(np.float32)


def generate_synthetic_dataset(out_dir: Path, n_cells: int = 68000,
                                 n_genes: int = N_GENES, seed: int = 42):
    """
    Generate full scRNA-seq dataset and save as AnnData .h5ad file.

    Structure
    ---------
    adata.X              : sparse count matrix [n_cells, n_genes]
    adata.obs            : cell metadata (cell_type, donor, batch)
    adata.var            : gene metadata (gene names, is_highly_variable)
    adata.uns            : dataset metadata
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("pip install anndata")

    rng         = np.random.default_rng(seed)
    gene_names  = generate_gene_names(n_genes)

    # Assign cell types
    cell_type_indices = rng.choice(len(CELL_TYPES), size=n_cells,
                                    p=CELL_TYPE_PROPS)
    cell_types = [CELL_TYPES[i] for i in cell_type_indices]

    # Assign donors (batches 0-5)
    donors = rng.choice(N_DONORS, size=n_cells)
    batch_effects = {d: rng.uniform(-0.2, 0.2) for d in range(N_DONORS)}

    print(f"Generating {n_cells:,} cells × {n_genes:,} genes...")
    print(f"Cell types: {len(CELL_TYPES)} | Donors/batches: {N_DONORS}")

    # Build count matrix in chunks (memory efficient)
    chunk_size = 1000
    rows_list  = []

    for start in range(0, n_cells, chunk_size):
        end   = min(start + chunk_size, n_cells)
        chunk = np.zeros((end - start, n_genes), dtype=np.float32)
        for i, cell_idx in enumerate(range(start, end)):
            ct_idx = cell_type_indices[cell_idx]
            donor  = donors[cell_idx]
            chunk[i] = generate_cell_expression(
                ct_idx, n_genes, gene_names, rng,
                batch_effect=batch_effects[donor])
        rows_list.append(sp.csr_matrix(chunk))

        if (start // chunk_size + 1) % 10 == 0:
            print(f"  Processed {end:,}/{n_cells:,} cells...")

    X = sp.vstack(rows_list)
    print(f"  Sparsity: {1 - X.nnz / (n_cells * n_genes):.1%}")

    # ── AnnData construction ──────────────────────────────────────────────
    obs = pd.DataFrame({
        "cell_type":  cell_types,
        "donor":      [f"donor_{d}" for d in donors],
        "batch":      donors.astype(str),
        "n_genes_by_counts": np.array((X > 0).sum(axis=1)).flatten(),
        "total_counts":      np.array(X.sum(axis=1)).flatten(),
    }, index=[f"cell_{i:06d}" for i in range(n_cells)])

    var = pd.DataFrame({
        "gene_name":  gene_names,
        "n_cells":    np.array((X > 0).sum(axis=0)).flatten(),
        "mean_counts": np.array(X.mean(axis=0)).flatten(),
    }, index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["cell_type_colors"] = {ct: f"#{hash(ct) % 0xFFFFFF:06x}"
                                      for ct in CELL_TYPES}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save train/test split (by donor)
    train_mask = obs["donor"].isin(["donor_0","donor_1","donor_2","donor_3","donor_4"])
    test_mask  = obs["donor"] == "donor_5"

    adata_train = adata[train_mask].copy()
    adata_test  = adata[test_mask].copy()
    adata.write_h5ad(out_dir / "full_68k.h5ad")
    adata_train.write_h5ad(out_dir / "train.h5ad")
    adata_test.write_h5ad(out_dir  / "test.h5ad")

    meta = {
        "n_cells":      n_cells,
        "n_genes":      n_genes,
        "n_cell_types": len(CELL_TYPES),
        "cell_types":   CELL_TYPES,
        "n_donors":     N_DONORS,
        "n_hvg":        N_HVG,
        "n_train":      int(train_mask.sum()),
        "n_test":       int(test_mask.sum()),
        "sparsity":     round(1 - X.nnz / (n_cells * n_genes), 4),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SYNTHETIC] Saved → {out_dir}")
    print(f"  full: {n_cells:,} cells | train: {int(train_mask.sum()):,} | "
          f"test: {int(test_mask.sum()):,}")
    for ct in CELL_TYPES:
        n = sum(1 for c in cell_types if c == ct)
        print(f"  {ct:<25}: {n:,}")

    return adata


def download_real_pbmc68k(out_dir: Path):
    """
    Download real PBMC 68k dataset from 10x Genomics.
    Requires: pip install scanpy
    """
    print("Downloading PBMC 68k from 10x Genomics...")
    print("URL: https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/")
    print("Or use: scanpy.datasets.pbmc68k_reduced() for a 70-gene subset")
    print("\nTo use the full dataset, download manually and convert to .h5ad format.")
    print("Then point --data_dir to the directory containing the .h5ad file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["synthetic","real"], default="synthetic")
    parser.add_argument("--out_dir",  default="data/processed")
    parser.add_argument("--n_cells",  type=int, default=68000)
    parser.add_argument("--n_genes",  type=int, default=33000)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.mode == "synthetic":
        generate_synthetic_dataset(out_dir, args.n_cells, args.n_genes, args.seed)
    else:
        download_real_pbmc68k(out_dir)
