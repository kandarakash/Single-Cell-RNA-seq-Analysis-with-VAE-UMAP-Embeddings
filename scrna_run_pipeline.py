"""
run_pipeline.py
---------------
Full end-to-end scRNA-seq analysis pipeline:
  1. Generate synthetic 68K × 33K dataset (AnnData .h5ad)
  2. Train scVI VAE → 30-dim latent representation
  3. Benchmark: PCA / t-SNE / UMAP / scVI silhouette scores
  4. Differential expression → 3 marker genes per cluster
  5. UMAP visualisation (clusters + cell types + batches)
  6. GPU vs CPU speedup measurement

Usage
-----
  python run_pipeline.py --data_dir data/processed --out_dir outputs
  python run_pipeline.py --skip_data_prep  # if .h5ad already exists
  python run_pipeline.py --quick_test      # 5K cells, 10 epochs
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    if not args.skip_data_prep:
        print("\n" + "═"*60)
        print("  STEP 1/6 — Synthetic scRNA-seq Dataset")
        print("═"*60)
        from data.prepare_dataset import generate_synthetic_dataset
        generate_synthetic_dataset(data_dir, n_cells=args.n_cells,
                                    n_genes=args.n_genes, seed=args.seed)

    # Load AnnData
    import anndata as ad
    adata = ad.read_h5ad(data_dir / "full_68k.h5ad")
    print(f"\nLoaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── Step 2: Train scVI ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 2/6 — scVI VAE Training")
    print("═"*60)
    from models.scvi_vae import train_scvi

    scvi_model, scvi_latent, gpu_time = train_scvi(
        adata,
        out_dir=out_dir / "scvi",
        n_latent=args.n_latent,
        n_hidden=256,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        batch_key="batch",
        seed=args.seed,
    )
    print(f"  GPU training time: {gpu_time:.2f} min  (target: <4 min for 68K)")

    # ── Step 3: Embedding benchmark ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 3/6 — Embedding Benchmark (PCA / t-SNE / UMAP / scVI)")
    print("═"*60)
    from benchmark.embedding_benchmark import run_benchmark

    bench_results = run_benchmark(
        adata, scvi_latent,
        out_dir=out_dir / "benchmark",
        seed=args.seed)

    # ── Step 4: UMAP on scVI latent ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 4/6 — UMAP 2D Projection of scVI Latent")
    print("═"*60)
    try:
        import umap as umap_lib
        print("  Running UMAP on 30-dim scVI latent → 2D...")
        reducer  = umap_lib.UMAP(n_components=2, n_neighbors=15,
                                  min_dist=0.1, random_state=args.seed)
        umap_2d  = reducer.fit_transform(scvi_latent)
        np.save(out_dir / "scvi_umap_2d.npy", umap_2d)
        print(f"  UMAP 2D shape: {umap_2d.shape}")
    except ImportError:
        print("  (umap-learn not installed, using PCA 2D as fallback)")
        from sklearn.decomposition import PCA
        umap_2d = PCA(n_components=2, random_state=args.seed).fit_transform(scvi_latent)
        np.save(out_dir / "scvi_umap_2d.npy", umap_2d)

    # ── Step 5: Differential expression ──────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 5/6 — Differential Expression (3 markers/cluster)")
    print("═"*60)
    import scipy.sparse as sp
    from evaluation.analysis import differential_expression

    X_raw      = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    gene_names = list(adata.var_names)
    cluster_labels = np.load(out_dir / "benchmark" / "cluster_labels_scvi.npy")

    marker_genes = differential_expression(
        X_raw, gene_names, cluster_labels, n_top_genes=3)

    n_with_markers = sum(1 for v in marker_genes.values() if v)
    avg_markers = np.mean([len(v) for v in marker_genes.values()])
    print(f"  Clusters with ≥1 marker: {n_with_markers}/{len(marker_genes)}")
    print(f"  Avg markers per cluster : {avg_markers:.1f}  (target: 3)")

    with open(out_dir / "marker_genes.json", "w") as f:
        json.dump(marker_genes, f, indent=2)

    # ── Step 5b: UMAP visualisation ───────────────────────────────────────
    from evaluation.analysis import plot_umap_clusters
    type_codes = adata.obs["cell_type"].astype("category").cat.codes.values
    batch_codes = adata.obs["batch"].astype("category").cat.codes.values
    plot_umap_clusters(umap_2d, cluster_labels, type_codes, batch_codes,
                        out_dir=out_dir / "plots")

    # ── Step 6: GPU vs CPU speedup ────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 6/6 — GPU vs CPU Speedup")
    print("═"*60)
    from evaluation.analysis import measure_gpu_cpu_speedup
    speedup_result = measure_gpu_cpu_speedup(adata, seed=args.seed)

    # ── Final summary ──────────────────────────────────────────────────────
    scvi_metrics = bench_results.get("scVI", {})
    pca_metrics  = bench_results.get("PCA",  {})
    comp         = bench_results.get("comparison", {})

    print("\n" + "═"*60)
    print("  PIPELINE COMPLETE")
    print("═"*60)
    print(f"  scVI silhouette  : {scvi_metrics.get('silhouette','N/A')}  (target: 0.74)")
    print(f"  PCA  silhouette  : {pca_metrics.get('silhouette','N/A')}   (target: 0.51)")
    print(f"  Improvement      : {comp.get('scvi_vs_pca_improvement_pct','N/A')}%  (target: 45%)")
    print(f"  scVI clusters    : {scvi_metrics.get('n_clusters','N/A')}  (target: 14)")
    print(f"  PCA  clusters    : {pca_metrics.get('n_clusters','N/A')}  (target: 9)")
    print(f"  Batch reduction  : {comp.get('batch_variation_reduction_pct','N/A')}%  (target: 62%)")
    print(f"  GPU speedup      : {speedup_result.get('speedup','N/A')}×  (target: 9.5×)")
    print(f"  Marker genes     : {avg_markers:.1f}/cluster  (target: 3)")
    print(f"\n  Outputs → {out_dir}")

    summary = {
        "scVI": scvi_metrics,
        "PCA":  pca_metrics,
        "comparison": comp,
        "speedup": speedup_result,
        "marker_gene_summary": {
            "n_clusters_with_markers": n_with_markers,
            "avg_markers_per_cluster": round(float(avg_markers), 2),
        },
    }
    with open(out_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scVI scRNA-seq Analysis Pipeline")
    parser.add_argument("--data_dir",       default="data/processed")
    parser.add_argument("--out_dir",        default="outputs")
    parser.add_argument("--n_cells",        type=int, default=68000)
    parser.add_argument("--n_genes",        type=int, default=33000)
    parser.add_argument("--n_latent",       type=int, default=30)
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--quick_test",     action="store_true",
                        help="Use 5K cells, 10 epochs for fast smoke test")
    args = parser.parse_args()

    if args.quick_test:
        args.n_cells  = 5000
        args.n_genes  = 5000
        args.epochs   = 10

    main(args)
