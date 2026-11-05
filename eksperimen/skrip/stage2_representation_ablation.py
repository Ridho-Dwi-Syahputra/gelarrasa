"""Train-only representation ablation; no semantic targets or annotation archives.

Frozen embeddings are compared with identical eight-cluster KMeans settings.
All scores are geometry/stability diagnostics, never semantic accuracy.  A
grouped holdout refit is compared with the full-train partition only to measure
its sensitivity to removal of train groups; that comparison is not OOF accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

SEEDS = (11, 29, 47)
PCA_SEED = 11
K = 8
N_NEIGHBORS = 10
SAMPLE_SIZE = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def neighbors(x: np.ndarray, fit_idx: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    """All-row top-10 train neighbors, excluding ALL matching pixel hashes."""
    sim = x @ x[fit_idx].T
    sim[hashes[:, None] == hashes[fit_idx][None, :]] = -np.inf
    if np.any(np.isfinite(sim).sum(axis=1) < N_NEIGHBORS):
        raise ValueError("Insufficient non-duplicate train neighbors")
    # Stable complete sort makes ties deterministic at the 10th neighbor.
    order = np.argsort(-sim, axis=1, kind="stable")[:, :N_NEIGHBORS]
    return fit_idx[order]


def retention(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, :, None] == right[:, None, :]).any(axis=2).mean(axis=1)


def align(reference, labels, fit_idx):
    contingency = np.zeros((K, K), dtype=np.int64)
    np.add.at(contingency, (reference[fit_idx], labels[fit_idx]), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = np.empty(K, dtype=np.int64)
    mapping[cols] = rows
    return mapping[labels], mapping


def reduce_features(raw, fit_idx, components):
    if components is None:
        return raw, None, 1.0
    if components > min(len(fit_idx), raw.shape[1]):
        raise ValueError(f"PCA{components} exceeds available dimensions/samples")
    reducer = PCA(n_components=components, svd_solver="randomized", random_state=PCA_SEED)
    reducer.fit(raw[fit_idx])
    transformed = normalize(reducer.transform(raw)).astype(np.float32)
    return transformed, reducer, float(reducer.explained_variance_ratio_.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--features", type=Path)
    parser.add_argument("--prefix", default="resnet50")
    parser.add_argument("--skip-resampling", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.prefix):
        parser.error("prefix must contain only letters, digits, underscores or hyphens")
    root = args.project_root.resolve()
    feature_path = (args.features or root / "output/embeddings/stage1_v1/resnet50_features.npz").resolve()
    profile_path = root / "output/manifests/stage1_v1/image_profile.csv"
    group_path = root / "output/groups/stage1_v1/groups.csv"
    report = root / "reports/errors/stage2_20261105/ablation" / args.prefix
    output = root / "output/groups/stage2_repr_v1" / args.prefix
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_handle = (report / "run.log").open("w", encoding="utf-8")

    def log(message):
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(text, flush=True)
        log_handle.write(text + "\n")
        log_handle.flush()

    df = pd.read_csv(profile_path, dtype={"image_id": str, "key": str})
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_keys = archive["keys"].astype(str)
        values = archive["features"].astype(np.float32)
    if len(set(feature_keys)) != len(feature_keys) or not df.key.is_unique:
        raise ValueError("Nonunique feature/profile keys")
    key_to_row = {key: i for i, key in enumerate(feature_keys)}
    if set(key_to_row) != set(df.key):
        raise ValueError("Feature/profile populations differ")
    raw = values[[key_to_row[key] for key in df.key]]
    if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(np.linalg.norm(raw, axis=1) == 0):
        raise ValueError("Invalid feature vectors")
    raw = normalize(raw).astype(np.float32)
    train_idx = np.flatnonzero(df.split.eq("train"))
    test_idx = np.flatnonzero(df.split.eq("test"))
    fit_idx = df.loc[df.split.eq("train")].drop_duplicates("pixel_sha256").index.to_numpy()
    if len(train_idx) != 1920 or len(test_idx) != 480 or len(fit_idx) != 1919:
        raise ValueError("Unexpected stage1 population; review contract before rerun")
    hashes = df.pixel_sha256.to_numpy()
    sample_idx = np.sort(np.random.default_rng(PCA_SEED).choice(fit_idx, min(SAMPLE_SIZE, len(fit_idx)), replace=False))
    df.iloc[sample_idx][["key", "split", "pixel_sha256"]].to_csv(report / "silhouette_sample.csv", index=False)
    provenance = dict(
        prefix=args.prefix, raw_feature_dimensions=int(raw.shape[1]), feature_path=str(feature_path),
        feature_sha256=sha256(feature_path), profile_sha256=sha256(profile_path),
        script_sha256=sha256(Path(__file__)), python=platform.python_version(), sklearn=sklearn.__version__,
        numpy=np.__version__, fit_population="unique train pixel hashes only", train_unique=len(fit_idx),
        train_rows=len(train_idx), test_rows=len(test_idx), test_used_in_fit=False, annotation_archives_read=False,
        semantic_categories_assigned=False, seeds=SEEDS, pca_seed=PCA_SEED, k=K, n_init=10, max_iter=300,
        pca_components=[None, 64, 128, 256], raw_l2_normalization=True, post_pca_l2_normalization=True,
        neighborhood_metric="cosine", neighborhood_topk=N_NEIGHBORS,
        neighbor_duplicate_policy="exclude every matching pixel hash including train-test duplicates",
        silhouette_metric="Euclidean on unit vectors", silhouette_sample_seed=PCA_SEED,
        silhouette_sample_size=len(sample_idx), thread_limit=4,
        selection_policy="No semantic winner selected; all metrics are unlabeled diagnostics",
        no_pca_retention_note="No-PCA kNN retention is exactly 1 by definition, not evidence of accuracy",
        holdout_note="One train candidate-group split; compare refit with full-train partition for sensitivity only; not OOF accuracy",
    )
    write_json(report / "provenance.json", provenance)
    resample_fit = resample_holdout = None
    if not args.skip_resampling:
        groups = pd.read_csv(group_path, dtype={"key": str}).set_index("key").loc[df.key, "candidate_group"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261105)
        a, b = next(splitter.split(raw[fit_idx], groups=groups[fit_idx]))
        resample_fit, resample_holdout = fit_idx[a], fit_idx[b]
        assert not set(groups[resample_fit]) & set(groups[resample_holdout])
        split_manifest = df.iloc[fit_idx][["key", "split", "pixel_sha256"]].copy()
        split_manifest["candidate_group"] = groups[fit_idx]
        split_manifest["refit_role"] = np.where(np.isin(fit_idx, resample_fit), "refit_train", "heldout_train")
        split_manifest.to_csv(report / "group_resampling_split.csv", index=False)
        provenance.update(group_sha256=sha256(group_path), resample_fit_rows=len(a), resample_holdout_rows=len(b))
    all_metrics, pairwise_rows, cluster_size_rows = [], [], []
    branch_assignments = df[["key", "split", "image_id"]].copy()
    summaries = []
    with threadpool_limits(limits=4):
        original_neighbors = neighbors(raw, fit_idx, hashes)
        original_sample_distances = pairwise_distances(raw[sample_idx], metric="euclidean")
        np.fill_diagonal(original_sample_distances, 0.)
        for components in (None, 64, 128, 256):
            branch = "nopca" if components is None else f"pca{components}"
            branch_start = time.time()
            x, reducer, variance = reduce_features(raw, fit_idx, components)
            branch_neighbors = original_neighbors if components is None else neighbors(x, fit_idx, hashes)
            preserved_neighbors = retention(original_neighbors, branch_neighbors)
            own_distances = original_sample_distances if components is None else pairwise_distances(x[sample_idx])
            np.fill_diagonal(own_distances, 0.)
            models, labels = {}, {}
            for seed in SEEDS:
                model = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=seed).fit(x[fit_idx])
                lab = model.predict(x)
                models[seed], labels[seed] = model, lab
                sizes = np.bincount(lab[fit_idx], minlength=K)
                metric = dict(branch=branch, seed=seed, dimensions=x.shape[1], variance_retained=variance,
                    silhouette_own=float(silhouette_score(own_distances, lab[sample_idx], metric="precomputed")),
                    silhouette_original=float(silhouette_score(original_sample_distances, lab[sample_idx], metric="precomputed")),
                    knn10_retention_train=float(preserved_neighbors[fit_idx].mean()),
                    knn10_retention_test=float(preserved_neighbors[test_idx].mean()),
                    neighbor_cohesion_original_train=float((lab[fit_idx, None] == lab[original_neighbors[fit_idx]]).mean()),
                    neighbor_cohesion_original_test=float((lab[test_idx, None] == lab[original_neighbors[test_idx]]).mean()),
                    neighbor_cohesion_own_train=float((lab[fit_idx, None] == lab[branch_neighbors[fit_idx]]).mean()),
                    min_cluster=int(sizes.min()), max_cluster=int(sizes.max()), inertia=float(model.inertia_),
                    fit_iterations=int(model.n_iter_))
                all_metrics.append(metric)
                for cluster, count in enumerate(sizes):
                    cluster_size_rows.append(dict(branch=branch, seed=seed, split="train_unique", cluster=cluster, count=int(count)))
                    cluster_size_rows.append(dict(branch=branch, seed=seed, split="test", cluster=cluster, count=int((lab[test_idx] == cluster).sum())))
                log(f"{branch} seed{seed} own_silhouette={metric['silhouette_own']:.6f} original_silhouette={metric['silhouette_original']:.6f} retention={metric['knn10_retention_train']:.4f}")
            primary = labels[11]
            seed_ari = []
            for left, right in itertools.combinations(SEEDS, 2):
                ari = float(adjusted_rand_score(labels[left][fit_idx], labels[right][fit_idx]))
                seed_ari.append(ari)
                pairwise_rows.append(dict(branch=branch, seed_a=left, seed_b=right, ari_train=ari))
            aligned = [align(primary, labels[seed], fit_idx)[0] for seed in SEEDS]
            seed_agreement = (np.stack(aligned) == primary).mean(axis=0)
            distances = models[11].transform(x)
            sorted_distance = np.sort(distances, axis=1)
            margin = (sorted_distance[:, 1] - sorted_distance[:, 0]) / np.maximum(sorted_distance[:, 1], 1e-12)
            assigned = distances[np.arange(len(df)), primary]
            p95 = np.array([np.quantile(assigned[fit_idx][primary[fit_idx] == cluster], .95) for cluster in range(K)])
            per_row = df[["key", "split", "image_id", "pixel_sha256"]].copy()
            per_row["cluster_index"] = primary
            per_row["cluster_id"] = [f"cluster_{c:02d}" for c in primary]
            per_row["centroid_distance"] = assigned
            per_row["relative_margin"] = margin
            per_row["seed_agreement"] = seed_agreement
            per_row["outlying_distance"] = assigned > p95[primary]
            per_row["knn10_retention"] = preserved_neighbors
            per_row["neighbor_cohesion_original"] = (primary[:, None] == primary[original_neighbors]).mean(axis=1)
            per_row["neighbor_cohesion_own"] = (primary[:, None] == primary[branch_neighbors]).mean(axis=1)
            for seed in SEEDS:
                per_row[f"seed{seed}_cluster_unaligned"] = labels[seed]
            per_row["review_flag"] = (margin < .1) | (seed_agreement < .8) | per_row.outlying_distance
            per_row.to_csv(output / f"{branch}_assignments.csv", index=False)
            branch_assignments[branch] = primary
            joblib.dump(dict(reducer=reducer, kmeans=models[11], raw_l2_normalization=True,
                             post_pca_l2_normalization=reducer is not None,
                             feature_sha256=provenance["feature_sha256"], semantic_category_mapping=None),
                        output / f"{branch}_seed11_model.joblib")
            summary = dict(branch=branch, dimensions=x.shape[1], variance_retained=variance,
                           seed_ari_min=min(seed_ari), seed_ari_mean=float(np.mean(seed_ari)),
                           seed_agreement_train=float(seed_agreement[fit_idx].mean()),
                           seed_agreement_test=float(seed_agreement[test_idx].mean()),
                           review_train_rows=int(per_row.iloc[train_idx].review_flag.sum()),
                           review_test_rows=int(per_row.iloc[test_idx].review_flag.sum()))
            if resample_fit is not None:
                rx, rpca, _ = reduce_features(raw, resample_fit, components)
                rmodel = KMeans(n_clusters=K, n_init=10, random_state=11, max_iter=300).fit(rx[resample_fit])
                rlabels = rmodel.predict(rx)
                raligned, rmap = align(primary, rlabels, resample_fit)
                summary.update(group_removed_holdout_ari_vs_full=float(adjusted_rand_score(primary[resample_holdout], rlabels[resample_holdout])),
                               group_removed_holdout_agreement_vs_full=float((primary[resample_holdout] == raligned[resample_holdout]).mean()),
                               group_removed_test_ari_vs_full=float(adjusted_rand_score(primary[test_idx], rlabels[test_idx])),
                               group_removed_test_agreement_vs_full=float((primary[test_idx] == raligned[test_idx]).mean()))
                pd.DataFrame(dict(key=df.key, split=df.split, full_partition=primary, refit_partition_aligned=raligned,
                                  heldout_train=np.isin(np.arange(len(df)), resample_holdout))).to_csv(output / f"{branch}_group_resampling_assignments.csv", index=False)
            summary["seconds"] = time.time() - branch_start
            summaries.append(summary)
            pd.DataFrame(all_metrics).to_csv(report / "metrics_by_seed.csv", index=False)
            pd.DataFrame(summaries).to_csv(report / "stability_summary.csv", index=False)
            log(f"{branch} complete: seed ARI min={summary['seed_ari_min']:.6f}, elapsed={summary['seconds']:.1f}s")
    branch_assignments.to_csv(output / "all_primary_assignments.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(report / "seed_pairwise_ari.csv", index=False)
    pd.DataFrame(cluster_size_rows).to_csv(report / "cluster_sizes.csv", index=False)
    branch_ari = [dict(branch_a=a, branch_b=b, ari_train=float(adjusted_rand_score(branch_assignments[a].iloc[fit_idx], branch_assignments[b].iloc[fit_idx])))
                  for a, b in itertools.combinations(["nopca", "pca64", "pca128", "pca256"], 2)]
    pd.DataFrame(branch_ari).to_csv(report / "branch_pairwise_ari.csv", index=False)
    provenance.update(seconds=time.time()-started, completed=True,
                      fitting_rows_hash=hashlib.sha256("\n".join(df.key.iloc[fit_idx]).encode()).hexdigest())
    write_json(report / "provenance.json", provenance)
    (report / "README.md").write_text(
        "# Representation ablation (unlabeled diagnostics)\n\n"
        "PCA and KMeans fit only unique train pixel hashes; test is prediction-only. "
        "No annotation archive or semantic category mapping is read. No model is selected as semantically best.\n\n"
        "Four branches: normalized original features, PCA64, PCA128, PCA256. "
        "All use KMeans8, seeds 11/29/47, n_init=10. PCA random seed is 11. "
        "Primary per-row exports and persisted models use seed11.\n\n"
        "Silhouette uses one fixed 1000-row train sample, in both each branch space and the common original feature space. "
        "Own-space scores across representations are not directly comparable. "
        "Top-10 neighbors exclude all exact pixel duplicates; no-PCA retention=1 by definition. "
        "Neighbor cohesion is internal agreement, not accuracy.\n\n"
        "One 80/20 candidate-group train split refits PCA and KMeans; agreement against the full-train partition "
        "measures deletion sensitivity only. It is not independent semantic validation, because the reference "
        "full model saw the held-out rows. Test labels never enter fitting.\n\n"
        "Each joblib contains reducer (or None), seed11 KMeans, normalization flags, and input feature digest. "
        "Normalize raw features first, then apply reducer and normalize again when reducer is present.\n",
        encoding="utf-8")
    log(f"COMPLETE in {provenance['seconds']:.1f}s; no semantic accuracy measured")
    log_handle.close()


if __name__ == "__main__":
    main()
