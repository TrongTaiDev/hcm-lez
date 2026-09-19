from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes


HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent
OUT = HERE / "results"
W2VPCA_SCRIPT = SOURCES / "scripts" / "run_w2vpca_combined.py"
W2VPCA_RESULTS = SOURCES / "reanalysis_combined_timed"
ANCHOR_MIN_COUNT = 5
MAX_ANCHORS = 2000


def load_w2vpca_module():
                                                                                   
    spec = importlib.util.spec_from_file_location("w2vpca", W2VPCA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load canonical script: {W2VPCA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_model(corpus: list[list[str]], seed: int) -> Word2Vec:
                                                                   
    return Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
        negative=5,
        sample=1e-3,
        epochs=20,
        seed=seed,
        workers=1,
        sorted_vocab=1,
    )


def unit_rows(matrix: np.ndarray) -> np.ndarray:
                                                                              
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered a zero-norm embedding vector.")
    return matrix / norms


def select_anchors(
    before: Word2Vec, after: Word2Vec, excluded_tokens: set[str]
) -> tuple[list[str], pd.DataFrame]:
\
\
\
\
\
       
    shared = set(before.wv.key_to_index) & set(after.wv.key_to_index)
    rows = []
    for token in shared:
        if token in excluded_tokens:
            continue
        before_count = int(before.wv.get_vecattr(token, "count"))
        after_count = int(after.wv.get_vecattr(token, "count"))
        if before_count >= ANCHOR_MIN_COUNT and after_count >= ANCHOR_MIN_COUNT:
            rows.append({
                "token": token,
                "count_before": before_count,
                "count_after": after_count,
                "min_count": min(before_count, after_count),
                "total_count": before_count + after_count,
            })
    anchors = (
        pd.DataFrame(rows)
        .sort_values(["min_count", "total_count", "token"], ascending=[False, False, True])
        .head(MAX_ANCHORS)
        .reset_index(drop=True)
    )
    if len(anchors) < 100:
        raise ValueError(
            f"Only {len(anchors)} shared anchors meet min_count={ANCHOR_MIN_COUNT}; "
            "at least 100 are required for a stable Procrustes fit."
        )
    return anchors["token"].tolist(), anchors


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_seed(
    before: Word2Vec, after: Word2Vec, words: list[str], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    anchors, anchor_frame = select_anchors(before, after, set(words))
    before_anchor = unit_rows(np.vstack([before.wv[token] for token in anchors]))
    after_anchor = unit_rows(np.vstack([after.wv[token] for token in anchors]))
    rotation, alignment_scale = orthogonal_procrustes(before_anchor, after_anchor)

    rows = []
    for word in words:
        if word not in before.wv or word not in after.wv:
            raise ValueError(f"Target word missing from one period model: {word}")
        before_vector = before.wv[word] @ rotation
        after_vector = after.wv[word]
        similarity = 1 - cosine_distance(before_vector, after_vector)
        rows.append({
            "seed": seed,
            "word": word,
            "procrustes_cosine_similarity": similarity,
            "procrustes_cosine_distance": 1 - similarity,
        })
    anchor_frame.insert(0, "seed", seed)
    return pd.DataFrame(rows), anchor_frame, float(alignment_scale)


def descending_rank(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    return frame[column].rank(ascending=False, method="min").astype(int).rename(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = load_w2vpca_module()
    documents = canonical.load_documents()
    words = list(canonical.TARGET_WORDS)
    seeds = list(canonical.SEEDS)

    before_corpus = documents.loc[
        documents["analysis_period"] == "before", "tokens"
    ].tolist()
    after_corpus = documents.loc[
        documents["analysis_period"] == "after", "tokens"
    ].tolist()
    if not before_corpus or not after_corpus:
        raise ValueError("Both temporal corpora must contain at least one document.")

    all_scores: list[pd.DataFrame] = []
    all_anchors: list[pd.DataFrame] = []
    run_metadata = []
    for seed in seeds:
        before_model = train_model(before_corpus, seed)
        after_model = train_model(after_corpus, seed)
        scores, anchors, alignment_scale = score_seed(before_model, after_model, words, seed)
        all_scores.append(scores)
        all_anchors.append(anchors)
        run_metadata.append({
            "seed": seed,
            "vocabulary_before": len(before_model.wv),
            "vocabulary_after": len(after_model.wv),
            "n_anchors": len(anchors),
            "alignment_scale": alignment_scale,
        })

    by_seed = pd.concat(all_scores, ignore_index=True)
    anchor_rows = pd.concat(all_anchors, ignore_index=True)
    run_frame = pd.DataFrame(run_metadata)
    summary = (
        by_seed.groupby("word", as_index=False)
        .agg(
            mean_procrustes_cosine_distance=("procrustes_cosine_distance", "mean"),
            sd_procrustes_cosine_distance=("procrustes_cosine_distance", "std"),
            mean_procrustes_cosine_similarity=("procrustes_cosine_similarity", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("mean_procrustes_cosine_distance", ascending=False)
    )

    w2_summary = pd.read_csv(W2VPCA_RESULTS / "keyword_displacement_summary.csv")
    comparison = w2_summary.merge(summary, on="word", validate="one_to_one")
    comparison["rank_w2vpca_abs_delta_pc1"] = descending_rank(
        comparison, "mean_abs_delta_pc1", "rank_w2vpca_abs_delta_pc1"
    )
    comparison["rank_procrustes_cosine_distance"] = descending_rank(
        comparison, "mean_procrustes_cosine_distance", "rank_procrustes_cosine_distance"
    )
    comparison = comparison.sort_values("rank_w2vpca_abs_delta_pc1")

    metrics = {
        "W2VPCA |ΔPC1|": "mean_abs_delta_pc1",
        "Procrustes cosine distance": "mean_procrustes_cosine_distance",
    }
    aggregate_correlations = []
    metric_items = list(metrics.items())
    for index, (left_name, left_column) in enumerate(metric_items):
        for right_name, right_column in metric_items[index + 1:]:
            aggregate_correlations.append({
                "metric_a": left_name,
                "metric_b": right_name,
                "spearman_rho": comparison[left_column].corr(comparison[right_column], method="spearman"),
                "n_keywords": len(comparison),
            })

    w2_by_seed = pd.read_csv(W2VPCA_RESULTS / "keyword_scores_by_seed.csv")
    per_seed = by_seed.merge(
        w2_by_seed[["seed", "word", "abs_delta_pc1"]],
        on=["seed", "word"],
        validate="one_to_one",
    )
    per_seed_correlations = []
    for seed, part in per_seed.groupby("seed"):
        per_seed_correlations.append({
            "seed": seed,
            "metric_a": "W2VPCA |ΔPC1|",
            "metric_b": "Procrustes cosine distance",
            "spearman_rho": part["abs_delta_pc1"].corr(
                part["procrustes_cosine_distance"], method="spearman"
            ),
            "n_keywords": len(part),
        })

    by_seed.to_csv(OUT / "procrustes_scores_by_seed.csv", index=False)
    summary.to_csv(OUT / "procrustes_summary.csv", index=False)
    comparison.to_csv(OUT / "procrustes_w2vpca_comparison.csv", index=False)
    pd.DataFrame(aggregate_correlations).to_csv(
        OUT / "procrustes_rank_correlations.csv", index=False
    )
    pd.DataFrame(per_seed_correlations).to_csv(
        OUT / "procrustes_rank_correlations_by_seed.csv", index=False
    )
    run_frame.to_csv(OUT / "procrustes_run_metadata.csv", index=False)
    anchor_rows.to_csv(OUT / "procrustes_anchor_vocabulary.csv", index=False)



if __name__ == "__main__":
    main()
