from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "results"
W2VPCA_SCRIPT = BASE / "scripts" / "run_w2vpca_combined.py"
W2VPCA_RESULTS = BASE / "reanalysis_combined_timed"


def load_w2vpca_module():
    spec = importlib.util.spec_from_file_location("w2vpca", W2VPCA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {W2VPCA_SCRIPT}")
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


def cosine_similarity(model: Word2Vec, before: str, after: str) -> float:
    before_vector = model.wv[before]
    after_vector = model.wv[after]
    return float(np.dot(before_vector, after_vector) / (
        np.linalg.norm(before_vector) * np.linalg.norm(after_vector)
    ))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = load_w2vpca_module()
    documents = canonical.load_documents()
    corpus = documents["tagged_tokens"].tolist()
    words = list(canonical.TARGET_WORDS)
    seeds = list(canonical.SEEDS)

    counts = pd.Series(
        [token for document in corpus for token in document], name="temporal_token"
    ).value_counts()
    frequencies = pd.DataFrame([
        {
            "word": word,
            "count_before": int(counts.get(f"{word}_before", 0)),
            "count_after": int(counts.get(f"{word}_after", 0)),
        }
        for word in words
    ])
    frequencies["count_total"] = frequencies["count_before"] + frequencies["count_after"]

    frames = []
    for seed in seeds:
        model = train_model(corpus, seed)
        rows = []
        for word in words:
            before, after = f"{word}_before", f"{word}_after"
            if before not in model.wv or after not in model.wv:
                raise ValueError(f"Missing temporal vector for {word}")
            similarity = cosine_similarity(model, before, after)
            rows.append({
                "seed": seed,
                "word": word,
                "cosine_similarity": similarity,
                "cosine_distance": 1 - similarity,
            })
        frames.append(pd.DataFrame(rows))

    by_seed = pd.concat(frames, ignore_index=True)
    summary = (
        by_seed.groupby("word", as_index=False)
        .agg(
            mean_cosine_distance=("cosine_distance", "mean"),
            sd_cosine_distance=("cosine_distance", "std"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("mean_cosine_distance", ascending=False)
    )
    w2_summary = pd.read_csv(W2VPCA_RESULTS / "keyword_displacement_summary.csv")
    comparison = w2_summary.merge(summary, on="word", validate="one_to_one").merge(
        frequencies, on="word", validate="one_to_one"
    )
    comparison["rank_w2vpca_abs_delta_pc1"] = comparison["mean_abs_delta_pc1"].rank(
        ascending=False, method="min"
    ).astype(int)
    comparison["rank_cosine_distance"] = comparison["mean_cosine_distance"].rank(
        ascending=False, method="min"
    ).astype(int)
    comparison = comparison.sort_values("rank_w2vpca_abs_delta_pc1")
    rho = comparison["mean_abs_delta_pc1"].corr(
        comparison["mean_cosine_distance"], method="spearman"
    )

    w2_by_seed = pd.read_csv(W2VPCA_RESULTS / "keyword_scores_by_seed.csv")
    per_seed = by_seed.merge(
        w2_by_seed[["seed", "word", "abs_delta_pc1"]],
        on=["seed", "word"],
        validate="one_to_one",
    )
    per_seed_correlations = pd.DataFrame([
        {
            "seed": seed,
            "metric_a": "W2VPCA |ΔPC1|",
            "metric_b": "Cosine distance",
            "spearman_rho": part["abs_delta_pc1"].corr(
                part["cosine_distance"], method="spearman"
            ),
            "n_keywords": len(part),
        }
        for seed, part in per_seed.groupby("seed")
    ])
    rank_correlation = pd.DataFrame([{
        "metric_a": "W2VPCA |ΔPC1|",
        "metric_b": "Cosine distance",
        "spearman_rho": rho,
        "n_keywords": len(comparison),
    }])
    by_seed.to_csv(OUT / "cosine_scores_by_seed.csv", index=False)
    summary.to_csv(OUT / "cosine_summary.csv", index=False)
    frequencies.to_csv(OUT / "target_word_frequencies.csv", index=False)
    comparison.to_csv(OUT / "cosine_w2vpca_comparison.csv", index=False)
    rank_correlation.to_csv(OUT / "cosine_rank_correlation.csv", index=False)
    per_seed_correlations.to_csv(OUT / "cosine_rank_correlations_by_seed.csv", index=False)


if __name__ == "__main__":
    main()
