from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reanalysis_combined_timed"
EVENT_DATE = pd.Timestamp("2025-08-08")
WINDOW_DAYS = 90
SEEDS = (13, 27, 41, 59, 83)

KEYWORD_GROUPS = {
    "Chính sách và thực thi": ["hạn_chế", "cấm", "khuyến_khích", "xử_phạt", "kiểm_soát", "giám_sát", "vi_phạm", "ưu_tiên"],
    "Phương tiện": ["xe_máy", "ô_tô", "xe_xăng", "xe_điện", "bus", "metro"],
    "Môi trường": ["ô_nhiễm", "khí_thải", "bụi_mịn", "phát_thải", "thải", "bụi", "xanh", "đẹp", "không_khí"],
    "Đánh giá và công bằng": ["công_bằng", "bất_cập", "hợp_lý", "vô_lý", "cần_thiết", "bất_tiện", "ủng_hộ", "phản_đối"],
    "Kinh tế": ["kinh_tế", "chi_phí", "tốn_kém", "tiền", "nghèo", "lãng_phí"],
    "Xe điện và hạ tầng": ["trạm_sạc", "pin", "cháy", "sạc", "chung_cư"],
}
TARGET_WORDS = [word for group in KEYWORD_GROUPS.values() for word in group]


def load_documents() -> pd.DataFrame:
    articles = pd.read_parquet(BASE / "outputs" / "preprocessing_step_3_articles.parquet")
    comments = pd.read_parquet(BASE / "outputs" / "preprocessing_step_3_comments.parquet")
    articles = articles.rename(columns={"content_tokenized": "tokenized_text"})
    comments = comments.rename(columns={"comment_text_tokenized": "tokenized_text"})
    articles = articles[["article_id", "publish_date", "tokenized_text"]].copy()
    comments = comments[["comment_id", "article_id", "publish_date", "tokenized_text"]].copy()
    articles["document_type"] = "article"
    comments["document_type"] = "comment"
    articles["document_id"] = articles["article_id"].astype(str)
    comments["document_id"] = comments["comment_id"].astype(str)
    documents = pd.concat(
        [articles[["document_id", "article_id", "publish_date", "document_type", "tokenized_text"]],
         comments[["document_id", "article_id", "publish_date", "document_type", "tokenized_text"]]],
        ignore_index=True,
    )
    documents["publish_date"] = pd.to_datetime(documents["publish_date"], errors="coerce")
    documents = documents.dropna(subset=["publish_date", "tokenized_text"]).copy()
    start, end = EVENT_DATE - pd.Timedelta(days=WINDOW_DAYS), EVENT_DATE + pd.Timedelta(days=WINDOW_DAYS)
    documents = documents.loc[(documents["publish_date"] >= start) & (documents["publish_date"] < end)].copy()
    documents["analysis_period"] = np.where(documents["publish_date"] < EVENT_DATE, "before", "after")
    documents["tokens"] = documents["tokenized_text"].map(lambda value: str(value).split())
    documents = documents.loc[documents["tokens"].map(bool)].copy()
    documents["tagged_tokens"] = documents.apply(
        lambda row: [f"{token}_{row.analysis_period}" for token in row.tokens], axis=1
    )
    return documents


def fit_seed(corpus: list[list[str]], seed: int) -> tuple[pd.DataFrame, float]:
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1,
                     sg=0, negative=5, sample=1e-3, epochs=20, seed=seed,
                     workers=1, sorted_vocab=1)
    labels = [f"{word}_{period}" for word in TARGET_WORDS for period in ("before", "after")]
    labels = [label for label in labels if label in model.wv]
    x = StandardScaler().fit_transform(np.vstack([model.wv[label] for label in labels]))
    pca = PCA(n_components=1).fit(x)
    scores = pca.transform(x).ravel()
    lookup = dict(zip(labels, scores))
    rows = []
    for word in TARGET_WORDS:
        before, after = f"{word}_before", f"{word}_after"
        if before in lookup and after in lookup:
            delta = lookup[after] - lookup[before]
            rows.append({"seed": seed, "word": word, "pc1_before": lookup[before],
                         "pc1_after": lookup[after], "delta_pc1": delta,
                         "abs_delta_pc1": abs(delta)})
    return pd.DataFrame(rows), float(pca.explained_variance_ratio_[0])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    documents = load_documents()
    frames, variance = [], []
    for seed in SEEDS:
        frame, explained = fit_seed(documents["tagged_tokens"].tolist(), seed)
        frames.append(frame); variance.append({"seed": seed, "pc1_explained_variance_ratio": explained})
    scores = pd.concat(frames, ignore_index=True)
    summary = scores.groupby("word", as_index=False).agg(mean_abs_delta_pc1=("abs_delta_pc1", "mean"), sd_abs_delta_pc1=("abs_delta_pc1", "std"), mean_delta_pc1=("delta_pc1", "mean"), n_runs=("seed", "nunique")).sort_values("mean_abs_delta_pc1", ascending=False)
    documents["token_count"] = documents.tokens.map(len)
    counts = documents.groupby(["analysis_period", "document_type"]).agg(documents=("document_id", "size"), parent_articles=("article_id", "nunique"), tokens=("token_count", "sum"), start=("publish_date", "min"), end=("publish_date", "max")).reset_index()
    scores.to_csv(OUT / "keyword_scores_by_seed.csv", index=False); summary.to_csv(OUT / "keyword_displacement_summary.csv", index=False)
    counts.to_csv(OUT / "sample_description.csv", index=False); pd.DataFrame(variance).to_csv(OUT / "pc1_variance_by_seed.csv", index=False)


if __name__ == "__main__":
    main()
