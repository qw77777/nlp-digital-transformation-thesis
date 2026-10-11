# -*- coding: utf-8 -*-
"""
03_feature_extraction.py
========================
第三步：特征提取 —— 三条技术路线同时实现，便于交叉验证

    路线一 词典法   Dictionary   命中词典词加权计数 → 可解释性最强（主测度）
    路线二 词嵌入   Word2Vec/LSA 文档向量与"数字化语义轴"的余弦相似度
    路线三 主题模型 LDA          以主题比例定测度 / 用主题高频词扩展词典

三条路线互为"三角互证"：若三者高度相关，说明测度捕捉的是同一个构念。

输出：
    data/interim/features.csv               企业-年度三类特征
    results/tables/table_lda_topics.csv     LDA 主题词表
    results/tables/table_lda_perplexity.csv 不同 K 的困惑度
    results/tables/table_emb_axis.csv       与"数字化语义轴"最接近的词
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from nlp_utils import (
    DATA_INTERIM, RESULTS_TABLES, DIGITAL_WORDS, DIGITAL_WEIGHT, DIGITAL_DIM,
    RANDOM_SEED,
)

# 尝试使用 gensim 训练 Word2Vec；若环境不可用则退化为 LSA（潜在语义分析）
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except Exception:                                    # pragma: no cover
    HAS_GENSIM = False


# ==========================================================================
# 路线一：词典法
# ==========================================================================
def dictionary_features(df):
    """加权词频法：命中词数 / 总词数 × 100。"""
    dimA, dimB = [], []
    cnt, cnt_w, ratio, ratio_w = [], [], [], []
    for toks in df["tokens"]:
        c = Counter(toks)
        n_total = max(len(toks), 1)
        raw = sum(c.get(w, 0) for w in DIGITAL_WORDS)
        raw_w = sum(c.get(w, 0) * DIGITAL_WEIGHT[w] for w in DIGITAL_WORDS)
        a = sum(c.get(w, 0) for w in DIGITAL_WORDS if DIGITAL_DIM[w].startswith("A"))
        b = sum(c.get(w, 0) for w in DIGITAL_WORDS if DIGITAL_DIM[w].startswith("B"))
        cnt.append(raw)
        cnt_w.append(round(raw_w, 2))
        ratio.append(round(raw / n_total * 100, 4))
        ratio_w.append(round(raw_w / n_total * 100, 4))
        dimA.append(a)
        dimB.append(b)
    out = pd.DataFrame({
        "dict_count": cnt, "dict_count_w": cnt_w,
        "dict_ratio": ratio, "dict_ratio_w": ratio_w,
        "dimA_count": dimA, "dimB_count": dimB,
    })
    return out


# ==========================================================================
# 路线二：词嵌入
# ==========================================================================
def embedding_features(df):
    docs = [t for t in df["tokens"]]
    if HAS_GENSIM:
        model = Word2Vec(docs, vector_size=100, window=5, min_count=3,
                         workers=4, sg=1, epochs=30, seed=RANDOM_SEED)
        wv = model.wv

        def vec(toks):
            vs = [wv[t] for t in toks if t in wv]
            return np.mean(vs, axis=0) if vs else np.zeros(wv.vector_size)

        axis = np.mean([wv[w] for w in DIGITAL_WORDS if w in wv], axis=0)

        def cos(v):
            na, nb = np.linalg.norm(v), np.linalg.norm(axis)
            return float(v @ axis / (na * nb)) if na > 0 and nb > 0 else 0.0

        doc_vecs = np.vstack([vec(t) for t in docs])
        score = np.array([cos(v) for v in doc_vecs])
        # 与"数字化语义轴"最接近的词（写进论文，用于证明轴的含义）
        sims = [(w, float(wv[w] @ axis / (np.linalg.norm(wv[w]) * np.linalg.norm(axis))))
                for w in wv.index_to_key]
        sims.sort(key=lambda x: -x[1])
        pd.DataFrame(sims[:30], columns=["word", "cos_with_axis"]).to_csv(
            RESULTS_TABLES / "table_emb_axis.csv", index=False, encoding="utf-8-sig")
        method = "Word2Vec"
    else:
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False,
                                token_pattern=None)
        X = tfidf.fit_transform(docs)
        svd = TruncatedSVD(n_components=100, random_state=RANDOM_SEED)
        doc_latent = svd.fit_transform(X)
        vocab = tfidf.get_feature_names_out()
        idx = {w: i for i, w in enumerate(vocab)}
        d = np.zeros(X.shape[1])
        for w in DIGITAL_WORDS:
            if w in idx:
                d[idx[w]] = 1.0
        axis_latent = svd.components_ @ d
        na = np.linalg.norm(axis_latent)
        score = doc_latent @ axis_latent / (
            np.linalg.norm(doc_latent, axis=1) * na + 1e-9)
        method = "LSA(TruncatedSVD)"

    out = pd.DataFrame({"emb_cos": np.round(score, 4)})
    print(f"[03] 词嵌入路线使用：{method}")
    return out, method


# ==========================================================================
# 路线三：主题模型 LDA
# ==========================================================================
def lda_features(df):
    docs = [" ".join(t) for t in df["tokens"]]
    cv = CountVectorizer(max_features=800, min_df=3, token_pattern=r"\S+")
    X = cv.fit_transform(docs)
    vocab = cv.get_feature_names_out()
    idx = {w: i for i, w in enumerate(vocab)}

    # 困惑度曲线（用于确定主题数 K）
    perp = []
    for k in [4, 6, 8, 10, 12]:
        lda = LatentDirichletAllocation(n_components=k, random_state=RANDOM_SEED,
                                        max_iter=15, learning_method="batch")
        lda.fit(X)
        perp.append({"K": k, "perplexity": round(float(lda.perplexity(X)), 2)})
    pd.DataFrame(perp).to_csv(RESULTS_TABLES / "table_lda_perplexity.csv",
                              index=False, encoding="utf-8-sig")

    K = 8
    lda = LatentDirichletAllocation(n_components=K, random_state=RANDOM_SEED,
                                    max_iter=25, learning_method="batch")
    doc_topic = lda.fit_transform(X)

    # 主题词表 + 识别"数字化主题"
    dict_idx = {idx[w] for w in DIGITAL_WORDS if w in idx}
    rows, topic_digital_share = [], []
    for k in range(K):
        comp = lda.components_[k]
        top = np.argsort(-comp)[:12]
        top_words = [vocab[i] for i in top]
        share = float(comp[list(dict_idx)].sum() / comp.sum()) if dict_idx else 0.0
        topic_digital_share.append(share)
        rows.append({"topic": k, "top_words": "、".join(top_words),
                     "digital_share": round(share, 4)})
    pd.DataFrame(rows).to_csv(RESULTS_TABLES / "table_lda_topics.csv",
                              index=False, encoding="utf-8-sig")

    dt_topic = int(np.argmax(topic_digital_share))
    digital_prop = doc_topic[:, dt_topic]

    # 路径 B：用数字化主题的高频词扩展词典，再做词典法计数
    comp = lda.components_[dt_topic]
    top_expand = [vocab[i] for i in np.argsort(-comp)[:30]]
    expand = [w for w in top_expand if w not in DIGITAL_WORDS][:12]
    counts, ratios = [], []
    for toks in df["tokens"]:
        c = Counter(toks)
        raw = sum(c.get(w, 0) for w in DIGITAL_WORDS) + sum(c.get(w, 0) for w in expand)
        counts.append(raw)
        ratios.append(round(raw / max(len(toks), 1) * 100, 4))

    out = pd.DataFrame({
        "lda_topic_prop": np.round(digital_prop, 4),
        "lda_expand_count": counts,
        "lda_expand_ratio": ratios,
    })
    print(f"[03] LDA 主题数 K={K}；数字化主题为 topic-{dt_topic}，"
          f"扩展词：{'、'.join(expand)}")
    return out, dt_topic, expand


def main():
    with open(DATA_INTERIM / "corpus_tokens.json", encoding="utf-8") as f:
        data = json.load(f)
    clean = pd.read_csv(DATA_INTERIM / "corpus_clean.csv", dtype={"firm_id": str})
    df = clean[["firm_id", "year"]].copy()
    df["tokens"] = data["tokens"]

    f1 = dictionary_features(df)
    f2, emb_method = embedding_features(df)
    f3, dt_topic, expand = lda_features(df)

    feats = pd.concat([df[["firm_id", "year"]], f1, f2, f3], axis=1)
    feats.to_csv(DATA_INTERIM / "features.csv", index=False, encoding="utf-8-sig")
    print(f"[03] 特征矩阵：{feats.shape[0]} 行 × {feats.shape[1]} 列")
    return feats


if __name__ == "__main__":
    main()
