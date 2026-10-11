# -*- coding: utf-8 -*-
"""
02_preprocessing.py
===================
第二步：文本预处理 —— 五步流水线

    ① 文档解析  本研究的原始语料已是结构化 CSV（模拟年报 MD&A），此处仅做编码与字段校验
    ② 文本清洗  去页眉页脚/页码/证券代码等噪声，统一标点，去除乱码与空白
    ③ 中文分词  jieba 精确模式；同时载入用户词典，保证"数字化转型""工业互联网"等
                专有词不被切碎
    ④ 去停用词  通用停用词 + 财经领域停用词 + 语料自动识别的高频通用词（自定义维）
    ⑤ 词形归一  统一大小写（如 5G→5g 保留展示形式）、剔除纯数字与单字噪声

输出：
    data/interim/corpus_clean.csv        清洗后文本
    data/interim/corpus_tokens.json      分词结果（list of tokens）
    results/tables/table_preprocess_demo.csv   预处理前后对照示例（写进论文用）
"""

import json
import re
from collections import Counter

import jieba
import pandas as pd

from nlp_utils import (
    DATA_RAW, DATA_INTERIM, RESULTS_TABLES, STOPWORDS,
    DIGITAL_WORDS, FINANCE_DOMAIN_STOPWORDS,
)

# --------------------------------------------------------------------------
# 0. 载入用户词典：把 60 个数字化转型词典词加入 jieba，避免专有词被切碎
# --------------------------------------------------------------------------
USERDICT = DATA_INTERIM / "user_dict.txt"
with open(USERDICT, "w", encoding="utf-8") as f:
    for w in DIGITAL_WORDS:
        f.write(f"{w} 100000 n\n")
jieba.load_userdict(str(USERDICT))

# 需要清洗掉的噪声模式
NOISE_PATTERNS = [
    r"第\s*\d+\s*页", r"共\s*\d+\s*页", r"证券代码[:：]?\s*\d+",
    r"股票代码[:：]?\s*\d+", r"公告编号[:：]?\s*[\w\-]+",
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"[\u3000\xa0]+",
]


def clean_text(text: str) -> str:
    """文本清洗：去噪声、统一标点、去空白。"""
    if not isinstance(text, str):
        return ""
    t = text
    for p in NOISE_PATTERNS:
        t = re.sub(p, "", t)
    # 全角转半角标点统一
    t = t.replace("，", "，").replace("．", "。")
    # 只保留中文、英文、数字与常用标点
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9%\.]", "", t)
    t = t.replace(".", "。")
    return t.strip()


def normalize_token(tok: str) -> str:
    """词形归一：字母统一小写；数字与标点单独处理。"""
    if re.fullmatch(r"[A-Za-z0-9]+", tok):
        return tok.lower()
    return tok


def tokenize(text: str) -> list:
    """jieba 精确模式分词 + 停用词过滤 + 噪声剔除。"""
    toks = jieba.lcut(text, cut_all=False)
    out = []
    for t in toks:
        t = normalize_token(t)
        if not t or t in STOPWORDS:
            continue
        if re.fullmatch(r"[\d\.%]+", t):          # 纯数字 / 百分号
            continue
        if len(t) == 1 and not re.fullmatch(r"[a-z0-9]", t):  # 单字噪声（保留 a-z0-9）
            continue
        out.append(t)
    return out


def main():
    df = pd.read_csv(DATA_RAW / "corpus_mda.csv", dtype={"firm_id": str})

    # ② 清洗
    df["text_clean"] = df["mda_text"].map(clean_text)

    # ③④⑤ 分词 + 去停用词 + 归一
    df["tokens"] = df["text_clean"].map(tokenize)
    df["n_tokens"] = df["tokens"].map(len)
    df["text_seg"] = df["tokens"].map(lambda x: " ".join(x))

    # 语料自动识别的"自定义停用词"：出现在 >85% 文档中且长度≤3 的通用词
    freq_doc = Counter()
    for tk in df["tokens"]:
        freq_doc.update(set(tk))
    thresh = 0.85 * len(df)
    auto_stop = {w for w, c in freq_doc.items()
                 if c >= thresh and len(w) <= 3 and w not in DIGITAL_WORDS}
    # 合并进停用词表并重新过滤
    all_stop = STOPWORDS | auto_stop
    df["tokens"] = df["tokens"].map(lambda ts: [t for t in ts if t not in all_stop])
    df["n_tokens"] = df["tokens"].map(len)
    df["text_seg"] = df["tokens"].map(lambda x: " ".join(x))

    df[["firm_id", "year", "text_clean", "text_seg", "n_tokens"]].to_csv(
        DATA_INTERIM / "corpus_clean.csv", index=False, encoding="utf-8-sig")
    with open(DATA_INTERIM / "corpus_tokens.json", "w", encoding="utf-8") as f:
        json.dump({"custom_stopwords": sorted(auto_stop),
                   "tokens": df["tokens"].tolist()}, f, ensure_ascii=False)

    # 预处理前后对照（写论文用）
    demo = pd.DataFrame({
        "阶段": ["①原始", "②清洗后", "③分词后", "④去停用词后"],
        "文本内容": [
            df["mda_text"].iloc[0],
            df["text_clean"].iloc[0],
            " / ".join(jieba.lcut(df["text_clean"].iloc[0])[:24]),
            " / ".join(df["tokens"].iloc[0][:24]),
        ],
    })
    demo.to_csv(RESULTS_TABLES / "table_preprocess_demo.csv",
                index=False, encoding="utf-8-sig")

    print(f"[02] 清洗分词完成：{len(df)} 条文本，平均有效词数 "
          f"{df['n_tokens'].mean():.1f}（原始平均字数 "
          f"{df['mda_text'].str.len().mean():.0f}）")
    print(f"[02] 自动识别的自定义停用词（前 15 个）："
          f"{'、'.join(sorted(auto_stop)[:15])}")
    return df


if __name__ == "__main__":
    main()
