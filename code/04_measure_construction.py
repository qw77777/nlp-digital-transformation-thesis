# -*- coding: utf-8 -*-
"""
04_measure_construction.py
==========================
第四步：指标构建 —— 从文本向量到可直接进回归的研究变量

三个必须写进论文的决策：
    决策一（分母）：统一采用"预处理后有效词数"作分母，规避篇幅差异；
                   并在稳健性检验中改用"总字数"口径。
    决策二（加权）：主测度采用等权（词典命中词数）；稳健性采用词频权重法。
    决策三（命名）：变量名体现"构念 + 计算方式"，
                   如 DT_Ratio（数字化词频占比）、DT_Composite（复合测度）。

三种指标层次：
    L1 原始计数  Dict_Count
    L2 标准化比率 DT_Ratio = 命中词数 / 有效词数 × 100
    L3 复合测度  DT_Composite = 三类特征标准化后的第一主成分

输出：
    data/processed/firm_year_panel.csv       最终企业-年份面板
    results/tables/table_descriptive.csv     描述性统计
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from nlp_utils import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, RESULTS_TABLES

WINSOR = (0.01, 0.99)


def winsorize(s: pd.Series, lo=WINSOR[0], hi=WINSOR[1]) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def main():
    feats = pd.read_csv(DATA_INTERIM / "features.csv", dtype={"firm_id": str})
    fin = pd.read_csv(DATA_RAW / "financial_panel.csv", dtype={"firm_id": str})
    df = feats.merge(fin, on=["firm_id", "year"], how="left", validate="1:1")

    # ---------- L1 / L2 ----------
    df["Dict_Count"] = df["dict_count"]
    df["DT_Ratio"] = df["dict_ratio"]                 # 主测度（等权词频占比）
    df["DT_Ratio_W"] = df["dict_ratio_w"]             # 稳健性：加权口径
    df["DT_Ratio_Expand"] = df["lda_expand_ratio"]    # 稳健性：LDA 扩展词典口径
    df["DT_LDA"] = df["lda_topic_prop"]               # 主题比例法（路线三路径 A）
    df["DT_Embed"] = df["emb_cos"]                    # 词嵌入余弦相似度

    # ---------- L3 复合测度：三类特征标准化后取第一主成分 ----------
    z = df[["DT_Ratio", "DT_Embed", "DT_LDA"]].apply(
        lambda s: (s - s.mean()) / s.std(ddof=1))
    pca = PCA(n_components=1, random_state=42).fit(z)
    pc1 = pca.transform(z)[:, 0]
    # 使复合测度方向与词典法一致
    if np.corrcoef(pc1, df["DT_Ratio"])[0, 1] < 0:
        pc1 = -pc1
    df["DT_Composite"] = (pc1 - pc1.mean()) / pc1.std(ddof=1)
    var_ratio = float(pca.explained_variance_ratio_[0])

    # ---------- 以主测度（词典法）为准的离散化：高低分组 ----------
    med = df["DT_Ratio"].median()
    df["DT_High"] = (df["DT_Ratio"] > med).astype(int)

    # ---------- 控制变量：对数化与缩尾 ----------
    df["ln_patents"] = np.log1p(df["patents"])
    df["RD"] = df["rd_intensity"]
    for v in ["DT_Ratio", "DT_Ratio_W", "DT_Ratio_Expand", "DT_LDA", "DT_Embed",
              "DT_Composite", "size", "lev", "roa", "growth", "ln_patents", "RD",
              "intangible", "tech_staff", "age"]:
        df[v] = winsorize(df[v].astype(float))

    df = df.sort_values(["firm_id", "year"]).reset_index(drop=True)
    df.to_csv(DATA_PROCESSED / "firm_year_panel.csv", index=False,
              encoding="utf-8-sig")

    # ---------- 描述性统计表（写进论文） ----------
    desc_vars = ["DT_Ratio", "Dict_Count", "DT_Embed", "DT_LDA", "DT_Composite",
                 "RD", "ln_patents", "size", "lev", "roa", "growth",
                 "intangible", "tech_staff"]
    desc = df[desc_vars].describe().T
    desc = desc[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["观测数", "均值", "标准差", "最小值", "中位数", "最大值"]
    desc = desc.round(4)
    desc.to_csv(RESULTS_TABLES / "table_descriptive.csv", encoding="utf-8-sig")

    print(f"[04] 面板构建完成：{df.shape[0]} 个企业-年度观测，{df.shape[1]} 列")
    print(f"[04] 复合测度第一主成分解释方差比例 = {var_ratio:.3f}")
    print(f"[04] 主测度 DT_Ratio：均值 {df['DT_Ratio'].mean():.3f}，"
          f"标准差 {df['DT_Ratio'].std():.3f}，"
          f"范围 [{df['DT_Ratio'].min():.3f}, {df['DT_Ratio'].max():.3f}]")
    return df


if __name__ == "__main__":
    main()
