# -*- coding: utf-8 -*-
"""
05_validity_reliability.py
==========================
第五步：信效度检验 —— 证明所构造的测度"值得信"

四类效度（内容 / 聚合 / 区分 / 预测）+ 两类信度（内部一致性 / 时间稳定性）
+ 方差分解。不通过检验则回到第三步更换技术路线。

输出（results/tables/）：
    table_content_validity.csv    内容效度：词典来源与分布
    table_convergent.csv          聚合效度：与同类测度/外部指数相关
    table_discriminant.csv        区分效度：与"不该相关"变量相关
    table_predictive.csv          预测效度：回归系数
    table_reliability.csv         信度：Cronbach α、分半、跨期相关
    table_variance_decomp.csv     方差分解：企业间/企业内差异占比
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from nlp_utils import (DATA_PROCESSED, DATA_INTERIM, RESULTS_TABLES,
                       DIGITAL_DICT, DIGITAL_WORDS, RANDOM_SEED)

GROUPS = {
    "G1_智能算法与AI": ["人工智能", "机器学习", "深度学习", "自然语言处理",
                      "计算机视觉", "神经网络", "智能算法", "知识图谱",
                      "数据挖掘", "智能决策"],
    "G2_大数据与云": ["大数据", "云计算", "云平台", "数据中台", "传感技术",
                    "数据驱动", "数据赋能", "数字经济", "数字产业化", "产业数字化"],
    "G3_物联网与区块链": ["区块链", "物联网", "工业互联网", "5G", "边缘计算",
                       "数字孪生", "自动化", "网络化", "数字技术", "数字生态"],
    "G4_智能制造与生产": ["智能制造", "智能工厂", "智能生产", "数字化车间",
                       "智能仓储", "智慧物流", "工业软件", "数字供应链",
                       "智慧供应链", "平台化"],
    "G5_数字化战略与营销": ["数字化转型", "数字化", "数字化改造", "数字化营销",
                         "精准营销", "电子商务", "线上渠道", "数字营销",
                         "数字化运营", "数字化战略"],
    "G6_信息化与数字服务": ["智能化", "信息化", "智慧化", "数字化管理", "智能管理",
                         "数字平台", "数字服务", "智能服务", "数字金融", "智能客服"],
}


def cronbach_alpha(X: np.ndarray) -> float:
    """Cronbach's α：X 为 (n_obs, n_items)。"""
    k = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def main():
    df = pd.read_csv(DATA_PROCESSED / "firm_year_panel.csv", dtype={"firm_id": str})

    # ---------------- 1. 内容效度 ----------------
    cd = pd.DataFrame(DIGITAL_DICT, columns=["词条", "维度", "权重"])
    cd.to_csv(RESULTS_TABLES / "table_content_validity.csv",
              index=False, encoding="utf-8-sig")
    dim_stats = cd.groupby("维度").agg(词数=("词条", "count"),
                                     平均权重=("权重", "mean")).round(2)

    # ---------------- 2. 聚合效度 ----------------
    pairs = {
        "DT_Embed(词嵌入)": "DT_Embed",
        "DT_LDA(主题比例)": "DT_LDA",
        "DT_Ratio_Expand(LDA扩展词典)": "DT_Ratio_Expand",
        "第三方数字化指数": "dt_rating",
        "无形资产占比": "intangible",
        "技术岗人员占比": "tech_staff",
    }
    rows = []
    for name, col in pairs.items():
        r, p = stats.pearsonr(df["DT_Ratio"], df[col])
        rows.append({"参照变量": name, "Pearson_r": round(r, 3),
                     "p值": f"{p:.2e}", "显著性": "***" if p < 0.01 else
                     ("**" if p < 0.05 else ("*" if p < 0.1 else ""))})
    conv = pd.DataFrame(rows)
    conv.to_csv(RESULTS_TABLES / "table_convergent.csv",
                index=False, encoding="utf-8-sig")

    # ---------------- 3. 区分效度 ----------------
    rows = []
    for name, col in {"公司规模 size": "size", "资产负债率 lev": "lev",
                      "总资产收益率 roa": "roa", "企业年龄 age": "age",
                      "营业收入增长率 growth": "growth"}.items():
        r, p = stats.pearsonr(df["DT_Ratio"], df[col])
        rows.append({"参照变量": name, "Pearson_r": round(r, 3),
                     "p值": f"{p:.2e}"})
    # 与行业虚拟变量的关系（点二列相关 / ANOVA）
    f_stat, p_anova = stats.f_oneway(
        *[g["DT_Ratio"].values for _, g in df.groupby("industry")])
    disc = pd.DataFrame(rows)
    disc.to_csv(RESULTS_TABLES / "table_discriminant.csv",
                index=False, encoding="utf-8-sig")
    print(f"[05] 区分效度：行业间 DT_Ratio 单因素方差分析 F={f_stat:.2f}, "
          f"p={p_anova:.3f}")

    # ---------------- 4. 预测效度 ----------------
    df["ln_size"] = np.log(df["size"])
    m1 = smf.ols("RD ~ DT_Ratio + ln_size + lev + roa + growth + age + soe "
                 "+ C(year) + C(industry)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["firm_id"]})
    m2 = smf.ols("ln_patents ~ DT_Ratio + ln_size + lev + roa + growth + age + soe "
                 "+ C(year) + C(industry)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["firm_id"]})
    pred = pd.DataFrame({
        "被解释变量": ["研发投入强度 RD", "专利产出 ln_patents"],
        "DT_Ratio系数": [round(m1.params["DT_Ratio"], 4),
                        round(m2.params["DT_Ratio"], 4)],
        "标准误": [round(m1.bse["DT_Ratio"], 4), round(m2.bse["DT_Ratio"], 4)],
        "t值": [round(m1.tvalues["DT_Ratio"], 3), round(m2.tvalues["DT_Ratio"], 3)],
        "p值": [f"{m1.pvalues['DT_Ratio']:.2e}", f"{m2.pvalues['DT_Ratio']:.2e}"],
        "N": [int(m1.nobs), int(m2.nobs)],
        "R2": [round(m1.rsquared, 3), round(m2.rsquared, 3)],
    })
    pred.to_csv(RESULTS_TABLES / "table_predictive.csv",
                index=False, encoding="utf-8-sig")

    # ---------------- 5. 信度 ----------------
    # 5.1 内部一致性：把 60 词词典划分为 6 个子词典，以各子词典占比为 6 个"题项"
    with open(DATA_INTERIM / "corpus_tokens.json", encoding="utf-8") as f:
        tokens = json.load(f)["tokens"]
    items_raw = np.zeros((len(df), len(GROUPS)))     # 子词典命中次数（题项，主口径）
    items = np.zeros((len(df), len(GROUPS)))         # 子词典占比（题项，对照口径）
    gnames = list(GROUPS.keys())
    for i, toks in enumerate(tokens):
        n = max(len(toks), 1)
        c = Counter(toks)
        for j, g in enumerate(gnames):
            hit = sum(c.get(w, 0) for w in GROUPS[g])
            items_raw[i, j] = hit
            items[i, j] = hit / n * 100
    # Cronbach's α 对题项的单位缩放不敏感，故次数口径直接可用
    alpha = cronbach_alpha(items_raw)
    alpha_ratio = cronbach_alpha(items)
    # 平均项目间相关系数（次数口径）
    cm = np.corrcoef(items_raw.T)
    mean_inter_item = float((cm.sum() - np.trace(cm)) / (len(gnames) * (len(gnames) - 1)))

    # 5.2 分半信度：把 60 个词典词随机分为两半，各算一个占比，报告 Spearman-Brown 校正
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(DIGITAL_WORDS))
    halfA = [DIGITAL_WORDS[k] for k in perm[:len(DIGITAL_WORDS) // 2]]
    halfB = [DIGITAL_WORDS[k] for k in perm[len(DIGITAL_WORDS) // 2:]]

    def half_ratio(wordset):
        out = []
        for toks in tokens:
            c = Counter(toks)
            out.append(sum(c.get(w, 0) for w in wordset) / max(len(toks), 1) * 100)
        return np.array(out)

    r_half = float(np.corrcoef(half_ratio(halfA), half_ratio(halfB))[0, 1])
    sb = 2 * r_half / (1 + r_half) if r_half > -1 else 0.0

    # 5.3 时间稳定性：相邻年度测度的自相关
    df_s = df.sort_values(["firm_id", "year"])
    df_s["DT_lag"] = df_s.groupby("firm_id")["DT_Ratio"].shift(1)
    ac = df_s[["DT_Ratio", "DT_lag"]].dropna()
    r_ac = float(np.corrcoef(ac["DT_Ratio"], ac["DT_lag"])[0, 1])
    # 5.4 测度准确性：与潜在真值 DT* 的相关（模拟数据独有）
    r_true = float(np.corrcoef(df["DT_Ratio"], df["dt_star_true"])[0, 1])

    rel = pd.DataFrame({
        "信度指标": [f"Cronbach's α（{len(gnames)} 个子词典·次数口径）",
                    "Cronbach's α（子词典·占比口径）", "子词典间平均相关系数",
                    "分半信度 Spearman-Brown", "跨期相关（相邻年度）",
                    "与潜在真值相关（模拟）"],
        "取值": [round(alpha, 3), round(alpha_ratio, 3),
                round(mean_inter_item, 3), round(sb, 3),
                round(r_ac, 3), round(r_true, 3)],
        "判断标准": ["α≥0.7 良好", "α≥0.7 良好", "0.3~0.5 适中",
                    "≥0.7 良好", "≥0.5 稳定", "越高越准"],
    })
    rel.to_csv(RESULTS_TABLES / "table_reliability.csv",
               index=False, encoding="utf-8-sig")

    # ---------------- 6. 方差分解 ----------------
    grand = df["DT_Ratio"].mean()
    between = df.groupby("firm_id")["DT_Ratio"].mean()
    n_per = df.groupby("firm_id")["DT_Ratio"].size()
    ss_between = float((n_per * (between - grand) ** 2).sum())
    ss_within = float(((df["DT_Ratio"] - df.groupby("firm_id")["DT_Ratio"]
                        .transform("mean")) ** 2).sum())
    vd = pd.DataFrame({
        "变异来源": ["企业间（Between）", "企业内（Within）"],
        "离差平方和": [round(ss_between, 3), round(ss_within, 3)],
        "方差占比": [round(ss_between / (ss_between + ss_within), 4),
                    round(ss_within / (ss_between + ss_within), 4)],
    })
    vd.to_csv(RESULTS_TABLES / "table_variance_decomp.csv",
              index=False, encoding="utf-8-sig")

    print(f"[05] 内容效度：{len(cd)} 个词，分布为 "
          f"{dict(dim_stats['词数'])}")
    print(f"[05] 聚合效度（与第三方指数）r="
          f"{conv.loc[conv['参照变量'] == '第三方数字化指数', 'Pearson_r'].values[0]}")
    print(f"[05] 预测效度：RD 系数 {m1.params['DT_Ratio']:.4f} "
          f"(p={m1.pvalues['DT_Ratio']:.2e})；"
          f"ln_patents 系数 {m2.params['DT_Ratio']:.4f} "
          f"(p={m2.pvalues['DT_Ratio']:.2e})")
    print(f"[05] 信度：Cronbach α={alpha:.3f}（项目间平均相关 "
          f"{mean_inter_item:.3f}），分半={sb:.3f}，跨期={r_ac:.3f}，"
          f"与真值={r_true:.3f}")
    print(f"[05] 方差分解：企业间占比 "
          f"{ss_between / (ss_between + ss_within):.3f}")
    return df, m1, m2


if __name__ == "__main__":
    main()
