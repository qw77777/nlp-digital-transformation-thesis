# -*- coding: utf-8 -*-
"""
06_regression_analysis.py
=========================
基于所构造测度的实证分析：主回归、稳健性、异质性、机制检验与图表输出。

模型设定：
    Innovation_it = α + β·DT_it + γ·Controls_it + μ_i + λ_t + ε_it
其中 Innovation 采用研发投入强度(RD)与专利产出(ln_patents)两个代理；
Controls 含规模、杠杆、ROA、成长性、年龄、产权性质；并控制年度与行业固定效应；
标准误按企业聚类。

输出：
    results/tables/table_main_regression.csv
    results/tables/table_robustness.csv
    results/tables/table_heterogeneity.csv
    results/figures/fig1_measure_distribution.png
    results/figures/fig2_dt_trend.png
    results/figures/fig3_correlation_heatmap.png
    results/figures/fig4_coefplot.png
    results/figures/fig5_lda_perplexity.png
"""

import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from nlp_utils import DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

CTRL_CORE = "ln_size + lev + roa + growth + age + soe"
CTRL = CTRL_CORE + " + C(year) + C(industry)"
CTRL_NOIND = CTRL_CORE + " + C(year)"
MODELS = {
    "RD": ["RD ~ DT_Ratio", f"RD ~ DT_Ratio + {CTRL}", f"RD ~ DT_Composite + {CTRL}"],
    "ln_patents": ["ln_patents ~ DT_Ratio",
                   f"ln_patents ~ DT_Ratio + {CTRL}",
                   f"ln_patents ~ DT_Composite + {CTRL}"],
}
SPEC_NAMES = ["(1) 单变量", "(2) 加控制+固定效应", "(3) 复合测度"]


def fit(formula, df):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["firm_id"]})


def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def main():
    df = pd.read_csv(DATA_PROCESSED / "firm_year_panel.csv", dtype={"firm_id": str})
    df["ln_size"] = np.log(df["size"])

    # ==================== 主回归 ====================
    main_rows, coef_store = [], []
    for dep, forms in MODELS.items():
        for i, f in enumerate(forms):
            m = fit(f, df)
            key = "DT_Composite" if i == 2 else "DT_Ratio"
            b, se, t, p = (m.params[key], m.bse[key], m.tvalues[key], m.pvalues[key])
            lo, hi = b - 1.96 * se, b + 1.96 * se
            main_rows.append({
                "被解释变量": dep, "模型": SPEC_NAMES[i], "变量": key,
                "系数": round(b, 4), "标准误": round(se, 4),
                "t值": round(t, 3), "p值": f"{p:.2e}", "显著性": star(p),
                "95%CI下限": round(lo, 4), "95%CI上限": round(hi, 4),
                "N": int(m.nobs), "R2": round(m.rsquared, 4),
            })
            if i == 1:
                coef_store.append({"被解释变量": dep, "系数": b,
                                   "下限": lo, "上限": hi})
    main = pd.DataFrame(main_rows)
    main.to_csv(RESULTS_TABLES / "table_main_regression.csv",
                index=False, encoding="utf-8-sig")

    # ==================== 稳健性检验 ====================
    rob_rows = []

    def add_rob(name, formula, key=None, sub=None):
        # 若未显式指定关键变量，则从公式中自动解析
        if key is None:
            key = re.search(r"~\s*([A-Za-z_][A-Za-z0-9_]*)", formula).group(1)
        d = df if sub is None else sub
        m = fit(formula, d)
        rob_rows.append({"检验": name, "变量": key,
                         "系数": round(m.params[key], 4),
                         "t值": round(m.tvalues[key], 3),
                         "p值": f"{m.pvalues[key]:.2e}",
                         "显著性": star(m.pvalues[key]),
                         "N": int(m.nobs), "R2": round(m.rsquared, 4)})

    # 替换测度口径
    add_rob("加权词典口径", f"RD ~ DT_Ratio_W + {CTRL}")
    add_rob("LDA扩展词典口径", f"RD ~ DT_Ratio_Expand + {CTRL}")
    add_rob("主题比例法", f"RD ~ DT_LDA + {CTRL}")
    add_rob("词嵌入法", f"RD ~ DT_Embed + {CTRL}")
    # 换被解释变量
    add_rob("专利产出为被解释变量", f"ln_patents ~ DT_Ratio + {CTRL}")
    # 滞后一期测度（缓解互为因果）
    d = df.sort_values(["firm_id", "year"]).copy()
    d["DT_lag"] = d.groupby("firm_id")["DT_Ratio"].shift(1)
    add_rob("测度滞后一期", f"RD ~ DT_lag + {CTRL}", key="DT_lag", sub=d.dropna(subset=["DT_lag"]))
    # 剔除极端年份
    add_rob("剔除2020年(疫情)", f"RD ~ DT_Ratio + {CTRL}",
            sub=df[df["year"] != 2020])
    # 分年度中位数替代均值
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(RESULTS_TABLES / "table_robustness.csv",
               index=False, encoding="utf-8-sig")

    # ==================== 异质性分析 ====================
    df["tech_hi"] = df["industry"].isin(["信息技术", "电子设备", "医药生物"]).astype(int)
    het_rows = []
    for gname, sub in [("国有企业", df[df["soe"] == 1]),
                       ("非国有企业", df[df["soe"] == 0]),
                       ("高技术行业", df[df["tech_hi"] == 1]),
                       ("传统行业", df[df["tech_hi"] == 0])]:
        m = fit(f"RD ~ DT_Ratio + {CTRL}", sub)
        het_rows.append({"分组/检验": gname, "变量": "DT_Ratio",
                         "系数": round(m.params["DT_Ratio"], 4),
                         "t值": round(m.tvalues["DT_Ratio"], 3),
                         "p值": f"{m.pvalues['DT_Ratio']:.2e}",
                         "显著性": star(m.pvalues["DT_Ratio"]),
                         "N": int(m.nobs)})
    # 组间系数差异的正式检验（交互项）
    for gname, formula in [
        ("交互项 DT×非国有", f"RD ~ DT_Ratio + DT_Ratio:soe + {CTRL}"),
        ("交互项 DT×高技术", f"RD ~ DT_Ratio + DT_Ratio:tech_hi + tech_hi + {CTRL_NOIND}"),
    ]:
        m = fit(formula, df)
        key = [k for k in m.params.index if ":" in k][0]
        het_rows.append({"分组/检验": gname, "变量": key.replace("DT_Ratio:", "DT*"),
                         "系数": round(m.params[key], 4),
                         "t值": round(m.tvalues[key], 3),
                         "p值": f"{m.pvalues[key]:.2e}",
                         "显著性": star(m.pvalues[key]),
                         "N": int(m.nobs)})
    het = pd.DataFrame(het_rows)
    het.to_csv(RESULTS_TABLES / "table_heterogeneity.csv",
               index=False, encoding="utf-8-sig")

    # ==================== 机制检验：DT → 技术人力 → 研发 ====================
    m_a = fit(f"tech_staff ~ DT_Ratio + {CTRL}", df)
    m_b = fit(f"RD ~ DT_Ratio + tech_staff + {CTRL}", df)

    # ==================== 图 1：测度分布 ====================
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].hist(df["DT_Ratio"], bins=40, color="#c0392b", alpha=0.85)
    axes[0].set_title("数字化转型测度 DT_Ratio 分布", fontsize=11)
    axes[0].set_xlabel("DT_Ratio（数字化词频占比‰×10）")
    axes[0].set_ylabel("企业-年度频数")
    axes[1].scatter(df["DT_Ratio"], df["DT_Embed"], s=6, alpha=0.35, color="#2471a3")
    axes[1].set_title("词典法与词嵌入法测度散点", fontsize=11)
    axes[1].set_xlabel("DT_Ratio（词典法）")
    axes[1].set_ylabel("DT_Embed（余弦相似度）")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "fig1_measure_distribution.png")
    plt.close(fig)

    # ==================== 图 2：年度趋势 ====================
    trend = df.groupby("year")["DT_Ratio"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.errorbar(trend["year"], trend["mean"], yerr=trend["std"],
                marker="o", capsize=4, color="#c0392b")
    ax.set_title("数字化转型测度的年度趋势（均值±标准差）", fontsize=11)
    ax.set_xlabel("年份"); ax.set_ylabel("DT_Ratio 均值")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "fig2_dt_trend.png")
    plt.close(fig)

    # ==================== 图 3：相关系数热力图 ====================
    cols = ["DT_Ratio", "DT_Ratio_W", "DT_Ratio_Expand", "DT_LDA", "DT_Embed",
            "RD", "ln_patents", "size", "lev", "roa"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    ax.set_title("主要变量相关系数矩阵", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "fig3_correlation_heatmap.png")
    plt.close(fig)

    # ==================== 图 4：系数森林图 ====================
    cs = pd.DataFrame(coef_store)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(cs))
    ax.errorbar(cs["系数"], y,
                xerr=[cs["系数"] - cs["下限"], cs["上限"] - cs["系数"]],
                fmt="o", capsize=5, color="#1e8449")
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_yticks(y); ax.set_yticklabels(cs["被解释变量"])
    ax.set_xlabel("DT_Ratio 系数（95% 置信区间）")
    ax.set_title("数字化转型对创新绩效的影响", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "fig4_coefplot.png")
    plt.close(fig)

    # ==================== 图 5：LDA 困惑度 ====================
    try:
        perp = pd.read_csv(RESULTS_TABLES / "table_lda_perplexity.csv")
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.plot(perp["K"], perp["perplexity"], marker="s", color="#6c3483")
        ax.set_xlabel("主题数 K"); ax.set_ylabel("困惑度 Perplexity")
        ax.set_title("LDA 主题数的困惑度曲线", fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_FIGURES / "fig5_lda_perplexity.png")
        plt.close(fig)
    except Exception as e:
        print("[06] 跳过图5：", e)

    # ==================== 控制台摘要 ====================
    print("=" * 62)
    print("[06] 主回归结果（DT_Ratio 系数）")
    print(main[["被解释变量", "模型", "变量", "系数", "t值", "显著性", "N", "R2"]]
          .to_string(index=False))
    print("-" * 62)
    print("[06] 机制检验：DT→技术岗占比 系数 %.4f (p=%.2e)；"
          % (m_a.params["DT_Ratio"], m_a.pvalues["DT_Ratio"]))
    print("     RD 方程中 tech_staff 系数 %.4f (p=%.2e)，DT_Ratio 系数 %.4f"
          % (m_b.params["tech_staff"], m_b.pvalues["tech_staff"],
             m_b.params["DT_Ratio"]))
    print("=" * 62)
    return df


if __name__ == "__main__":
    main()
