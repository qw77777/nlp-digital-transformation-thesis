# -*- coding: utf-8 -*-
"""
01_generate_corpus.py
=====================
第一步：语料获取 —— 构造模拟语料库与企业财务面板。

【重要说明 / IMPORTANT】
本脚本生成的是 **模拟（synthetic）数据集**，用于毕业论文"预演/模拟稿"的教学演示。
真实的上市公司年报文本受版权与数据库授权限制，无法随代码公开分发；为使整个
NLP 测度构建流程可完整复现，这里用一个"潜在数字化程度 DT*"作为数据生成过程
（DGP）的核心，同时生成：
    (1) 企业-年度 MD&A 模拟文本（含词典词的数字化句 + 中性经营句）
    (2) 企业-年度财务面板（规模、杠杆、ROA、研发强度、专利、无形资产占比等）
    (3) 一个第三方"数字化转型指数"（模拟），用于后续聚合效度检验
三类产出彼此通过 DT* 关联，从而保证后续"词典法 / 词嵌入 / 主题模型"三条
特征提取路线能够真实地跑通并得到可解释的结果。

输出：
    data/raw/corpus_mda.csv          企业-年度 MD&A 模拟文本
    data/raw/financial_panel.csv     企业-年度财务面板（含外部指数）
"""

import numpy as np
import pandas as pd

from nlp_utils import (
    DATA_RAW, RANDOM_SEED,
    NEUTRAL_SENTENCES, DIGITAL_SENTENCES,
)

N_FIRMS = 400
YEARS = list(range(2018, 2024))          # 2018—2023，共 6 年
INDUSTRIES = ["电子设备", "工业制造", "医药生物", "批发零售",
              "信息技术", "交通运输", "建筑材料", "食品饮料"]

rng = np.random.default_rng(RANDOM_SEED)


def make_stock_code(i: int) -> str:
    """生成形如 000001 / 600001 的 6 位股票代码。"""
    if i % 2 == 0:
        return f"{600000 + i:06d}"
    return f"{1 + i:06d}"


def generate():
    # ---------------- 企业层面固定特征 ----------------
    firm_ids = [make_stock_code(i) for i in range(N_FIRMS)]
    industry = rng.choice(INDUSTRIES, size=N_FIRMS)
    soe = rng.binomial(1, 0.35, size=N_FIRMS)

    # 企业"数字化禀赋"：不同企业天然处在不同水平（信息技术、电子设备偏高）
    ind_bonus = {
        "信息技术": 0.8, "电子设备": 0.6, "工业制造": 0.2, "医药生物": 0.1,
        "批发零售": 0.0, "交通运输": -0.2, "建筑材料": -0.3, "食品饮料": -0.4,
    }
    firm_base = rng.normal(0, 1.0, size=N_FIRMS) + \
        np.array([ind_bonus[x] for x in industry])
    firm_size_base = rng.normal(22.2, 1.2, size=N_FIRMS)
    firm_age = rng.integers(5, 31, size=N_FIRMS)

    rows_text, rows_fin = [], []
    for fi in range(N_FIRMS):
        code = firm_ids[fi]
        for t in YEARS:
            # ---------- 1. 潜在数字化程度 DT*（随时间上升）----------
            trend = 0.20 * (t - 2018)              # 数字化转型的整体时间趋势
            dt_star = firm_base[fi] + trend + rng.normal(0, 0.55)
            # 映射到 [0,1]：用标准正态 CDF 的近似，得到可比的分位标度
            dt_index = float(np.clip(0.5 + 0.5 * np.tanh(dt_star / 1.6), 0.01, 0.99))

            # ---------- 2. 生成 MD&A 模拟文本 ----------
            # 句子不重复抽取（一家公司的年报不会重复同一句话），降低组合噪声
            n_neutral = int(rng.integers(9, 12))
            p_digital = 0.04 + 0.86 * dt_index
            n_digital = int(rng.binomial(18, p_digital))
            sents = list(rng.choice(NEUTRAL_SENTENCES,
                                    size=min(n_neutral, len(NEUTRAL_SENTENCES)),
                                    replace=False))
            if n_digital > 0:
                d_idx = rng.choice(len(DIGITAL_SENTENCES),
                                   size=min(n_digital, len(DIGITAL_SENTENCES)),
                                   replace=False)
                sents += [DIGITAL_SENTENCES[k] for k in d_idx]
            rng.shuffle(sents)
            mda_text = "管理层讨论与分析。" + "。".join(sents) + "。"

            rows_text.append({"firm_id": code, "year": t, "mda_text": mda_text})

            # ---------- 3. 财务与其他企业特征 ----------
            size = firm_size_base[fi] + rng.normal(0, 0.12)
            lev = float(np.clip(rng.normal(0.42, 0.18), 0.05, 0.92))
            roa = float(np.clip(rng.normal(0.045, 0.055), -0.20, 0.25))
            age = int(firm_age[fi] + (t - 2018))
            growth = float(np.clip(rng.normal(0.10, 0.20), -0.45, 0.90))
            # 无形资产占比（聚合效度参照之一）
            intangible = float(np.clip(0.06 + 0.035 * dt_index + rng.normal(0, 0.025),
                                       0.005, 0.45))
            # 数字技术岗位/研发人员占比
            tech_staff = float(np.clip(0.10 + 0.18 * dt_index + rng.normal(0, 0.05),
                                       0.01, 0.70))
            # 数字化转型对创新的转化效率存在异质性：
            # 非国有企业机制更灵活、高技术行业技术吸收能力更强，转化系数更高
            HIGH_TECH = industry[fi] in ("信息技术", "电子设备", "医药生物")
            het = 1.0 + 0.45 * (1 - soe[fi]) + 0.25 * (1 if HIGH_TECH else 0)
            # 研发投入强度（被解释变量之一）
            rd_intensity = float(np.clip(
                0.014 + 0.042 * het * dt_index + 0.004 * (size - 22.2)
                + rng.normal(0, 0.010), 0.0, 0.30))
            # 专利申请数（被解释变量之二，计数型）
            patents = int(max(0, round(np.exp(
                0.5 + 0.90 * het * dt_index + 0.28 * (size - 22.2)
                + rng.normal(0, 0.50)))))
            # 第三方数字化转型指数（模拟的"外部评级"，聚合效度参照）
            dt_rating = float(np.clip(30 + 45 * dt_index + rng.normal(0, 9), 0, 100))

            rows_fin.append({
                "firm_id": code, "year": t, "industry": industry[fi],
                "soe": int(soe[fi]), "size": round(size, 4), "lev": round(lev, 4),
                "roa": round(roa, 4), "age": age, "growth": round(growth, 4),
                "intangible": round(intangible, 4), "tech_staff": round(tech_staff, 4),
                "rd_intensity": round(rd_intensity, 4), "patents": patents,
                "dt_rating": round(dt_rating, 2),
                # 记录潜在真值，便于校验测度是否"测得准"，正式分析中不使用
                "dt_star_true": round(dt_star, 4),
            })

    df_text = pd.DataFrame(rows_text)
    df_fin = pd.DataFrame(rows_fin)
    df_text.to_csv(DATA_RAW / "corpus_mda.csv", index=False, encoding="utf-8-sig")
    df_fin.to_csv(DATA_RAW / "financial_panel.csv", index=False, encoding="utf-8-sig")
    print(f"[01] 语料：{len(df_text)} 条企业-年度文本；"
          f"平均长度 {df_text['mda_text'].str.len().mean():.0f} 字")
    print(f"[01] 财务面板：{df_fin.shape[0]} 行 × {df_fin.shape[1]} 列，"
          f"{df_fin['firm_id'].nunique()} 家公司")
    return df_text, df_fin


if __name__ == "__main__":
    generate()
