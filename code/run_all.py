# -*- coding: utf-8 -*-
"""
run_all.py
==========
一键复现脚本：按 NLP 测度构建五步流程，依次执行 01—06 号脚本。

用法：
    python code/run_all.py
"""

import runpy
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

STEPS = [
    ("01_generate_corpus.py", "第一步 语料获取：生成模拟 MD&A 语料与财务面板"),
    ("02_preprocessing.py", "第二步 文本预处理：清洗 / 分词 / 去停用词"),
    ("03_feature_extraction.py", "第三步 特征提取：词典法 / 词嵌入 / LDA"),
    ("04_measure_construction.py", "第四步 指标构建：L1/L2/L3 与面板"),
    ("05_validity_reliability.py", "第五步 信效度检验"),
    ("06_regression_analysis.py", "实证分析：主回归 / 稳健性 / 异质性"),
]


def main():
    t0 = time.time()
    for fname, title in STEPS:
        print("\n" + "=" * 66)
        print(f"▶ {title}")
        print("=" * 66)
        runpy.run_path(str(CODE / fname), run_name="__main__")
    print("\n" + "=" * 66)
    print(f"✅ 全部完成，用时 {time.time() - t0:.1f} 秒")
    print("   结果表：results/tables/    图：results/figures/")
    print("   面板数据：data/processed/firm_year_panel.csv")


if __name__ == "__main__":
    main()
