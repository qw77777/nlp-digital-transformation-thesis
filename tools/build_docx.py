# -*- coding: utf-8 -*-
"""
tools/build_docx.py
===================
把 paper/*.md 转换为排版规范的中文 Word 文档（.docx）。

支持：多级标题、段落、无序/有序列表、表格（Markdown 管道表）、图片、加粗。
用法：
    python tools/build_docx.py
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

CN_FONT = "宋体"
CN_HEI = "黑体"


def set_cn(run, font=CN_FONT, size=None, bold=None, color=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_para(doc, text, size=12, bold=False, align=None, first_indent=True,
             font=CN_FONT, space_after=6):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent and align is None:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    # 处理 **粗体**
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2])
            set_cn(r, font=font, size=size, bold=True)
        else:
            r = p.add_run(part)
            set_cn(r, font=font, size=size, bold=bold)
    return p


def parse_table(lines):
    rows = []
    for ln in lines:
        if re.match(r"^\s*\|[\s\-:|]+\|\s*$", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def main():
    md_files = sorted(PAPER.glob("*.md"))
    if not md_files:
        print("未找到 paper/*.md")
        return
    md_path = md_files[0]
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = CN_FONT
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    for s in doc.sections:
        s.top_margin = Cm(2.5); s.bottom_margin = Cm(2.5)
        s.left_margin = Cm(3.0); s.right_margin = Cm(2.6)

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        # 表格
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            rows = parse_table(block)
            if rows:
                t = doc.add_table(rows=len(rows), cols=len(rows[0]))
                t.style = "Table Grid"
                for ri, row in enumerate(rows):
                    for ci, cell in enumerate(row):
                        if ci < len(t.rows[ri].cells):
                            c = t.rows[ri].cells[ci]
                            c.text = ""
                            p = c.paragraphs[0]
                            r = p.add_run(cell)
                            set_cn(r, size=9, bold=(ri == 0))
                doc.add_paragraph()
            continue
        # 图片
        m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if m:
            img = (md_path.parent / m.group(2)).resolve()
            if img.exists():
                try:
                    doc.add_picture(str(img), width=Cm(15))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e:
                    add_para(doc, f"[图片：{m.group(1)} — {e}]")
            i += 1
            continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            if level == 1:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(title); set_cn(r, font=CN_HEI, size=18, bold=True)
                p.paragraph_format.space_after = Pt(14)
            else:
                p = doc.add_paragraph()
                r = p.add_run(title)
                set_cn(r, font=CN_HEI, size={2: 15, 3: 13.5, 4: 12}.get(level, 12), bold=True)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(6)
            i += 1
            continue
        # 引用块
        if s.startswith(">"):
            add_para(doc, s.lstrip("> ").strip(), size=10, first_indent=False)
            i += 1
            continue
        # 列表
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            txt = re.sub(r"^([-*]|\d+\.)\s+", "· ", s)
            p = add_para(doc, txt, size=12, first_indent=False)
            p.paragraph_format.left_indent = Pt(18)
            i += 1
            continue
        # 分隔线
        if re.match(r"^-{3,}$", s):
            add_para(doc, "—" * 20, align="center", size=9, first_indent=False)
            i += 1
            continue
        add_para(doc, s)
        i += 1

    out = PAPER / (md_path.stem + ".docx")
    doc.save(str(out))
    print(f"已生成：{out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        (ROOT / "_docx_err.txt").write_text(tb, encoding="utf-8")
        print(tb)
