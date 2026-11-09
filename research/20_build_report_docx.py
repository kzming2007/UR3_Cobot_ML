"""Build the two-column Word report from its Markdown source: python -X utf8 20_build_report_docx.py <report.md>"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL, WD_LINE_SPACING
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

KO, EN = "Malgun Gothic", "Times New Roman"
COLUMN_PT, GUTTER_PT = 250, 11
# name: font, size, bold, align, line pitch, first-line indent, space before, space after
STYLES = {
    "Normal": (KO, 9, False, AL.JUSTIFY, 12, 8, 0, 0),
    "Heading 1": (KO, 10, True, AL.CENTER, 14, 0, 10, 4),
    "Heading 2": (KO, 9, True, AL.LEFT, 12, 0, 6, 2),
    "Title KO": (KO, 13, True, AL.CENTER, 20, 0, 0, 4),
    "Title EN": (EN, 13, True, AL.CENTER, 16.5, 0, 12, 3),
    "Author KO": (KO, 9, False, AL.CENTER, 14, 0, 0, 0),
    "Author EN": (EN, 9, False, AL.CENTER, 12, 0, 0, 0),
    "Abstract Head": (EN, 10, True, AL.CENTER, 12, 0, 6, 0),
    "Abstract": (EN, 10, False, AL.JUSTIFY, 12, 8, 0, 0),
    "Key Words": (EN, 10, False, AL.JUSTIFY, 12, 0, 0, 6),
    "Figure": (KO, 9, False, AL.CENTER, None, 0, 4, 2),
    "Caption KO": (KO, 8, False, AL.CENTER, 10, 0, 4, 0),
    "Caption EN": (EN, 8, False, AL.CENTER, 10, 0, 0, 2),
    "Table Text": (EN, 7, False, AL.LEFT, 9, 0, 0, 0),
    "Table Note": (EN, 6.5, False, AL.JUSTIFY, 8, 0, 2, 8),
    "Reference": (EN, 8, False, AL.LEFT, 10, 0, 0, 0),
    "Link": (EN, 7, False, AL.LEFT, 9, 0, 0, 2),  # small enough to keep the repository address on one line
}
# Column widths in points (each table fills the column) and the number of left-aligned text columns.
TABLES = {
    1: ([44, 26, 28, 30, 30, 32, 30, 30], 1),
    2: ([36, 46, 61, 53, 54], 2),
    3: ([36, 54, 52, 54, 54], 2),
    4: ([36, 62, 44, 58, 50], 2),
    5: ([66, 10, 22, 22, 22, 22, 22, 22, 21, 21], 1),
    6: ([36, 56, 36, 32, 30, 30, 30], 2),
    7: ([40, 42, 42, 42, 42, 42], 1),
    8: ([36, 50, 55, 55, 54], 2),
}


def set_fonts(rpr, font: str) -> None:
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.attrib.clear()
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), font)


def define_styles(doc) -> None:
    existing = {s.name for s in doc.styles}
    for name, (font, size, bold, align, pitch, first, before, after) in STYLES.items():
        style = doc.styles[name] if name in existing else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        set_fonts(style.element.get_or_add_rPr(), font)
        style.font.size, style.font.bold, style.font.italic = Pt(size), bold, False
        style.font.color.rgb = RGBColor(0, 0, 0)
        fmt = style.paragraph_format
        fmt.alignment, fmt.first_line_indent = align, Pt(first)
        fmt.space_before, fmt.space_after = Pt(before), Pt(after)
        if pitch:
            fmt.line_spacing_rule, fmt.line_spacing = WD_LINE_SPACING.EXACTLY, Pt(pitch)
        else:
            fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    for name in ("Heading 1", "Heading 2", "Caption KO"):
        doc.styles[name].paragraph_format.keep_with_next = True
    ref = doc.styles["Reference"].paragraph_format
    ref.left_indent, ref.first_line_indent = Pt(14), Pt(-14)
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:eastAsia"), "ko-KR")
    doc.styles["Normal"].element.get_or_add_rPr().append(lang)


def set_columns(section, count: int) -> None:
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(GUTTER_PT * 20))


def add_page_number(doc, section) -> None:
    para = section.footer.paragraphs[0]
    para.style = doc.styles["Author EN"]
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = para.add_run()
        if kind:
            node = OxmlElement("w:fldChar")
            node.set(qn("w:fldCharType"), kind)
        else:
            node = OxmlElement("w:instrText")
            node.text = text
        run._r.append(node)


def add_text(doc, text: str, style: str):
    para = doc.add_paragraph(style=style)
    for i, part in enumerate(text.split("**")):
        if part:
            para.add_run(part).bold = True if i % 2 else None
    return para


def add_link(doc, url: str) -> None:
    para = doc.add_paragraph(style="Link")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), para.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = para.add_run(url)
    run.underline = True
    run.font.color.rgb = RGBColor(0x05, 0x63, 0xC1)
    link.append(run._r)
    para._p.append(link)


def border(cell, *edges: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in edges:
        node = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "6"), ("space", "0"), ("color", "000000")):
            node.set(qn(f"w:{key}"), value)
        borders.append(node)


def add_table(doc, block: str, number: int) -> None:
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in block.splitlines()
        if not re.match(r"^\|[-| ]+\|$", line)
    ]
    widths, text_cols = TABLES[number]
    assert sum(widths) == COLUMN_PT and all(len(row) == len(widths) for row in rows), f"table {number} shape"
    table = doc.add_table(rows=len(rows), cols=len(widths))
    table.alignment, table.autofit = WD_TABLE_ALIGNMENT.CENTER, False
    margins = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), "30")
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    table._tbl.tblPr.append(margins)
    for j, width in enumerate(widths):
        table.columns[j].width = Pt(width)
    for i, row in enumerate(rows):
        table.rows[i]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.width = Pt(widths[j])
            para = cell.paragraphs[0]
            para.style = doc.styles["Table Text"]
            para.alignment = AL.LEFT if j < text_cols else AL.CENTER
            para.paragraph_format.space_before = para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.keep_with_next = True  # every table is followed by its note
            run = para.add_run(value)
            run.bold = True if i == 0 else None
            if i == 0:
                border(cell, "top", "bottom")
            if i == len(rows) - 1:
                border(cell, "bottom")
            if "Det." in rows[0][j]:  # sets detection columns apart from the alert columns in Table 5
                shade = OxmlElement("w:shd")
                for key, value in (("val", "clear"), ("color", "auto"), ("fill", "EDEDED")):
                    shade.set(qn(f"w:{key}"), value)
                cell._tc.get_or_add_tcPr().append(shade)
    # A blank first cell continues the group name above, so the name can wrap inside the merged cell.
    for i, row in enumerate(rows):
        below_blank = i + 1 < len(rows) and not rows[i + 1][0]
        if i and (not row[0] or below_blank):
            merge = OxmlElement("w:vMerge")
            if row[0]:
                merge.set(qn("w:val"), "restart")
            table.rows[i]._tr.tc_lst[0].tcPr.insert(1, merge)


def add_figure(doc, alt: str, path: Path) -> None:
    para = doc.add_paragraph(style="Figure")
    para.paragraph_format.keep_with_next = True
    shape = para.add_run().add_picture(str(path), width=Pt(COLUMN_PT))
    shape._inline.docPr.set("descr", alt)


def build(source: Path, out: Path) -> None:
    blocks = [b.strip() for b in source.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n\n") if b.strip()]
    body = next(i for i, b in enumerate(blocks) if b.startswith("## "))
    front, blocks = blocks[:body], blocks[body:]

    doc = Document()
    define_styles(doc)
    section = doc.sections[0]
    section.page_width, section.page_height = Pt(595.3), Pt(841.9)
    section.left_margin, section.right_margin = Pt(42.6), Pt(42.7)
    section.top_margin, section.bottom_margin, section.footer_distance = Pt(45), Pt(62), Pt(44)
    set_columns(section, 1)
    add_page_number(doc, section)

    title_ko, author_ko, affil_ko, title_en, author_en, affil_en, head, abstract, keywords = front
    add_text(doc, title_ko[2:], "Title KO")
    add_text(doc, author_ko, "Author KO")
    add_text(doc, affil_ko, "Author KO")
    add_text(doc, title_en[2:], "Title EN")
    add_text(doc, author_en, "Author EN")
    add_text(doc, affil_en, "Author EN")
    add_text(doc, head, "Abstract Head")
    add_text(doc, abstract, "Abstract")
    add_text(doc, keywords.replace("Key words:", "**Key words:**", 1), "Key Words")
    set_columns(doc.add_section(WD_SECTION.CONTINUOUS), 2)

    table_number, after_table = 0, False
    for block in blocks:
        if after_table:
            add_text(doc, block, "Table Note")
            after_table = False
        elif block.startswith("## "):
            doc.add_paragraph(block[3:], style="Heading 1")
        elif block.startswith("### "):
            doc.add_paragraph(block[4:], style="Heading 2")
        elif match := re.match(r"!\[(.*)\]\((.*)\)", block):
            add_figure(doc, match[1], source.parent / match[2])
        elif match := re.match(r"(표|그림) (\d+)\.", block):
            korean, english = block.split("\n")
            add_text(doc, korean, "Caption KO")
            caption = add_text(doc, english, "Caption EN")
            if match[1] == "표":
                table_number = int(match[2])
                caption.paragraph_format.keep_with_next = True
            else:
                caption.paragraph_format.space_after = Pt(8)
        elif block.startswith("|"):
            add_table(doc, block, table_number)
            after_table = True
        elif block.startswith("https://"):
            add_link(doc, block)
        elif re.match(r"\[\d+\] ", block):
            add_text(doc, block, "Reference")
        else:
            add_text(doc, block, "Normal")

    set_columns(doc.add_section(WD_SECTION.CONTINUOUS), 1)  # the break balances the two columns of the last page
    compat = doc.settings.element.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        doc.settings.element.append(compat)
    setting = OxmlElement("w:compatSetting")
    for key, value in (("name", "compatibilityMode"), ("uri", "http://schemas.microsoft.com/office/word"), ("val", "15")):
        setting.set(qn(f"w:{key}"), value)
    compat.append(setting)
    props = doc.core_properties
    props.title, props.author, props.comments, props.last_modified_by = title_ko[2:], "", "", ""
    props.created = props.modified = dt.datetime.now()
    doc.save(out)
    print(out)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    build(src, src.with_suffix(".docx"))
