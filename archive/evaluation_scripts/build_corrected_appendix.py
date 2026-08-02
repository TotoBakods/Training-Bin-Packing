from __future__ import annotations

import argparse
import copy
import io
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


# 1-based paragraph index in document order -> expected old text, corrected text
REPLACEMENTS: dict[int, tuple[str, str]] = {
    28: ("0.0073", "0.2276"),
    33: ("0.0290", "0.0732"),
    113: ("6,966", "6,485"),
    119: ("7,499", "7,803"),
    121: ("95.50% / 100%", "94.50% / 100%"),
    125: ("6,696", "7,387"),
    131: ("6,805", "8,393"),
    137: ("34,223", "41,437"),
    143: ("39,061", "43,817"),
    144: ("31.12%", "30.99%"),
    145: ("96.50% / 100%", "95.50% / 100%"),
    146: ("2.26%", "2.25%"),
    149: ("37,405", "44,711"),
    155: ("34,818", "42,807"),
    161: ("104,024", "127,808"),
    167: ("103,439", "116,872"),
    168: ("31.24%", "31.00%"),
    169: ("95.33% / 100%", "95.67% / 100%"),
    173: ("110,575", "128,820"),
    179: ("109,287", "128,164"),
    192: ("95.33", "95.67"),
    193: ("92.44", "91.61"),
    194: ("0.112", "0.103"),
    195: ("103.4s", "116.9s"),
    201: ("109.3s", "128.2s"),
    207: ("102.8s", "127.8s"),
    213: ("110.6s", "128.8s"),
    251: ("95.33% PSR, 92.44% BBox Eff.", "95.67% PSR, 91.61% BBox Eff."),
}


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    text_nodes = paragraph.findall(".//w:t", NS)
    if not text_nodes:
        run = paragraph.find("w:r", NS)
        if run is None:
            run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
        text_node = ET.SubElement(run, f"{{{W_NS}}}t")
        text_node.text = text
        return

    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""


def update_document_xml(document_xml: bytes) -> tuple[bytes, list[str]]:
    root = ET.fromstring(document_xml)
    all_paragraphs = root.findall(".//w:p", NS)
    paragraphs = []
    for paragraph in all_paragraphs:
        if paragraph_text(paragraph).strip():
            paragraphs.append(paragraph)
    audit_lines: list[str] = []

    for idx, paragraph in enumerate(paragraphs, start=1):
        replacement = REPLACEMENTS.get(idx)
        if not replacement:
            continue

        expected_old, corrected = replacement
        current = paragraph_text(paragraph)
        if current != expected_old:
            audit_lines.append(
                f"WARNING paragraph {idx}: expected {expected_old!r}, found {current!r}"
            )
            continue

        set_paragraph_text(paragraph, corrected)
        audit_lines.append(f"UPDATED paragraph {idx}: {expected_old!r} -> {corrected!r}")

    updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return updated_xml, audit_lines


def build_corrected_appendix(source: Path, output: Path, report: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source, "r") as zin:
        document_xml = zin.read("word/document.xml")
        updated_xml, audit_lines = update_document_xml(document_xml)

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = updated_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    if report is not None:
        content = [
            "# Corrected Appendix Build Report",
            "",
            f"- Source: `{source}`",
            f"- Output: `{output}`",
            f"- Replacement targets: `{len(REPLACEMENTS)}`",
            "",
            "## Audit",
            "",
        ]
        content.extend(f"- {line}" for line in audit_lines)
        report.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corrected appendix .docx from the downloaded source.")
    parser.add_argument(
        "--source",
        default=r"C:\Users\TotoBakod\Downloads\APPENDIX.docx",
        help="Path to the original appendix .docx",
    )
    parser.add_argument(
        "--output",
        default=str(Path("tmp") / "APPENDIX_corrected.docx"),
        help="Path for the corrected appendix .docx",
    )
    parser.add_argument(
        "--report",
        default=str(Path("tmp") / "APPENDIX_corrected_report.md"),
        help="Optional audit report path",
    )
    args = parser.parse_args()

    build_corrected_appendix(Path(args.source), Path(args.output), Path(args.report))


if __name__ == "__main__":
    main()
