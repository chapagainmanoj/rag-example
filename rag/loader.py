import re
import os
from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path: str) -> str:
    if not pdf_path:
        raise ValueError("pdf_path is None or empty. Pass --pdf <path> or set PDF_PATH in main.py.")

    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {path}\n"
            f"Current working directory: {os.getcwd()}\n"
            "Tip: use an absolute path or run from the directory containing the PDF."
        )

    with open(path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str) -> list[dict]:
    """
    Splits resume text into labeled chunks by section.
    Each chunk: { id, section, content }
    """
    section_pattern = re.compile(
        r"\n(?=Work Experience|Education|Skills|Publications|Projects|Summary|Certifications)", re.IGNORECASE
    )
    raw_sections = section_pattern.split(text)

    chunks = []
    for i, section in enumerate(raw_sections):
        section = section.strip()
        if not section:
            continue

        section_label = section.split("\n")[0][:50]

        if len(section) > 600:
            lines = section.split("\n")
            buffer, sub_idx = "", 0
            for line in lines:
                buffer += line + "\n"
                if len(buffer) >= 300:
                    chunks.append({"id": f"chunk_{i}_{sub_idx}", "content": buffer.strip(), "section": section_label})
                    buffer, sub_idx = "", sub_idx + 1
            if buffer.strip():
                chunks.append({"id": f"chunk_{i}_{sub_idx}", "content": buffer.strip(), "section": section_label})
        else:
            chunks.append({"id": f"chunk_{i}", "content": section, "section": section_label})

    return chunks


def load_and_chunk(pdf_path: str) -> list[dict]:
    print(f"Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    for c in chunks:
        print(f"  [{c['id']}] {c['section'][:40]} — {len(c['content'])} chars")
    return chunks
