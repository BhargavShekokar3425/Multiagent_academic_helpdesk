"""Conservative source-owned table passages; never reconstruct or invent cells."""
import re

def table_passages(text):
    lines = list(re.finditer(r"[^\n]*(?:\n|$)", text))
    passages = []
    for i, line in enumerate(lines):
        if not re.match(r"\s*Table\s+\d+(?:\.\d+)+\b", line.group()):
            continue
        start = line.start()
        end = None
        numeric = 0
        for later in lines[i + 1:]:
            value = later.group().strip()
            if (re.match(r"(?:Table\s+\d|[●•]|\d+\.\s+[A-Z]|\d+\.\d+\s+[A-Z])", value)
                    or later.end() - start > 4000):
                break
            if re.search(r"\d(?:\.\d+)?\s*$", value):
                numeric += 1
                end = later.end()
        if end and numeric >= 2:
            passages.append(text[start:end].strip())
    return passages

def is_table_passage(text, quote):
    norm = lambda x: " ".join(x.split())
    last = quote.strip().splitlines()[-1] if quote.strip() else ""
    full_row = any(norm(line) == norm(last) for line in text.splitlines())
    return any(norm(p) == norm(quote) or (
        norm(p).startswith(norm(quote)) and full_row and len(quote.splitlines()) >= 4
        and re.search(r"\d\s*$", quote))
        for p in table_passages(text))
