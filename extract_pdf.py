from pypdf import PdfReader
import pandas as pd

reader = PdfReader("penal_code.pdf")
page = reader.pages[3]


def visitor_body(text, cm, tm, font_dict, font_size):
    y = tm[5]
    if y > 50 and y < 1000:
        parts.append(text)
        parts_meta.append(
            {
                "text": text,
                "tm": tm,
                "cm": cm,
                "font_dict": font_dict,
                "font_size": font_size,
            }
        )


for page in reader.pages[14:]:
    parts_meta = []
    parts = []

    _ = page.extract_text(visitor_text=visitor_body)
    text_body = "".join(parts)

    for num, line in enumerate(text_body.splitlines()):
        # check if line starts with number and [
        # check if line contains ".—
        # check for chapter break
        # check for Illustrations
        # exclude references.
        print(num, line)
    break
