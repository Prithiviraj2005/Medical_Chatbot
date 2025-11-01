# src/preprocess.py
import re

def clean_text(text: str) -> str:
    """
    Clean PDF/txt extraction artifacts:
    - collapse whitespace
    - fix hyphen-splits like 'gesta- tion' -> 'gestation'
    - remove stray control characters
    """
    if not text:
        return ""
    # replace non-breaking spaces, etc.
    text = text.replace('\xa0', ' ')
    # remove weird control chars
    text = re.sub(r'[\r\f\t]', ' ', text)
    # fix hyphenated line-breaks e.g. 'gesta-\n tion' or 'gesta- tion'
    text = re.sub(r'-\s*\n\s*', '', text)         # hyphen+newline
    text = re.sub(r'-\s+', '', text)              # hyphen + space artifacts
    # collapse multiple newlines to paragraph breaks
    text = re.sub(r'\n{2,}', '\n\n', text)
    # collapse remaining newlines to single space inside paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
