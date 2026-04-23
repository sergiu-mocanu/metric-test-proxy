import re

def extract_code(text: str) -> str:
    # 1) fenced blocks
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2) fallback: return everything after first "def"
    idx = text.find('def ')
    if idx != -1:
        return text[idx:].strip()

    return text.strip()