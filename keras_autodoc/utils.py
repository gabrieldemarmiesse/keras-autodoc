import re
import os


def count_leading_spaces(s):
    ws = re.search(r"\S", s)
    if ws:
        return ws.start()
    else:
        return 0


def insert_in_file(markdown_text, file_path):
    """Save module page.

    Either insert content into existing page,
    or create page otherwise."""
    if file_path.exists():
        template = file_path.read_text(encoding="utf-8")
        if "{{autogenerated}}" not in template:
            raise RuntimeError(f"Template found for {file_path} but missing "
                               f"{{autogenerated}} tag.")
        markdown_text = template.replace("{{autogenerated}}", markdown_text)
        print("...inserting autogenerated content into template:", file_path)
    else:
        print("...creating new page with autogenerated content:", file_path)
    os.makedirs(file_path.parent, exist_ok=True)
    file_path.write_text(markdown_text, encoding="utf-8")


def code_snippet(snippet):
    return (
        f'```python\n'
        f'{snippet.encode("unicode_escape").decode("utf8")}\n'
        f'```\n')
