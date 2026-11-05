"""Build the uploadable skill zip for claude.ai / Claude Desktop.

    python tools/build_skill.py          ->  dist/raw-photo-editor.zip

The zip contains one folder `raw-photo-editor/` with SKILL.md, the
`photo_editor` package (including the bundled face models) and
requirements.txt.  Upload it in Claude: Settings -> Capabilities -> Skills.
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "raw-photo-editor"
OUT = os.path.join(ROOT, "dist", NAME + ".zip")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = [(os.path.join(ROOT, ".claude", "skills", NAME, "SKILL.md"), "SKILL.md"),
             (os.path.join(ROOT, "requirements.txt"), "requirements.txt")]
    pkg = os.path.join(ROOT, "photo_editor")
    for d, _, fs in os.walk(pkg):
        if "__pycache__" in d:
            continue
        for f in fs:
            if f.endswith((".pyc", ".part")):
                continue
            full = os.path.join(d, f)
            files.append((full, os.path.relpath(full, ROOT)))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in sorted(files, key=lambda x: x[1]):
            z.write(src, f"{NAME}/{arc}")
    print(f"{OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
