"""
export_figures.py
=================

Extract every plot embedded in the project's notebooks and save it as a
standalone PNG under `figures/`.

The notebooks are committed with their outputs, so the images are already
inside the .ipynb files -- this script just unpacks them. Nothing is
re-computed and no notebook is executed.

Naming
------
    <notebook name>__fig<NN>__<section it came from>.png

for example

    md_dimension_study__fig05__section-4-under-observed-m-d-the-hard-case.png

so it is always obvious which notebook and which section a figure belongs to.
The section is taken from the nearest markdown heading above the cell that
produced the plot.

Usage
-----
    python export_figures.py

Supervised research project, Dr Mathieu Gerber, University of Bristol.
"""

import base64
import json
import re
from pathlib import Path

NOTEBOOKS = [
    "kf_pf_lab.ipynb",           # Part 1 -- the 1D study
    "md_dimension_study.ipynb",  # Part 2 -- dimension regimes
    "robustness_study.ipynb",    # Part 3 -- robustness and proposals
]
OUT_DIR = Path("figures")


def slugify(text, max_len=60):
    """Turn a markdown heading into a short, filesystem-safe slug."""
    # Strip LaTeX ($...$), markdown emphasis, and backticks first, since those
    # carry no meaning in a filename.
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = text.replace("*", " ").replace("`", " ").replace("#", " ")
    text = text.lower()
    # Anything that is not a letter or digit becomes a separator.
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "figure"


def heading_of(cell):
    """Return the first markdown heading in a cell, or None."""
    for line in "".join(cell["source"]).splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def export(nb_path, out_dir):
    """Write every embedded PNG in one notebook to out_dir. Returns the names."""
    nb = json.loads(Path(nb_path).read_text())
    stem = Path(nb_path).stem
    written, section, index = [], "overview", 0

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            # Remember the most recent heading, to label figures that follow it.
            head = heading_of(cell)
            if head:
                section = slugify(head)
            continue

        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png is None:
                continue
            index += 1
            name = f"{stem}__fig{index:02d}__{section}.png"
            (out_dir / name).write_bytes(base64.b64decode(png))
            written.append(name)

    return written


PART_OF = {
    "kf_pf_lab": "Part 1 - the 1D Kalman vs particle filter study",
    "md_dimension_study": "Part 2 - dimension regimes (M vs d)",
    "robustness_study": "Part 3 - robustness, long horizons and proposals",
}


def write_index(by_notebook, out_dir):
    """Write figures/README.md so the folder is browsable on GitHub."""
    lines = ["# Figures", "",
             "Every plot from the project's notebooks, exported as standalone PNGs.",
             "Regenerate with `python export_figures.py` from the repository root.", "",
             "Filenames follow `<notebook>__fig<NN>__<section>.png`.", ""]
    for nb_path, names in by_notebook.items():
        stem = Path(nb_path).stem
        lines += [f"## `{nb_path}`", "", f"*{PART_OF.get(stem, stem)}*", "",
                  "| Figure | Section |", "|---|---|"]
        for n in names:
            # Recover the section slug from the filename for the index table.
            section = n.split("__")[-1].removesuffix(".png").replace("-", " ")
            lines.append(f"| [`{n}`]({n}) | {section} |")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    by_notebook, total = {}, 0
    for nb_path in NOTEBOOKS:
        if not Path(nb_path).exists():
            print(f"  skipping {nb_path} (not found)")
            continue
        names = export(nb_path, OUT_DIR)
        by_notebook[nb_path] = names
        total += len(names)
        print(f"{nb_path}  ->  {len(names)} figures")
        for n in names:
            print(f"    {n}")
    write_index(by_notebook, OUT_DIR)
    print(f"\n{total} figures written to {OUT_DIR}/  (plus README.md index)")


if __name__ == "__main__":
    main()
