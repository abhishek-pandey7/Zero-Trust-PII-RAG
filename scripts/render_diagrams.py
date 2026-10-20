"""
render_diagrams.py – Render the README diagrams from HTML to PNG.

Each docs/diagrams/<name>.html is screenshotted at 2x into docs/images/<name>.png. Diagrams are
HTML rather than Mermaid so they stay readable: real fonts, real layout, and a solid background
that works in GitHub's light and dark themes.

    pip install playwright && playwright install chromium
    python scripts/render_diagrams.py            # all diagrams
    python scripts/render_diagrams.py architecture
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "docs" / "diagrams"
OUTPUT = ROOT / "docs" / "images"


def main(names: list[str]) -> None:
    pages = [SOURCES / f"{n}.html" for n in names] if names else sorted(SOURCES.glob("*.html"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800}, device_scale_factor=2)
        for source in pages:
            page.goto(source.as_uri())
            page.wait_for_load_state("networkidle")
            target = OUTPUT / f"{source.stem}.png"
            page.locator("#canvas").screenshot(path=str(target))
            print(f"{source.name} -> {target.relative_to(ROOT)}")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
