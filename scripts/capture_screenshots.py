"""
capture_screenshots.py – Drive the running app with synthetic client records and screenshot it.

Start the app first (with a throwaway vault), then:

    python scripts/capture_screenshots.py http://localhost:5000

Writes docs/images/ui-*.png. Every value in the records is made up.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

RECORDS = """Client Name: Margaret Ellison
DOB: 03/14/1961
SSN: 536-22-1874
Email: m.ellison@example.com
Phone: (415) 555-0142
Account number: 004512339871
Address: 1180 Harbor View Dr, Sausalito, CA 94965
Advisor: Daniel Brooks
Notes: Wants to roll her 401(k) into an IRA before retiring next year.

Client Name: Rajesh Iyer
DOB: 09/02/1978
SSN: 287-41-0954
Email: rajesh.iyer@example.com
Phone: (212) 555-0187
Account number: 778201945563
Address: 45 Park Terrace E, New York, NY 10034
Advisor: Daniel Brooks
Notes: Moderate risk; asked about municipal bonds for the taxable account."""

QUESTION = "What is Margaret Ellison's account number, and who is her advisor?"


def main(base_url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(base_url)
        page.wait_for_selector("#ingest-data")

        page.fill("#ingest-data", RECORDS)
        page.click("#ingest-btn")
        page.wait_for_selector("#result-modal.active", timeout=120_000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "ui-ingest.png"))
        page.click("#modal-close-btn")

        page.fill("#query-input", QUESTION)
        page.click("#query-btn")
        page.wait_for_selector("#result-modal.active", timeout=120_000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "ui-answer.png"))
        page.click("#modal-close-btn")
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "ui-dashboard.png"))

        for view in ("vault", "audit", "policies"):
            page.click(f"#nav-{view}")
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"ui-{view}.png"))
            print(f"ui-{view}.png")
        browser.close()
    print(f"Screenshots in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000")
