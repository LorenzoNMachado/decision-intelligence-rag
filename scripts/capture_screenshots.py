"""Capture the documentation screenshots from a locally running Streamlit app.

Playwright is not a project dependency; run this script in an isolated
environment, for example:

    uv run streamlit run app/streamlit_app.py --server.port 8599 --server.headless true &
    uvx --with playwright python scripts/capture_screenshots.py --url http://localhost:8599

Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use an existing Chromium build instead of
``playwright install chromium``. The app must run without DIRAG_* variables
(offline mode), so that no provider setting can appear in a screenshot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
QUESTION = "What budget was approved for the decision to trial a patient scheduling system?"


def _wait_idle(page: Page) -> None:
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000
    )
    page.wait_for_timeout(800)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        page.goto(args.url)
        page.get_by_text("Decision Intelligence RAG", exact=True).first.wait_for(timeout=120_000)
        _wait_idle(page)

        # Defaults: rules-v3 retrieval, sufficiency check on, extractive-v2 generator.
        page.get_by_label("Question", exact=True).fill(QUESTION)
        page.get_by_label("Question", exact=True).press("Enter")
        _wait_idle(page)
        page.get_by_role("button", name="Ask").click()
        page.get_by_text("Retrieved documents").wait_for(timeout=60_000)
        _wait_idle(page)
        page.screenshot(path=str(OUT / "demo-overview.png"))
        page.get_by_text("Retrieved documents").scroll_into_view_if_needed()
        _wait_idle(page)
        page.screenshot(path=str(OUT / "retrieval-evidence.png"), full_page=True)

        # Streamlit scrolls inside its own container, so a taller viewport shows both result tables.
        page.set_viewport_size({"width": 1400, "height": 1700})
        page.get_by_role("tab", name="Measured results").click()
        page.get_by_text("Protocol v3 · holdout").first.wait_for(timeout=60_000)
        _wait_idle(page)
        page.screenshot(
            path=str(OUT / "experimental-results.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 1250}
        )
        browser.close()
    for name in ("demo-overview.png", "retrieval-evidence.png", "experimental-results.png"):
        print(OUT / name, (OUT / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
