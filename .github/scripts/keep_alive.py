"""Visit the Streamlit demo in a real headless browser so Streamlit Community Cloud
counts it as genuine traffic, and click through the wake-up prompt if the app is asleep.

A plain HTTP GET (curl, requests) only fetches the static HTML shell and never boots the
app's Python backend, so it doesn't prevent or clear sleep. This needs real page rendering.
"""

from playwright.sync_api import sync_playwright

URL = "https://himanshumjain15-recsys.streamlit.app"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=60_000, wait_until="networkidle")

    wake_button = page.get_by_text(WAKE_BUTTON_TEXT)
    if wake_button.count() > 0:
        print("App was asleep, clicking wake-up button")
        wake_button.click()
        page.wait_for_timeout(15_000)
    else:
        print("App was already awake")

    browser.close()
