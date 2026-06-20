#!/usr/bin/env python3
"""Capture real screenshots of the Flask app for the manuscript using Playwright.
Saves: figures/app_dashboard.png, app_video_form.png, app_video_result.png."""
import time, subprocess, sys, os, socket
from playwright.sync_api import sync_playwright

PY = sys.executable
os.makedirs("figures", exist_ok=True)


def wait_port(host, port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


srv = subprocess.Popen([PY, "webapp/app.py"],
                       stdout=open("webapp/server.log", "w"), stderr=subprocess.STDOUT)
try:
    assert wait_port("127.0.0.1", 5000, 40), "server did not start"
    time.sleep(2)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1180, "height": 900},
                        device_scale_factor=2)

        # --- dashboard ---
        pg.goto("http://127.0.0.1:5000/", wait_until="networkidle")
        pg.wait_for_timeout(1500)
        pg.screenshot(path="figures/app_dashboard.png")

        # --- video selector form ---
        pg.goto("http://127.0.0.1:5000/video", wait_until="networkidle")
        pg.wait_for_function("document.querySelectorAll('#subject option').length>0")
        pg.select_option("#subject", "11")
        pg.wait_for_timeout(400)
        pg.select_option("#exercise", "06")
        pg.select_option("#camera", "cam4")
        pg.select_option("#side", "r")
        pg.select_option("#model", "yolo11n-pose.pt")
        pg.wait_for_timeout(300)
        pg.locator(".card").first.screenshot(path="figures/app_video_form.png")

        # --- run and capture result ---
        pg.click("#go")
        pg.wait_for_selector("#out", state="visible", timeout=120000)
        pg.wait_for_function("document.getElementById('plot').complete && "
                             "document.getElementById('plot').naturalWidth>0",
                             timeout=120000)
        pg.wait_for_timeout(800)
        pg.locator("#out").screenshot(path="figures/app_video_result.png")
        b.close()
    print(">> wrote figures/app_dashboard.png, app_video_form.png, app_video_result.png")
finally:
    srv.terminate()
