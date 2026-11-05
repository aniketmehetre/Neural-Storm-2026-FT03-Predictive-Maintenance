#!/usr/bin/env python3
"""demo.py — Interactive Demo, Screen-Recording Guide, & GIF Generator.

Team void main | Neural Storm 2026 Hackathon (FT-03 Predictive Maintenance)

Features:
1. Automated & Guided Browser Walkthrough for live screen recording.
2. Narrator Pitch Script with timed cues for judges.
3. Live Backend API Verification (FastAPI REST, SSE, and LangGraph Agent).
4. Headless Demo GIF Generator updating docs/img/frontend_demo.gif.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

ROOT = Path(__file__).resolve().parent
FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000"

# ── Mission Chapter Walkthrough Data & Speaking Script ───────────────────────
CHAPTER_SCRIPTS = [
    {
        "id": "ch1",
        "p": 0.00,
        "title": "CHAPTER 1: ORBITAL INSERTION (p=0.00)",
        "badge": "CH 01 // ORBITAL INSERTION",
        "headline": "Deploy. Orient. Transmit.",
        "duration": 6,
        "action": "Observe intact satellite in deep space, Earth limb glow, and live backend connection.",
        "narration": (
            "\"Welcome to the Aerospace Satellite Digital Twin for Predictive Maintenance, "
            "built by Team void main for Neural Storm 2026. Here in Chapter 1, our Earth-observation "
            "satellite is in nominal geostationary orbit. In the top bar, you can see our live "
            "FastAPI backend connection pill is green on port 8000, and our dual solar array wings "
            "are deployed in technical aerospace titanium-grey.\""
        ),
    },
    {
        "id": "live_stream",
        "p": 0.18,
        "title": "LIVE BACKEND IOT REPLAY & SSE STREAMING",
        "badge": "LIVE TELEMETRY STREAM",
        "headline": "Real-time IoT Telemetry Ingestion",
        "duration": 7,
        "action": "Click '▶ LIVE IOT STREAM' in the top bar to stream live IoT cycles from FastAPI.",
        "narration": (
            "\"By clicking the 'LIVE IOT STREAM' button in the top bar, our frontend connects directly "
            "to the FastAPI Server-Sent Events endpoint at /stream/1. Real-time sensor readings stream "
            "into causal 30-cycle rolling buffers, dynamically driving the screen-projected mini-waveforms "
            "and live cycle counters on our 3D subsystem callouts.\""
        ),
    },
    {
        "id": "ch3",
        "p": 0.35,
        "title": "CHAPTER 3: COMPONENT REVEAL & AI MRO AGENT (p=0.35)",
        "badge": "CH 03 // COMPONENT REVEAL",
        "headline": "Aerospace digital twin architecture.",
        "duration": 7,
        "action": "Click on any 3D subsystem to inspect engineering specs and trigger the AI MRO Agent.",
        "narration": (
            "\"In Chapter 3, the camera zooms into the spacecraft body. Clicking any component—such as "
            "the Main Bus, Solar Wings, Parabolic High-Gain Dish, or Earth Observation Payload—opens "
            "the Aerospace Engineering Panel. Clicking 'MRO AI STRATEGY' triggers an automated FastAPI call, "
            "generating an instant procurement order, SKU list, and downtime estimate from our AI maintenance agent.\""
        ),
    },
    {
        "id": "ch4",
        "p": 0.55,
        "title": "CHAPTER 4: EXPLODED ASSEMBLY & ANOMALY DETECTION (p=0.55)",
        "badge": "CH 04 // EXPLODED ASSEMBLY",
        "headline": "Subsystems separate along flight axes.",
        "duration": 7,
        "action": "Observe exploded subsystem separation and PCA + Isolation Forest detection card.",
        "narration": (
            "\"As degradation onset begins, the satellite separates into an exploded assembly along its flight axes. "
            "Our unsupervised anomaly detector—combining PCA reconstruction error and Isolation Forest with CUSUM "
            "change-point detection—flags degradation onset with a median lead time of 164.5 cycles to failure, "
            "alerting operators before RUL drops below 80 cycles in 98.9% of cases.\""
        ),
    },
    {
        "id": "ch5",
        "p": 0.72,
        "title": "CHAPTER 5: CALIBRATED SPLIT-CONFORMAL UNCERTAINTY (p=0.72)",
        "badge": "CH 05 // DEEP INSPECTION",
        "headline": "Calibrated split-conformal telemetry.",
        "duration": 6,
        "action": "Inspect the calibrated conformal fan band [p10, p90] on the right HUD overlay.",
        "narration": (
            "\"Point predictions can be dangerous in mission-critical aerospace. Our system derives finite-sample "
            "calibrated 80% split-conformal prediction intervals [p10, p90], empirically verified at 84.0% coverage "
            "on held-out engines, computing the exact mission failure probability P(failure within 30 cycles).\""
        ),
    },
    {
        "id": "ch6",
        "p": 0.85,
        "title": "CHAPTER 6: DECIDE WITH DOLLARS ($29.54M FLEET SAVINGS) (p=0.85)",
        "badge": "CH 06 // DECIDE WITH DOLLARS",
        "headline": "Zero in-service failures.",
        "duration": 7,
        "action": "Examine the cost-optimal policy card and OR-Tools CP-SAT scheduler comparison bars.",
        "narration": (
            "\"In Chapter 6, we translate ML predictions directly into business value. Our cost-optimal replacement "
            "policy paired with an OR-Tools CP-SAT hangar scheduler achieved zero in-service failures across a "
            "100-engine fleet backtest, saving $29,542,700—an 84.4% reduction in maintenance expenditure over "
            "reactive run-to-failure.\""
        ),
    },
    {
        "id": "ch7",
        "p": 1.00,
        "title": "CHAPTER 7: OPERATIONAL DAWN & AUDIT LEDGER (p=1.00)",
        "badge": "CH 07 // REASSEMBLY & DAWN",
        "headline": "Predict. Prove. Prevent.",
        "duration": 6,
        "action": "Watch the spacecraft reassemble at dawn with SHA-256 cryptographic ledger validation.",
        "narration": (
            "\"Finally, the spacecraft reassembles operational at dawn. Every diagnostic prediction, numeric critic "
            "check, and automated purchase order is permanently anchored in a SHA-256 tamper-evident cryptographic "
            "audit ledger. Predict. Prove. Prevent. Thank you!\""
        ),
    },
]


def check_servers() -> tuple[bool, bool]:
    """Check if Vite and FastAPI servers are responding."""
    frontend_ok = False
    backend_ok = False

    try:
        r = requests.get(f"{FRONTEND_URL}/", timeout=1.5)
        frontend_ok = r.status_code == 200
    except Exception:
        pass

    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        backend_ok = r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        pass

    return frontend_ok, backend_ok


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║       ⚡ SATELLITE DIGITAL TWIN & PREDICTIVE MAINTENANCE SYSTEM ⚡            ║
    ║               Neural Storm 2026 Hackathon · Team: void main                  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def run_backend_verification():
    """Verify all FastAPI backend endpoints live and display metrics."""
    console.print(Panel("[bold green]Running Live FastAPI Backend Verification...[/bold green]"))

    table = Table(title="Backend Endpoint Verification Table", header_style="bold cyan")
    table.add_column("Endpoint", style="bold")
    table.add_column("HTTP Method", style="yellow")
    table.add_column("Status", style="bold green")
    table.add_column("Returned Data / Metric", style="white")

    # 1. Health
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        table.add_row("/health", "GET", f"HTTP {r.status_code}", str(r.json()))
    except Exception as e:
        table.add_row("/health", "GET", "FAILED", str(e))

    # 2. Model Metadata
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=2)
        meta = r.json()
        summary = f"Model: {meta.get('model_type')} | Val RMSE: {meta.get('val_rmse')} | Features: {meta.get('features')}"
        table.add_row("/", "GET", f"HTTP {r.status_code}", summary)
    except Exception as e:
        table.add_row("/", "GET", "FAILED", str(e))

    # 3. AI Maintenance Strategy
    try:
        r = requests.post(
            f"{BACKEND_URL}/maintenance-strategy",
            json={
                "unit_id": 1,
                "predicted_rul": 24.5,
                "dominant_sensor": "sensor_11",
                "operating_cycle": 185,
            },
            timeout=3,
        )
        data = r.json()
        po = data.get("parts_order", {})
        summary = f"Alert: {data.get('alert_level')} | PO: {po.get('order_id')} | Total: ${po.get('total_estimated_cost_usd', 0):,}"
        table.add_row("/maintenance-strategy", "POST", f"HTTP {r.status_code}", summary)
    except Exception as e:
        table.add_row("/maintenance-strategy", "POST", "FAILED", str(e))

    console.print(table)


def find_browser_executable() -> str | None:
    """Find local Chrome or Edge path for clean app launch."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_live_walkthrough(auto_advance: bool = False):
    """Launch the browser and guide the user through recording the demo."""
    frontend_ok, backend_ok = check_servers()
    if not frontend_ok or not backend_ok:
        console.print(f"[bold red]Servers status: Frontend ({'ONLINE' if frontend_ok else 'OFFLINE'}), Backend ({'ONLINE' if backend_ok else 'OFFLINE'})[/bold red]")
        console.print("[yellow]Please ensure both servers are running: `npx vite` and `uvicorn src.api.main:app`[/yellow]")
        return

    browser_exe = find_browser_executable()

    console.print(Panel(
        "[bold cyan]🎙️ SCREEN RECORDING INSTRUCTIONS FOR JUDGES PITCH[/bold cyan]\n\n"
        "1. Open your screen recorder:\n"
        "   • [bold green]Windows Game Bar:[/bold green] Press [bold white]Win + Alt + R[/bold white] to start recording instantly.\n"
        "   • [bold green]OBS Studio:[/bold green] Set source to Fullscreen or Browser window (1920x1080 recommended).\n\n"
        "2. The demo script will open the 3D Satellite Digital Twin in your browser.\n"
        "3. Read aloud the highlighted [bold yellow]\"Pitch Narration Script\"[/bold yellow] at each step.\n"
        "4. Press [bold green]ENTER[/bold green] to advance to the next mission chapter.\n"
    ))

    input("👉 Press ENTER to launch the browser and start the demo walkthrough...")

    # Open first chapter
    first_url = f"{FRONTEND_URL}/?noloader=1&p=0.0"
    if browser_exe:
        subprocess.Popen([browser_exe, f"--app={first_url}", "--window-size=1920,1080"])
    else:
        webbrowser.open(first_url)

    time.sleep(2)

    for i, step in enumerate(CHAPTER_SCRIPTS, 1):
        target_url = f"{FRONTEND_URL}/?noloader=1&p={step['p']}"

        console.rule(f"[bold cyan]STEP {i}/{len(CHAPTER_SCRIPTS)}: {step['title']}[/bold cyan]")
        console.print(f"[bold green]▶ Target URL:[/bold green] {target_url}")
        console.print(f"[bold magenta]👁️ Visual Action:[/bold magenta] {step['action']}")
        console.print(Panel(step['narration'], title=f"[bold yellow]🎙️ Read Aloud (Chapter {i})[/bold yellow]", border_style="yellow"))

        if auto_advance:
            console.print(f"[dim]Auto-advancing in {step['duration']} seconds...[/dim]")
            time.sleep(step['duration'])
        else:
            cmd = input(f"👉 Press [ENTER] to advance to the next chapter (or type 'open' to open URL): ").strip()
            if cmd.lower() == 'open':
                webbrowser.open(target_url)

    console.print(Panel("[bold green]🎉 Demo Walkthrough Completed Successfully![/bold green]\nStop your screen recording (`Win + Alt + R`) and save the video!"))


def generate_demo_gif():
    """Capture high-resolution frames using headless Chrome and compile docs/img/frontend_demo.gif."""
    browser_exe = find_browser_executable()
    if not browser_exe:
        console.print("[bold red]Chrome or Edge executable not found for headless GIF generation.[/bold red]")
        return

    frontend_ok, _ = check_servers()
    if not frontend_ok:
        console.print("[bold red]Vite frontend must be running on http://localhost:5173 to generate demo GIF.[/bold red]")
        return

    gif_path = ROOT / "docs" / "img" / "frontend_demo.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = ROOT / "docs" / "img" / "_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    progress_p_values = [0.00, 0.18, 0.35, 0.55, 0.72, 0.85, 1.00]
    frame_images = []

    console.print(Panel("[bold cyan]📸 Generating High-Resolution Demo Frames for README GIF...[/bold cyan]"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("[cyan]Capturing 3D Satellite Twin frames...", total=len(progress_p_values))

        for idx, p_val in enumerate(progress_p_values):
            frame_file = temp_dir / f"frame_{idx:02d}.png"
            target_url = f"{FRONTEND_URL}/?noloader=1&p={p_val}"

            # Capture frame via Chrome Headless
            cmd = [
                browser_exe,
                "--headless=new",
                "--virtual-time-budget=3500",
                "--window-size=1280,720",
                f"--screenshot={frame_file}",
                target_url,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if frame_file.exists():
                img = Image.open(frame_file)
                # Resize slightly for optimized README loading speed
                img_resized = img.resize((960, 540), Image.Resampling.LANCZOS)
                frame_images.append(img_resized)

            progress.update(task, advance=1)

    if frame_images:
        console.print(f"[bold cyan]Assembling {len(frame_images)} frames into animated GIF...[/bold cyan]")
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=2000,  # 2.0s per frame
            loop=0,
            optimize=True,
        )
        console.print(Panel(f"[bold green]✨ Successfully generated demo GIF:[/bold green] {gif_path} ({os.path.getsize(gif_path):,} bytes)"))

        # Clean up temporary frame files
        for f in temp_dir.glob("*.png"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Engines Health Monitor Demo Walkthrough & Recording Tool")
    parser.add_argument("--auto", action="store_true", help="Auto-advance through walkthrough chapters without waiting for Enter")
    parser.add_argument("--gif", action="store_true", help="Generate updated docs/img/frontend_demo.gif from live frontend")
    parser.add_argument("--verify", action="store_true", help="Verify live FastAPI backend endpoints")

    args = parser.parse_args()

    print_banner()

    if args.verify:
        run_backend_verification()
        return

    if args.gif:
        generate_demo_gif()
        return

    console.print("[bold yellow]Select Demo Mode:[/bold yellow]")
    console.print("  [bold cyan]1)[/bold cyan] 🎥 Guided Screen-Recording Walkthrough (Recommended for recording README video)")
    console.print("  [bold cyan]2)[/bold cyan] 📸 Automatically Generate New Demo GIF for README (docs/img/frontend_demo.gif)")
    console.print("  [bold cyan]3)[/bold cyan] ⚡ Verify Live FastAPI Backend & AI Strategy API")
    console.print("  [bold cyan]4)[/bold cyan] 🚀 Full Automated Walkthrough (Auto-advancing timer)")

    choice = input("\nEnter choice [1-4] (default 1): ").strip()

    if choice == "2":
        generate_demo_gif()
    elif choice == "3":
        run_backend_verification()
    elif choice == "4":
        run_live_walkthrough(auto_advance=True)
    else:
        run_live_walkthrough(auto_advance=False)


if __name__ == "__main__":
    main()
