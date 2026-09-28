#!/usr/bin/env python3
"""Snake decided by EuLLM's `/v1/systemone`, as a video to post.

Plays a game with the model, the same game `snake_web.py` shows, keeps every
move and renders it with the same page, frame by frame, in a headless
browser. Three files come out, ready to upload:

    snake-wide.mp4     1920x1080, for X
    snake-tall.mp4     1080x1920, for Instagram Reels (and X, vertical)
    snake-readme.mp4   1280x720 and small, for a GitHub README

Every move on screen is the game as it was played: the model's answers,
each with the time it took, and the moves code made. Only the pace is
chosen for watching: the opening moves at a readable speed, then fast
forward, marked on screen, then the end of the game.

Besides the Python standard library it needs Playwright, a browser for it,
and the ffmpeg program:

    pip install playwright
    python -m playwright install chromium
    sudo apt install ffmpeg        # or your system's own package

Then, with `eullm serve --decision-model ...` running:

    python examples/decision-snake/snake_video.py --url http://localhost:11434

`--games 3` plays three games and keeps the highest score. The game is saved
too (`snake-game.jsonl`): `--from snake-game.jsonl` renders it again, with
another `--seconds` or `--caption`, without playing it again.
"""

import argparse
import bisect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snake_web  # noqa: E402  (the game and its page, next to this file)

FPS = 30
LAYOUTS = {"wide": (1920, 1080), "tall": (1080, 1920)}
OPENING = 5.0  # moves a second at the start, slow enough to read the options
ENDING = 10.0  # moves a second for the last moves
HOLD = 3.0  # seconds the final board stays


def play(args, seed):
    """One game with the model: every frame the page would have shown."""
    show = snake_web.Show(
        types.SimpleNamespace(
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            seed=seed,
            tick=0.0,
            wait=True,  # every answer, however long it takes
            width=args.width,
            height=args.height,
        )
    )
    frames = [show.frame]
    show.publish = frames.append  # keep the frames instead of serving them
    started = time.perf_counter()
    try:
        while show.end is None:
            show.step(show.decider.decide(show.game))
            if len(frames) % 500 == 0:
                print(
                    f"  move {len(frames) - 1}, score {show.game.score} "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    finally:
        show.decider.close()
    return frames


def schedule(moves, seconds):
    """For every video frame, the move it shows and how much faster than the
    opening the game runs there. The opening moves at `OPENING` a second, the
    middle faster and faster, up to what fits in `seconds`, the last moves
    at `ENDING` a second, then the final board for `HOLD` seconds."""
    intro = min(moves, 40)
    outro = min(moves - intro, 30)
    middle = moves - intro - outro
    budget = seconds - intro / OPENING - outro / ENDING - HOLD
    if middle and budget <= 1.0:
        needed = seconds - budget + 1.0
        raise SystemExit(f"--seconds {seconds} is too short: at least {needed:.0f}")

    def pace(u, peak):  # moves a second, u from 0 to 1 through the middle
        if u < 0.1:
            return OPENING + (peak - OPENING) * u / 0.1
        if u > 0.95:
            return peak + (ENDING - peak) * (u - 0.95) / 0.05
        return peak

    def length(peak):
        return sum(1.0 / pace((i + 0.5) / middle, peak) for i in range(middle))

    peak = OPENING
    if middle and length(OPENING) > budget:
        low, high = OPENING, 10000.0
        for _ in range(60):
            peak = (low + high) / 2
            if length(peak) > budget:
                low = peak
            else:
                high = peak
        peak = high
    durations = [1.0 / OPENING] * intro
    durations += [1.0 / pace((i + 0.5) / middle, peak) for i in range(middle)]
    durations += [1.0 / ENDING] * outro
    starts, t = [], 0.0
    for d in durations:
        starts.append(t)
        t += d
    plan = []
    for frame in range(round((t + HOLD) * FPS)):
        i = min(bisect.bisect_right(starts, frame / FPS) - 1, moves - 1)
        plan.append((i, (1.0 / durations[i]) / OPENING))
    return plan


def encoder(outputs):
    """ffmpeg, reading PNG frames on stdin, writing each (path, width, crf)."""
    command = [shutil.which("ffmpeg"), "-loglevel", "error", "-y"]
    command += ["-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-"]
    for path, width, crf in outputs:
        if width:
            command += ["-vf", f"scale={width}:-2:flags=lanczos"]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
        command += ["-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart"]
        command.append(str(path))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render(browser, frames, layout, outputs, seconds, caption):
    """Draw every video frame with the page and pipe it to ffmpeg."""
    width, height = LAYOUTS[layout]
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(snake_web.PAGE.as_uri() + "?video=" + layout)
    plan = schedule(len(frames), seconds)
    ffmpeg = encoder(outputs)
    shown, image = None, None
    started = time.perf_counter()
    try:
        for n, (i, speed) in enumerate(plan):
            badge = round(speed) if speed >= 2 else None
            if (i, badge) != shown:
                video = {"speed": badge, "caption": caption}
                page.evaluate("([f, v]) => render(f, v)", [frames[i], video])
                image = page.screenshot(type="png")
                shown = (i, badge)
            ffmpeg.stdin.write(image)
            if n and n % (FPS * 10) == 0:
                print(
                    f"  {layout}: {n // FPS} of {len(plan) // FPS} s "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        page.close()
    if ffmpeg.returncode:
        raise SystemExit(f"ffmpeg failed on {layout} (exit {ffmpeg.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://localhost:11434", help="EuLLM server URL"
    )
    parser.add_argument(
        "--model", default=None, help="decision model (default: the one loaded)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument(
        "--games", type=int, default=1, help="games to play; the best one is kept"
    )
    parser.add_argument("--seed", type=int, default=None, help="food placement seed")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="render this saved game instead of playing one",
    )
    parser.add_argument(
        "--save", default="snake-game.jsonl", help="where the game played is saved"
    )
    parser.add_argument(
        "--seconds", type=float, default=60.0, help="video length (default 60)"
    )
    parser.add_argument(
        "--caption",
        default="",
        help='a line under the title, e.g. "Jev-Style 0.8B on an RTX 5070 Ti"',
    )
    parser.add_argument(
        "--layouts",
        default="wide,tall",
        help="which videos: wide (X, README) and/or tall (Instagram)",
    )
    parser.add_argument("--out-dir", default=".", help="where the videos go")
    parser.add_argument(
        "--browser",
        default=None,
        help="a Chrome or Chromium to draw with, instead of Playwright's own",
    )
    args = parser.parse_args()
    layouts = [name for name in args.layouts.split(",") if name]
    for name in layouts:
        if name not in LAYOUTS:
            parser.error(f"unknown layout {name!r}: wide or tall")

    # Everything the rendering needs is checked before a game is played.
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found: install it (sudo apt install ffmpeg)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright not found: pip install playwright, then "
            "python -m playwright install chromium"
        ) from None

    if args.source:
        with open(args.source, encoding="utf-8") as f:
            frames = [json.loads(line) for line in f if line.strip()]
    else:
        frames = None
        for n in range(args.games):
            seed = None if args.seed is None else args.seed + n
            print(f"game {n + 1} of {args.games}", flush=True)
            game = play(args, seed)
            print(f"  score {game[-1]['score']}: {game[-1]['end']}", flush=True)
            if frames is None or game[-1]["score"] > frames[-1]["score"]:
                frames = game
        with open(args.save, "w", encoding="utf-8") as f:
            for frame in frames:
                f.write(json.dumps(frame) + "\n")
        print(f"kept the game that scored {frames[-1]['score']}: {args.save}")

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser)
        try:
            for layout in layouts:
                outputs = [(out / f"snake-{layout}.mp4", None, 18)]
                if layout == "wide":
                    outputs.append((out / "snake-readme.mp4", 1280, 30))
                render(browser, frames, layout, outputs, args.seconds, args.caption)
                for path, _, _ in outputs:
                    size = path.stat().st_size / 1e6
                    print(f"{path}: {size:.1f} MB", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
