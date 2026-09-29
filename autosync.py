"""Auto-download new Instagram posts for every channel that has an Instagram
link, using gallery-dl. Drop this next to app.py.

How a run works, per linked channel:
  1. gallery-dl fetches the profile's posts into a temp folder. Its download
     archive (.data/ig-archive.sqlite) means it never re-downloads a post it
     has already handled.
  2. Every file gets a metadata sidecar; anything belonging to a PINNED post
     is thrown away.
  3. FIRST run for a channel: keep only the newest unpinned post (all of its
     photos/videos if it's a carousel). Everything else just gets marked as
     seen, so from then on only genuinely new posts show up.
     LATER runs: keep every new unpinned post.
  4. Kept files move into the channel's folder, get scanned, and are attached
     to the channel like a manual upload.
Needs the same cookies.txt the Import page uses (.data/cookies.txt).
"""
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime

from db import get_db
from importer import cookies_path, is_instagram, staging_root, tool_available
from scanner import PHOTO_EXTS, VIDEO_EXTS, data_root, library_root, scan_library

FIRST_RUN_RANGE = "1-6"    # pinned posts sit on top (max 3), so look a little past them
LATER_RANGE = "1-30"       # newest N posts per profile per run
POST_PATHS = ("p", "reel", "reels", "tv")
FILENAME = "{date:%Y%m%d-%H%M%S}_{post_shortcode}_{num}.{extension}"
BASELINE_KEY = "ig_baselined"

STATE = {"running": False, "last_run": None, "last_result": "never run", "log": []}
_LOCK = threading.Lock()


def archive_path():
    return os.path.join(data_root(), "ig-archive.sqlite")


def target_url(link):
    """Profile link -> its posts feed. A single post/reel link is kept as-is."""
    m = re.match(r"^https?://[^/]+/([^/?#]+)(?:/([^/?#]+))?", link.strip())
    if not m:
        return None
    first = m.group(1)
    if first.lower() in POST_PATHS:
        return link.strip()
    return f"https://www.instagram.com/{first}/posts/"


def _log(msg):
    STATE["log"].append(f"{datetime.now():%H:%M:%S} {msg}")
    del STATE["log"][:-40]


def _baselined():
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (BASELINE_KEY,)).fetchone()
    conn.close()
    try:
        return set(json.loads(row["value"])) if row and row["value"] else set()
    except ValueError:
        return set()


def _mark_baselined(album_id):
    ids = _baselined() | {album_id}
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                 (BASELINE_KEY, json.dumps(sorted(ids))))
    conn.commit()
    conn.close()


def _is_pinned(meta):
    """gallery-dl's field name has varied, so treat any truthy *pinned* key as pinned."""
    return any("pinned" in str(k).lower() and v for k, v in meta.items())


def _fetch(url, job_dir, rng):
    cmd = [
        shutil.which("gallery-dl"), "-d", job_dir, "--no-mtime",
        "--filename", FILENAME, "--write-metadata",
        "--download-archive", archive_path(),
        "--range", rng,
        "--sleep", "1-3", "--sleep-request", "3-6",
    ]
    if os.path.exists(cookies_path()):
        cmd += ["--cookies", cookies_path()]
    cmd.append(url)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return (p.stdout + "\n" + p.stderr), p.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc), 1


def _pick_files(job_dir, first_run):
    """Returns (files_to_keep, saw_any_post). Pinned posts are always dropped."""
    posts = {}                      # shortcode -> {"date": str, "files": [...]}
    saw_post = False
    for dp, _d, names in os.walk(job_dir):
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext not in PHOTO_EXTS | VIDEO_EXTS:
                continue
            path = os.path.join(dp, n)
            saw_post = True
            try:
                with open(path + ".json", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                meta = {}
            if _is_pinned(meta):
                continue
            # filename is <date>_<shortcode>_<num>.<ext>
            parts = n.split("_", 2)
            date = parts[0] if len(parts) == 3 else ""
            code = parts[1] if len(parts) == 3 else n
            posts.setdefault(code, {"date": date, "files": []})["files"].append(path)
    if not posts:
        return [], saw_post
    if first_run:
        newest = max(posts.values(), key=lambda p: p["date"])
        return newest["files"], saw_post
    return [f for p in posts.values() for f in p["files"]], saw_post


def run_once(helpers):
    """helpers: dict with
         dest_dir(conn, album_id, album_name) -> folder for that channel
         after_new(media_ids, album_id)       -> optional (face/look indexing)"""
    if not _LOCK.acquire(blocking=False):
        return
    STATE["running"] = True
    added_total = 0
    try:
        if not tool_available():
            STATE["last_result"] = "gallery-dl isn't installed (pip install gallery-dl)"
            return
        if not os.path.exists(cookies_path()):
            STATE["last_result"] = "no cookies.txt uploaded yet (Import page)"
            return

        conn = get_db()
        albums = [dict(r) for r in conn.execute(
            "SELECT id, name, link FROM albums WHERE link IS NOT NULL AND link != ''")]
        conn.close()
        albums = [a for a in albums if is_instagram(a["link"]) and target_url(a["link"])]
        done_before = _baselined()

        root = library_root()
        for i, a in enumerate(albums):
            first_run = a["id"] not in done_before
            job_dir = os.path.join(staging_root(), "auto", str(a["id"]))
            shutil.rmtree(job_dir, ignore_errors=True)
            os.makedirs(job_dir, exist_ok=True)

            out, code = _fetch(target_url(a["link"]), job_dir,
                               FIRST_RUN_RANGE if first_run else LATER_RANGE)
            low = out.lower()
            if "429" in low or "login required" in low or "checkpoint" in low:
                _log(f"{a['name']}: Instagram is rate-limiting or the session expired, stopping this run")
                shutil.rmtree(job_dir, ignore_errors=True)
                break

            keep, saw_post = _pick_files(job_dir, first_run)

            if keep:
                conn = get_db()
                dest_dir = helpers["dest_dir"](conn, a["id"], a["name"])
                conn.close()
                os.makedirs(dest_dir, exist_ok=True)
                rels = []
                for f in sorted(keep):
                    dest = os.path.join(dest_dir, os.path.basename(f))
                    base, ext = os.path.splitext(dest)
                    k = 1
                    while os.path.exists(dest):
                        dest = f"{base}_{k}{ext}"
                        k += 1
                    shutil.move(f, dest)
                    rels.append(os.path.relpath(dest, root))

                scan_library(source="instagram")
                conn = get_db()
                ph = ",".join("?" for _ in rels)
                ids = [r["id"] for r in conn.execute(
                    f"SELECT id FROM media WHERE relpath IN ({ph})", rels)]
                for mid in ids:
                    conn.execute(
                        "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
                        (mid, a["id"]))
                if ids:
                    conn.execute(
                        "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ?",
                        (ids[0], a["id"]))
                conn.commit()
                conn.close()
                if ids and helpers.get("after_new"):
                    helpers["after_new"](ids, a["id"])
                added_total += len(ids)
                _log(f"{a['name']}: {len(ids)} new" + (" (first run: latest post only)" if first_run else ""))
            elif code == 0 or saw_post:
                _log(f"{a['name']}: nothing new")
            else:
                _log(f"{a['name']}: failed ({out.strip()[-120:]})")

            # Baseline only once the profile was actually read, so a failed
            # first attempt gets treated as a first run again next time.
            if first_run and (saw_post or code == 0):
                _mark_baselined(a["id"])

            shutil.rmtree(job_dir, ignore_errors=True)
            if i < len(albums) - 1:
                time.sleep(random.uniform(20, 60))   # be gentle between profiles

        STATE["last_result"] = f"{added_total} new item(s) from {len(albums)} linked channel(s)"
    except Exception as exc:  # noqa: BLE001
        STATE["last_result"] = f"error: {exc}"
    finally:
        STATE["running"] = False
        STATE["last_run"] = datetime.now().isoformat(timespec="seconds")
        _LOCK.release()


def start_loop(helpers, default_minutes=180):
    """Runs run_once every POSTMARK_IG_SYNC_MIN minutes (0 turns it off)."""
    try:
        mins = int(os.environ.get("POSTMARK_IG_SYNC_MIN", default_minutes))
    except ValueError:
        mins = default_minutes
    if mins <= 0:
        return

    def loop():
        time.sleep(90)                       # let the app finish starting
        while True:
            run_once(helpers)
            time.sleep(mins * 60)

    threading.Thread(target=loop, daemon=True).start()
