"""Download media from Instagram with gallery-dl, then work out which
channel each source (username or subreddit) belongs to."""
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher

from scanner import data_root, PHOTO_EXTS, VIDEO_EXTS

AUTO_THRESHOLD = 0.85   # at or above: file into the channel automatically
ASK_THRESHOLD = 0.70    # at or above (but below auto): ask the user first


def staging_root():
    return os.path.join(data_root(), "imports")


def cookies_path():
    return os.path.join(data_root(), "cookies.txt")


def tool_available():
    return shutil.which("gallery-dl") is not None


def normalize(name):
    """'r/Jane_Doe.official' and '@jane doe official' -> 'janedoeofficial'."""
    name = (name or "").lower().strip()
    name = re.sub(r"^(r/|u/|@)", "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def score(source, channel):
    a, b = normalize(source), normalize(channel)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and short in long_:
        ratio = max(ratio, 0.9)
    return ratio


def rank_channels(source, albums, aliases):
    """Returns (top_matches, decision). `aliases` maps normalized source ->
    album_id. decision is 'auto', 'ask' or 'new'."""
    by_id = {a["id"]: a for a in albums}
    alias_id = aliases.get(normalize(source))
    if alias_id in by_id:
        return [{"id": alias_id, "name": by_id[alias_id]["name"], "score": 1.0}], "auto"
    scored = sorted(
        ({"id": a["id"], "name": a["name"], "score": score(source, a["name"])} for a in albums),
        key=lambda m: m["score"], reverse=True,
    )
    top = [m for m in scored if m["score"] >= ASK_THRESHOLD][:3]
    if not top:
        return [], "new"
    runner_up = top[1]["score"] if len(top) > 1 else 0
    if top[0]["score"] >= AUTO_THRESHOLD and top[0]["score"] - runner_up >= 0.05:
        return top[:1], "auto"
    return top, "ask"


def download(url, job_dir, log):
    """Runs gallery-dl into job_dir. Layout is <platform>/<name>/files.
    Returns None on success or an error string."""
    exe = shutil.which("gallery-dl")
    if not exe:
        return "gallery-dl isn't installed. Run: pip install gallery-dl"
    cmd = [exe, "-d", job_dir, "--no-mtime"]
    if os.path.exists(cookies_path()):
        cmd += ["--cookies", cookies_path()]
    cmd.append(url)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.append(line.strip())
            del log[:-8]
        proc.wait()
    except OSError as exc:
        return str(exc)
    return None


def collect_groups(job_dir):
    """Finds <platform>/<source>/ folders that contain photos or videos."""
    groups = []
    if not os.path.isdir(job_dir):
        return groups
    for platform in sorted(os.listdir(job_dir)):
        pdir = os.path.join(job_dir, platform)
        if not os.path.isdir(pdir):
            continue
        for source in sorted(os.listdir(pdir)):
            sdir = os.path.join(pdir, source)
            files = []
            for dirpath, _d, names in os.walk(sdir):
                for n in names:
                    if os.path.splitext(n)[1].lower() in PHOTO_EXTS | VIDEO_EXTS:
                        files.append(os.path.join(dirpath, n))
            if files:
                groups.append({"platform": platform, "source": source, "files": files})
    return groups


def is_instagram(url):
    m = re.match(r"^https?://([^/]+)/", url + "/")
    host = (m.group(1).lower() if m else "")
    return host == "instagram.com" or host.endswith(".instagram.com")
