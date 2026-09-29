import os
import re
import json
import time
import random
import shutil
import hashlib
import zipfile
import threading
from datetime import datetime, timezone, timedelta
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, jsonify, abort, Response
)
from werkzeug.utils import secure_filename

from db import get_db, init_db, now_iso
import faces
import look
from scanner import (
    scan_library, ensure_thumbnail, ensure_preview, thumbnail_path, preview_path,
    library_root, data_root, PHOTO_EXTS, VIDEO_EXTS
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB per upload request

PAGE_SIZE = 40
SPOTLIGHT_SIZE = 6  # how many images rotate through a collection's spotlight/cover

# Feed leaderboard scoring (ported from the old Discord bot's !totalvotes points
# formula). A session only counts once it actually ran a little while — filters
# out instantly opening then closing a feed.
MIN_FEED_DURATION_FOR_POINTS = 30
POINTS_PER_SESSION = 1
POINTS_PER_MINUTE = 1
POINTS_PER_ITEM = 0.1


def today_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _spotlight_for_group(conn, albums, seed_key):
    """Deterministically picks one channel from a list of channels to feature,
    seeded from today's date plus the group's own key — stable for the whole
    day, different again tomorrow, and different across categories on the
    same day, with no scheduled job or saved state needed."""
    if not albums:
        return None
    seed = int(hashlib.sha256(f"{today_key()}:{seed_key}".encode()).hexdigest(), 16)
    return albums[seed % len(albums)]


def _home_spotlights(conn):
    """One spotlight card per category, each featuring one channel from
    inside it — 'channel of the day' per category, ported from the old
    Discord bot's rotating category spotlight cards — plus one more for
    any channels that aren't in a category at all, if there are any."""
    spotlights = []
    for c in _all_categories(conn):
        albums = [a for a in _all_albums(conn, category_id=c["id"]) if a["item_count"] > 0]
        channel = _spotlight_for_group(conn, albums, f"cat:{c['id']}")
        if not channel:
            continue
        spotlights.append({
            "group_label": c["name"],
            "group_icon": "\U0001F4C1 ",  # 📁
            "channel": channel,
            "urls": [url_for("preview", media_id=mid) for mid in _album_spotlight_ids(conn, channel["id"], limit=SPOTLIGHT_SIZE)],
        })

    uncategorized = [a for a in _all_albums(conn, uncategorized_only=True) if a["item_count"] > 0]
    channel = _spotlight_for_group(conn, uncategorized, "uncategorized")
    if channel:
        spotlights.append({
            "group_label": "Uncategorized",
            "group_icon": "",
            "channel": channel,
            "urls": [url_for("preview", media_id=mid) for mid in _album_spotlight_ids(conn, channel["id"], limit=SPOTLIGHT_SIZE)],
        })
    return spotlights


def _default_instagram_link(album_name):
    """Your channel names are already Instagram usernames, so guess the
    profile link from the name instead of making you paste every one by
    hand. Only used when the channel has no manually-saved link yet."""
    handle = (album_name or "").strip().replace(" ", "")
    return f"https://instagram.com/{handle}" if handle else None


def _all_albums(conn, category_id=None, uncategorized_only=False):
    clauses = []
    params = []
    if category_id is not None:
        clauses.append("a.category_id = ?")
        params.append(category_id)
    elif uncategorized_only:
        clauses.append("a.category_id IS NULL")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(
        f"""
        SELECT a.id, a.name, a.cover_media_id, a.link, a.category_id, c.name AS category_name,
               COUNT(ma.media_id) AS item_count
        FROM albums a
        LEFT JOIN media_albums ma ON ma.album_id = a.id
        LEFT JOIN categories c ON c.id = a.category_id
        {where_sql}
        GROUP BY a.id
        ORDER BY a.name COLLATE NOCASE
        """,
        params,
    ).fetchall()


def _all_categories(conn):
    """Categories with a rollup of how many channels (collections) and how
    many media items live under them."""
    return conn.execute(
        """
        SELECT c.id, c.name,
               COUNT(DISTINCT a.id) AS channel_count,
               COUNT(ma.media_id) AS item_count
        FROM categories c
        LEFT JOIN albums a ON a.category_id = c.id
        LEFT JOIN media_albums ma ON ma.album_id = a.id
        GROUP BY c.id
        ORDER BY c.name COLLATE NOCASE
        """
    ).fetchall()


def _category_spotlight_ids(conn, category_id, limit=SPOTLIGHT_SIZE):
    """A handful of media ids from anywhere inside this category's channels,
    used the same way an individual collection's spotlight/cover works."""
    rows = conn.execute(
        """
        SELECT ma.media_id FROM media_albums ma
        JOIN albums a ON a.id = ma.album_id
        WHERE a.category_id = ?
        ORDER BY RANDOM() LIMIT ?
        """,
        (category_id, limit),
    ).fetchall()
    return [r["media_id"] for r in rows]


def _get_or_create_category(conn, category_id, new_name):
    """Shared by every 'assign to category' form: use an existing category id,
    or create one from a typed name, or return None for 'no category'."""
    new_name = (new_name or "").strip()
    if new_name:
        conn.execute(
            "INSERT OR IGNORE INTO categories (name, created_at) VALUES (?, ?)",
            (new_name, now_iso()),
        )
        row = conn.execute("SELECT id FROM categories WHERE name = ?", (new_name,)).fetchone()
        return row["id"] if row else None
    return int(category_id) if category_id else None


def _album_spotlight_ids(conn, album_id, limit=SPOTLIGHT_SIZE):
    """A handful of media ids from this collection to rotate through as its
    cover / spotlight preview, instead of one fixed (often missing) image."""
    rows = conn.execute(
        """
        SELECT media_id FROM media_albums
        WHERE album_id = ?
        ORDER BY RANDOM() LIMIT ?
        """,
        (album_id, limit),
    ).fetchall()
    return [r["media_id"] for r in rows]


def _library_stats(conn):
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN media_type = 'photo' THEN 1 ELSE 0 END) AS photos,
            SUM(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END) AS videos,
            SUM(CASE WHEN favorite = 1 THEN 1 ELSE 0 END) AS favorites,
            COALESCE(SUM(size_bytes), 0) AS total_bytes,
            COALESCE(SUM(CASE WHEN media_type = 'photo' THEN size_bytes END), 0) AS photo_bytes,
            COALESCE(SUM(CASE WHEN media_type = 'video' THEN size_bytes END), 0) AS video_bytes
        FROM media
        """
    ).fetchone()
    collections = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
    categories = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    return {
        "total": row["total"] or 0,
        "photos": row["photos"] or 0,
        "videos": row["videos"] or 0,
        "favorites": row["favorites"] or 0,
        "total_bytes": row["total_bytes"] or 0,
        "photo_bytes": row["photo_bytes"] or 0,
        "video_bytes": row["video_bytes"] or 0,
        "collections": collections,
        "categories": categories,
    }


def _top_channels(conn, limit=5):
    """Channels with the most items, for the dashboard's Top Channels card.
    Each row also carries the channel's total size on disk."""
    return conn.execute(
        """
        SELECT a.id, a.name, COUNT(ma.media_id) AS item_count,
               COALESCE(SUM(m.size_bytes), 0) AS total_bytes
        FROM albums a
        JOIN media_albums ma ON ma.album_id = a.id
        JOIN media m ON m.id = ma.media_id
        GROUP BY a.id
        ORDER BY item_count DESC, a.name COLLATE NOCASE
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _library_growth(conn, max_years=6):
    """Running total of the library at the end of each year (and each month
    inside it), by the date a photo/video was taken (falling back to when it
    was added). Returns a list of {year, added, total, months}, oldest first,
    limited to the latest years. Each month is {month, name, added, total}."""
    this_year = datetime.now(timezone.utc).year
    rows = conn.execute(
        """
        SELECT substr(COALESCE(NULLIF(taken_at, ''), added_at), 1, 4) AS year,
               substr(COALESCE(NULLIF(taken_at, ''), added_at), 6, 2) AS month,
               COUNT(*) AS n,
               COALESCE(SUM(size_bytes), 0) AS b
        FROM media
        GROUP BY year, month
        ORDER BY year, month
        """
    ).fetchall()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_year = {}
    for r in rows:
        y = r["year"]
        if not (y and y.isdigit() and 1990 <= int(y) <= this_year + 1):
            continue
        entry = by_year.setdefault(int(y), {"months": {}, "bytes": {}, "unknown": 0, "unknown_bytes": 0})
        m = r["month"]
        if m and m.isdigit() and 1 <= int(m) <= 12:
            entry["months"][int(m)] = entry["months"].get(int(m), 0) + r["n"]
            entry["bytes"][int(m)] = entry["bytes"].get(int(m), 0) + r["b"]
        else:
            entry["unknown"] += r["n"]
            entry["unknown_bytes"] += r["b"]
    running, out = 0, []
    for year in sorted(by_year):
        entry = by_year[year]
        year_start = running
        months = []
        for m in sorted(entry["months"]):
            running += entry["months"][m]
            months.append({"month": m, "name": names[m - 1],
                           "added": entry["months"][m], "total": running,
                           "added_bytes": entry["bytes"].get(m, 0)})
        if entry["unknown"]:
            running += entry["unknown"]
            months.append({"month": 13, "name": "Unknown",
                           "added": entry["unknown"], "total": running,
                           "added_bytes": entry["unknown_bytes"]})
        out.append({"year": year, "added": running - year_start,
                    "total": running, "months": months,
                    "added_bytes": sum(x["added_bytes"] for x in months)})
    return out[-max_years:]


def _format_span(months):
    """'~8 months', '~3.4 years' — a rough time label for the forecast."""
    if months < 1:
        return "under a month"
    if months < 24:
        n = int(round(months))
        return f"~{n} month{'s' if n != 1 else ''}"
    years = months / 12
    if years >= 100:
        return "100+ years"
    return f"~{years:.1f} years"


def _storage_forecast(conn, window_months=12):
    """Growth rate and time-until-full, from the library's own history.
    Uses the same dates as the Library Growth card (taken date, falling back
    to added date) and the size of each file. The rate is the average over
    the last `window_months` complete months (fewer if the library is
    younger; at least 2 months of history are needed). Free space is read
    from the drive the library lives on. Returns None when there is not
    enough history to say anything."""
    now = datetime.now(timezone.utc)
    cur_idx = now.year * 12 + now.month - 1
    rows = conn.execute(
        """
        SELECT substr(COALESCE(NULLIF(taken_at, ''), added_at), 1, 4) AS year,
               substr(COALESCE(NULLIF(taken_at, ''), added_at), 6, 2) AS month,
               COALESCE(SUM(size_bytes), 0) AS b
        FROM media
        GROUP BY year, month
        """
    ).fetchall()
    by_idx = {}
    for r in rows:
        y, m = r["year"], r["month"]
        if not (y and y.isdigit() and m and m.isdigit()):
            continue
        y, m = int(y), int(m)
        if not (1990 <= y <= now.year + 1 and 1 <= m <= 12):
            continue
        idx = y * 12 + m - 1
        if idx >= cur_idx:  # skip the current (partial) month and the future
            continue
        by_idx[idx] = by_idx.get(idx, 0) + r["b"]
    window = {i: b for i, b in by_idx.items() if i >= cur_idx - window_months}
    if not window:
        return None
    first = min(window)
    span = min(cur_idx - first, window_months)
    if span < 2:
        return None
    rate = sum(window.values()) / span
    if rate <= 0:
        return None
    try:
        free = shutil.disk_usage(library_root()).free
    except OSError:
        free = None
    out = {"rate_bytes": rate, "span_months": span, "free_bytes": free,
           "months_left": None, "time_left": None}
    if free is not None:
        out["months_left"] = free / rate
        out["time_left"] = _format_span(free / rate)
    return out


def _human_size(num_bytes):
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


app.jinja_env.filters["human_size"] = _human_size


def _format_duration(seconds):
    """Compact watch-time label: '45s', '12m 10s', '3h 05m'."""
    total = int(round(seconds or 0))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


app.jinja_env.filters["duration"] = _format_duration


def _album_folder_name(name):
    """Filesystem-safe folder name for a collection, so files tied to it
    actually live in their own folder on disk instead of one flat pile."""
    return secure_filename((name or "").strip()) or "collection"


def _fs_name(name):
    """Folder name for a channel that keeps the name as typed (only characters
    the filesystem cannot hold are replaced), so a rescan maps the folder back
    to the same channel."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", (name or "").strip()).strip(" .")
    return cleaned or "collection"


def _channel_home_rel(conn, album_id, album_name):
    """Folder (relative to the library) that holds most of a channel's files and
    is named after the channel, or None if there is no such folder yet."""
    if not (album_id and album_name):
        return None
    names = {album_name.strip().lower(), _album_folder_name(album_name).lower(), _fs_name(album_name).lower()}
    rows = conn.execute(
        "SELECT m.relpath FROM media m JOIN media_albums ma ON ma.media_id = m.id WHERE ma.album_id = ?",
        (album_id,),
    ).fetchall()
    counts = {}
    for r in rows:
        d = os.path.dirname(r["relpath"])
        if d and os.path.basename(d).lower() in names:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _album_dest_dir(conn, root, album_id, album_name, category_name):
    """Folder where files for this channel belong.
    1. The folder that already holds most of the channel's files AND is named
       after the channel (so a photo that was only 'added' from another
       channel's folder can never make us pick that other folder).
    2. Otherwise Category/Channel (or just Channel) built from the names."""
    home = _channel_home_rel(conn, album_id, album_name)
    if home:
        return os.path.join(root, home)
    if not album_name:
        return root
    parts = [p for p in (category_name and _fs_name(category_name), _fs_name(album_name)) if p]
    canon = os.path.join(root, *parts)
    if category_name and not os.path.isdir(canon):
        # A channel folder that already exists uncategorized: reuse it instead of making a twin.
        flat = os.path.join(root, _fs_name(album_name))
        if os.path.isdir(flat):
            return flat
    return canon


def _album_category_name(conn, album_id):
    if not album_id:
        return None
    r = conn.execute(
        "SELECT c.name FROM albums a LEFT JOIN categories c ON c.id = a.category_id WHERE a.id = ?",
        (album_id,),
    ).fetchone()
    return r["name"] if r else None


def _unique_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    dest = os.path.join(dest_dir, filename)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
        i += 1
    return dest


def _filtered_media(args):
    """Builds the WHERE/JOIN clauses shared by the browse grid, a collection's
    grid, and the media viewer's prev/next navigation, so all three agree on
    the same ordering and filters."""
    q = args.get("q", "").strip()
    album_id = args.get("album")
    favorites_only = args.get("fav") == "1"
    media_type = args.get("type") if args.get("type") in ("photo", "video") else None

    clauses = []
    params = []
    if q:
        clauses.append("m.filename LIKE ?")
        params.append(f"%{q}%")
    if favorites_only:
        clauses.append("m.favorite = 1")
    if media_type:
        clauses.append("m.media_type = ?")
        params.append(media_type)
    # Optional date range (YYYY-MM-DD), on the same date the library uses
    # everywhere else: taken date, falling back to added date.
    date_expr = "substr(COALESCE(NULLIF(m.taken_at, ''), m.added_at), 1, 10)"
    date_from = (args.get("from") or "").strip()
    date_to = (args.get("to") or "").strip()
    if len(date_from) == 10:
        clauses.append(f"{date_expr} >= ?")
        params.append(date_from)
    if len(date_to) == 10:
        clauses.append(f"{date_expr} <= ?")
        params.append(date_to)

    joins = ""
    join_params = []
    if album_id:
        joins = " JOIN media_albums ma ON ma.media_id = m.id AND ma.album_id = ?"
        join_params = [album_id]

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    ctx = {"q": q, "album": album_id, "fav": "1" if favorites_only else None, "type": media_type}
    return joins, join_params, where_sql, params, ctx


@app.context_processor
def inject_nav():
    """Shared sidebar data: each category alongside the channels inside it
    (for the collapsible channel tree), the flat uncategorized-channel list,
    and library-wide totals (for the 'All Media' / 'Favorites' quick links)."""
    conn = get_db()
    nav_categories = []
    for c in _all_categories(conn):
        nav_categories.append({
            "id": c["id"],
            "name": c["name"],
            "channel_count": c["channel_count"],
            "item_count": c["item_count"],
            "channels": _all_albums(conn, category_id=c["id"]),
        })
    nav_uncategorized = _all_albums(conn, uncategorized_only=True)
    nav_stats = _library_stats(conn)
    conn.close()
    return {
        "nav_categories": nav_categories,
        "nav_uncategorized": nav_uncategorized,
        "nav_stats": nav_stats,
    }


@app.route("/")
def home():
    """Catalog page: library-wide stats, then one spotlight card per
    category (each showing a channel-of-the-day from inside it), plus one
    for anything uncategorized. No flat channel list anymore — every
    channel is still reachable from its category's own page or the
    sidebar."""
    conn = get_db()
    stats = _library_stats(conn)
    spotlights = _home_spotlights(conn)
    top_channels = _top_channels(conn)
    growth = _library_growth(conn)
    forecast = _storage_forecast(conn)
    conn.close()
    return render_template(
        "home.html",
        stats=stats,
        spotlights=spotlights,
        top_channels=top_channels,
        growth=growth,
        forecast=forecast,
    )


@app.route("/categories", methods=["POST"])
def create_category():
    name = request.form.get("name", "").strip()
    conn = get_db()
    if name:
        _get_or_create_category(conn, None, name)
        conn.commit()
    conn.close()
    return redirect(url_for("home"))


@app.route("/category/<int:category_id>")
def category_detail(category_id):
    """The 'Category' page — every channel (collection) sorted into this
    category, each shown with its own cover, name, and item count."""
    conn = get_db()
    category = conn.execute("SELECT * FROM categories WHERE id = ?", (category_id,)).fetchone()
    if not category:
        conn.close()
        abort(404)

    albums = _all_albums(conn, category_id=category_id)
    spotlights = {
        a["id"]: [url_for("preview", media_id=mid) for mid in _album_spotlight_ids(conn, a["id"], limit=4)]
        for a in albums
    }
    total_items = sum(a["item_count"] for a in albums)
    hero_urls = [url_for("preview", media_id=mid) for mid in _category_spotlight_ids(conn, category_id, limit=SPOTLIGHT_SIZE)]
    conn.close()

    return render_template(
        "category.html",
        category=category,
        albums=albums,
        spotlights=spotlights,
        total_items=total_items,
        hero_urls=hero_urls,
    )


@app.route("/category/<int:category_id>/delete", methods=["POST"])
def delete_category(category_id):
    """Removes the category itself; the channels inside it aren't deleted —
    they just fall back to uncategorized on the catalog page."""
    conn = get_db()
    conn.execute("DELETE FROM categories WHERE id = ?", (category_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Channel rename / delete, and a repair that puts files in their channel folders
# ---------------------------------------------------------------------------
def _repoint_prefix(conn, old_rel, new_rel):
    """After a folder is renamed, fix the stored path of every file inside it."""
    prefix = old_rel + os.sep
    rows = conn.execute(
        "SELECT id, relpath FROM media WHERE substr(relpath, 1, ?) = ?", (len(prefix), prefix)
    ).fetchall()
    for r in rows:
        conn.execute("UPDATE media SET relpath = ? WHERE id = ?",
                     (new_rel + os.sep + r["relpath"][len(prefix):], r["id"]))
    return len(rows)


@app.route("/collection/<int:album_id>/rename", methods=["POST"])
def rename_collection(album_id):
    """Renames the channel AND its folder on disk, so a rescan finds the same channel."""
    new_name = re.sub(r"\s+", " ", request.form.get("name", "")).strip()[:80]
    back = lambda msg: redirect(url_for("collection_detail", album_id=album_id, msg=msg))
    conn = get_db()
    al = conn.execute("SELECT id, name FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not al:
        conn.close()
        abort(404)
    if not new_name or new_name == al["name"]:
        conn.close()
        return back("Type a new name first." if not new_name else "That is already its name.")
    if conn.execute("SELECT 1 FROM albums WHERE lower(name) = lower(?) AND id != ?", (new_name, album_id)).fetchone():
        conn.close()
        return back(f"A channel called \u201c{new_name}\u201d already exists.")

    root = library_root()
    home = _channel_home_rel(conn, album_id, al["name"])
    renamed_dir = None
    try:
        if home:
            parent = os.path.dirname(home)
            new_rel = os.path.join(parent, _fs_name(new_name)) if parent else _fs_name(new_name)
            old_abs, new_abs = os.path.join(root, home), os.path.join(root, new_rel)
            if new_rel != home:
                if os.path.exists(new_abs):
                    conn.close()
                    return back(f"A folder named \u201c{_fs_name(new_name)}\u201d already exists there. Pick another name.")
                os.rename(old_abs, new_abs)
                renamed_dir = (old_abs, new_abs)
                _repoint_prefix(conn, home, new_rel)
        conn.execute("UPDATE albums SET name = ? WHERE id = ?", (new_name, album_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        if renamed_dir:
            try:
                os.rename(renamed_dir[1], renamed_dir[0])
            except OSError:
                pass
        conn.close()
        return back(f"Could not rename: {e}. Nothing was changed.")
    conn.close()
    return back(f"Renamed to \u201c{new_name}\u201d" + (" (folder renamed too)." if renamed_dir else "."))


@app.route("/collection/<int:album_id>/delete", methods=["POST"])
def delete_collection(album_id):
    """mode=keep : remove the channel, keep every photo/video.
    mode=files: remove the channel and the files that are only in this channel."""
    mode = request.form.get("mode", "keep")
    conn = get_db()
    al = conn.execute("SELECT id, name FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not al:
        conn.close()
        abort(404)
    root = library_root()
    home = _channel_home_rel(conn, album_id, al["name"])
    member_rows = conn.execute(
        "SELECT m.id, m.relpath, m.filename, "
        "(SELECT COUNT(*) FROM media_albums x WHERE x.media_id = m.id) AS n "
        "FROM media m JOIN media_albums ma ON ma.media_id = m.id WHERE ma.album_id = ?", (album_id,)
    ).fetchall()
    removed = kept = 0
    leftovers = []

    if mode == "files":
        for r in member_rows:
            if r["n"] > 1:
                kept += 1                       # also in another channel: only unlink
                continue
            if _trash_media(conn, r["id"]):     # to the Trash, not gone for good
                removed += 1
    else:
        # Files inside this channel's own folder must leave it, or the next
        # rescan would recreate the channel from the folder name.
        home_prefix = (home + os.sep) if home else None
        for r in member_rows:
            if not home_prefix or not r["relpath"].startswith(home_prefix):
                continue
            other = conn.execute(
                "SELECT a.id, a.name FROM media_albums ma JOIN albums a ON a.id = ma.album_id "
                "WHERE ma.media_id = ? AND a.id != ? ORDER BY a.id LIMIT 1", (r["id"], album_id)
            ).fetchone()
            dest = (_album_dest_dir(conn, root, other["id"], other["name"], _album_category_name(conn, other["id"]))
                    if other else root)
            old_full = os.path.join(root, r["relpath"])
            try:
                os.makedirs(dest, exist_ok=True)
                new_full = _unique_path(dest, r["filename"])
                shutil.move(old_full, new_full)
                conn.execute("UPDATE media SET relpath = ? WHERE id = ?", (os.path.relpath(new_full, root), r["id"]))
            except OSError:
                leftovers.append(r["filename"])
        kept = len(member_rows)

    if leftovers:
        conn.commit()
        conn.close()
        return redirect(url_for("collection_detail", album_id=album_id,
                                msg=f"{len(leftovers)} file(s) could not be moved out of the folder, so the channel was kept. Nothing else was deleted."))
    conn.execute("DELETE FROM feed_stats WHERE album_id = ?", (album_id,))
    conn.execute("DELETE FROM media_albums WHERE album_id = ?", (album_id,))
    conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))
    conn.commit()
    conn.close()
    if home:
        try:
            os.rmdir(os.path.join(root, home))     # only succeeds if empty
        except OSError:
            pass
    msg = (f"Channel deleted. {removed} file(s) deleted." + (f" {kept} kept (they are in other channels)." if kept else "")
           if mode == "files" else f"Channel deleted. All {kept} photos/videos were kept.")
    return redirect(url_for("home", msg=msg))


def _sync_plan(conn):
    """[(media_id, destination_dir)] for files that sit in a different folder
    than their channel's. Files linked to several channels are left alone."""
    root = library_root()
    plan = []
    for al in conn.execute(
        "SELECT a.id, a.name, c.name AS cat FROM albums a LEFT JOIN categories c ON c.id = a.category_id"
    ).fetchall():
        dest = _album_dest_dir(conn, root, al["id"], al["name"], al["cat"])
        dest_abs = os.path.abspath(dest)
        rows = conn.execute(
            "SELECT m.id, m.relpath FROM media m JOIN media_albums ma ON ma.media_id = m.id "
            "WHERE ma.album_id = ? AND (SELECT COUNT(*) FROM media_albums x WHERE x.media_id = m.id) = 1",
            (al["id"],),
        ).fetchall()
        for r in rows:
            if os.path.abspath(os.path.dirname(os.path.join(root, r["relpath"]))) != dest_abs:
                plan.append((r["id"], dest, al["name"], r["relpath"]))
    return plan


@app.route("/maintenance/files-check")
def files_check():
    conn = get_db()
    plan = _sync_plan(conn)
    conn.close()
    root = library_root()
    return jsonify({"count": len(plan), "library": root,
                    "examples": [{"from": p[3], "to": os.path.relpath(p[1], root), "channel": p[2]} for p in plan[:4]]})


@app.route("/maintenance/files-sync", methods=["POST"])
def files_sync():
    limit = min(max(request.args.get("limit", type=int) or 200, 1), 1000)
    conn = get_db()
    plan = _sync_plan(conn)
    root = library_root()
    moved = failed = 0
    for mid, dest, _name, _rel in plan[:limit]:
        row = conn.execute("SELECT relpath, filename FROM media WHERE id = ?", (mid,)).fetchone()
        old_full = os.path.join(root, row["relpath"])
        if not os.path.exists(old_full):
            failed += 1
            continue
        try:
            os.makedirs(dest, exist_ok=True)
            new_full = _unique_path(dest, row["filename"])
            shutil.move(old_full, new_full)
        except OSError:
            failed += 1
            continue
        conn.execute("UPDATE media SET relpath = ? WHERE id = ?", (os.path.relpath(new_full, root), mid))
        moved += 1
    conn.commit()
    conn.close()
    return jsonify({"moved": moved, "failed": failed, "remaining": max(len(plan) - moved, 0)})


@app.route("/collection/<int:album_id>/set_category", methods=["POST"])
def set_collection_category(album_id):
    category_id = request.form.get("category_id") or None
    new_category_name = request.form.get("new_category", "").strip()

    conn = get_db()
    row = conn.execute("SELECT id FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)
    resolved_id = _get_or_create_category(conn, category_id, new_category_name)
    conn.execute("UPDATE albums SET category_id = ? WHERE id = ?", (resolved_id, album_id))
    conn.commit()
    conn.close()
    return redirect(url_for("collection_detail", album_id=album_id))


@app.route("/browse")
def browse():
    conn = get_db()
    page = max(int(request.args.get("page", 1)), 1)
    joins, join_params, where_sql, params, ctx = _filtered_media(request.args)

    total = conn.execute(
        f"SELECT COUNT(*) FROM media m {joins} {where_sql}", join_params + params
    ).fetchone()[0]

    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"""
        SELECT m.* FROM media m {joins} {where_sql}
        ORDER BY COALESCE(m.taken_at, m.added_at) DESC
        LIMIT ? OFFSET ?
        """,
        join_params + params + [PAGE_SIZE, offset],
    ).fetchall()

    conn.close()

    if ctx["type"] == "photo":
        heading = "photos"
    elif ctx["type"] == "video":
        heading = "videos"
    elif ctx["fav"]:
        heading = "favorites"
    else:
        heading = "all-media"

    return render_template(
        "browse.html",
        media=rows,
        page=page,
        total=total,
        page_size=PAGE_SIZE,
        has_more=offset + PAGE_SIZE < total,
        q=ctx["q"],
        favorites_only=bool(ctx["fav"]),
        media_type=ctx["type"],
        heading=heading,
    )


@app.route("/collection/<int:album_id>")
def collection_detail(album_id):
    conn = get_db()
    album = conn.execute("SELECT * FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not album:
        conn.close()
        abort(404)

    page = max(int(request.args.get("page", 1)), 1)
    args = request.args.copy()
    args = args.to_dict()
    args["album"] = album_id
    joins, join_params, where_sql, params, ctx = _filtered_media(args)

    total = conn.execute(
        f"SELECT COUNT(*) FROM media m {joins} {where_sql}", join_params + params
    ).fetchone()[0]

    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"""
        SELECT m.* FROM media m {joins} {where_sql}
        ORDER BY COALESCE(m.taken_at, m.added_at) DESC
        LIMIT ? OFFSET ?
        """,
        join_params + params + [PAGE_SIZE, offset],
    ).fetchall()

    spotlight_urls = [url_for("preview", media_id=mid) for mid in _album_spotlight_ids(conn, album_id, limit=SPOTLIGHT_SIZE)]
    total_bytes = conn.execute(
        "SELECT COALESCE(SUM(m.size_bytes), 0) FROM media m JOIN media_albums ma ON ma.media_id = m.id WHERE ma.album_id = ?",
        (album_id,),
    ).fetchone()[0]

    categories = _all_categories(conn)
    category = None
    if album["category_id"]:
        category = conn.execute("SELECT * FROM categories WHERE id = ?", (album["category_id"],)).fetchone()

    is_default_link = not album["link"]
    instagram_link = album["link"] or _default_instagram_link(album["name"])

    conn.close()

    return render_template(
        "collection.html",
        album=album,
        media=rows,
        page=page,
        total=total,
        page_size=PAGE_SIZE,
        has_more=offset + PAGE_SIZE < total,
        spotlight_urls=spotlight_urls,
        total_bytes=total_bytes,
        categories=categories,
        category=category,
        instagram_link=instagram_link,
        is_default_link=is_default_link,
    )


@app.route("/collection/<int:album_id>/link", methods=["POST"])
def set_collection_link(album_id):
    link = request.form.get("link", "").strip()
    if link and not (link.startswith("http://") or link.startswith("https://")):
        link = "https://" + link
    conn = get_db()
    row = conn.execute("SELECT id FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)
    conn.execute("UPDATE albums SET link = ? WHERE id = ?", (link or None, album_id))
    conn.commit()
    conn.close()
    return redirect(url_for("collection_detail", album_id=album_id))


def _neighbor_ids(conn, media_id, args):
    """Ordered id list for whatever filter context the viewer was opened
    from, so next/prev walks the same set the person was just looking at."""
    joins, join_params, where_sql, params, _ctx = _filtered_media(args)
    rows = conn.execute(
        f"""
        SELECT m.id FROM media m {joins} {where_sql}
        ORDER BY COALESCE(m.taken_at, m.added_at) DESC
        """,
        join_params + params,
    ).fetchall()
    ids = [r["id"] for r in rows]
    if media_id not in ids:
        return ids, None, None, None
    idx = ids.index(media_id)
    prev_id = ids[idx - 1] if len(ids) > 1 else None
    next_id = ids[(idx + 1) % len(ids)] if len(ids) > 1 else None
    return ids, idx, prev_id, next_id


@app.route("/media/<int:media_id>")
def media_detail(media_id):
    conn = get_db()
    item = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if not item:
        conn.close()
        abort(404)

    item_albums = conn.execute(
        """
        SELECT a.id, a.name FROM albums a
        JOIN media_albums ma ON ma.album_id = a.id
        WHERE ma.media_id = ? ORDER BY a.name COLLATE NOCASE
        """,
        (media_id,),
    ).fetchall()
    all_albums = _all_albums(conn)

    ids, idx, prev_id, next_id = _neighbor_ids(conn, media_id, request.args)
    position = f"{idx + 1} / {len(ids)}" if idx is not None else None

    album_id = request.args.get("album")
    if album_id:
        back_url = url_for("collection_detail", album_id=album_id)
    else:
        back_url = url_for(
            "browse", q=request.args.get("q") or None,
            fav=request.args.get("fav") or None,
            type=request.args.get("type") or None,
        )

    def nav_url(target_id):
        if target_id is None:
            return None
        return url_for(
            "media_detail", media_id=target_id,
            album=request.args.get("album") or None,
            q=request.args.get("q") or None,
            fav=request.args.get("fav") or None,
            type=request.args.get("type") or None,
        )

    conn.close()

    return render_template(
        "media_detail.html",
        item=item,
        item_albums=item_albums,
        all_albums=all_albums,
        prev_url=nav_url(prev_id),
        next_url=nav_url(next_id),
        position=position,
        back_url=back_url,
    )


@app.route("/thumb/<int:media_id>")
def thumb(media_id):
    conn = get_db()
    row = conn.execute("SELECT relpath, media_type FROM media WHERE id = ?", (media_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    path = ensure_thumbnail(media_id, row["relpath"], row["media_type"])
    if not path:
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/preview/<int:media_id>")
def preview(media_id):
    """Preview tier disabled for now (space test): serve the original file
    directly instead of generating/reading a resized preview. Feed, category
    hero shots, and collection spotlights all go through this route, so all
    of them now show full-res originals."""
    conn = get_db()
    row = conn.execute("SELECT relpath, media_type FROM media WHERE id = ?", (media_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    full_path = os.path.join(library_root(), row["relpath"])
    if not os.path.exists(full_path):
        abort(404)
    return send_file(full_path, conditional=True)


@app.route("/file/<int:media_id>")
def file(media_id):
    conn = get_db()
    row = conn.execute("SELECT relpath FROM media WHERE id = ?", (media_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    full_path = os.path.join(library_root(), row["relpath"])
    if not os.path.exists(full_path):
        abort(404)
    return send_file(full_path, conditional=True)


@app.route("/download/<int:media_id>")
def download(media_id):
    """Same file as /file, but sent as an attachment so the browser saves it
    to the phone (iOS Safari: Downloads in the Files app) instead of playing it."""
    conn = get_db()
    row = conn.execute("SELECT relpath, filename FROM media WHERE id = ?", (media_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    full_path = os.path.join(library_root(), row["relpath"])
    if not os.path.exists(full_path):
        abort(404)
    return send_file(full_path, as_attachment=True,
                     download_name=row["filename"] or os.path.basename(full_path))


# ---------------------------------------------------------------------------
# Downloads to the phone: a selection, a category, or the whole library, sent
# as one zip that is built while it downloads (nothing is written to disk on
# the server, so a big library does not need spare space).
# ---------------------------------------------------------------------------
class _ZipSink:
    """Write target for zipfile that hands finished bytes to the response."""

    def __init__(self):
        self._parts = []
        self._pos = 0

    def write(self, data):
        data = bytes(data)
        self._parts.append(data)
        self._pos += len(data)
        return len(data)

    def tell(self):
        return self._pos

    def flush(self):
        pass

    def drain(self):
        data = b"".join(self._parts)
        self._parts = []
        return data


def _zip_stream(entries):
    """entries: [(absolute_path, name_inside_zip)]. Photos and videos are
    already compressed, so files are stored as they are (fast on a phone)."""
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path, arcname in entries:
            try:
                info = zipfile.ZipInfo.from_file(path, arcname)
                info.compress_type = zipfile.ZIP_STORED
                src = open(path, "rb")
            except OSError:
                continue
            with src, zf.open(info, "w") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
                    out = sink.drain()
                    if out:
                        yield out
            out = sink.drain()
            if out:
                yield out
    out = sink.drain()
    if out:
        yield out


def _zip_response(entries, label):
    if not entries:
        return Response("Nothing to download.", status=404, mimetype="text/plain")
    stamp = datetime.now().strftime("%Y-%m-%d")
    zipname = f"postmark-{_album_folder_name(label)}-{stamp}.zip"
    resp = Response(_zip_stream(entries), mimetype="application/zip")
    resp.headers.set("Content-Disposition", "attachment", filename=zipname)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


def _unique_name(used, name):
    if name not in used:
        used.add(name)
        return name
    stem, ext = os.path.splitext(name)
    n = 2
    while f"{stem} ({n}){ext}" in used:
        n += 1
    name = f"{stem} ({n}){ext}"
    used.add(name)
    return name


def _entries_for_rows(rows, folder_key=None):
    """rows -> zip entries. With folder_key, files go in that sub-folder."""
    root = library_root()
    used = set()
    seen = set()
    entries = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        path = os.path.join(root, r["relpath"])
        if not os.path.isfile(path):
            continue
        base = r["filename"] or os.path.basename(path)
        if folder_key:
            base = f"{_album_folder_name(r[folder_key])}/{base}"
        entries.append((path, _unique_name(used, base)))
    return entries


@app.route("/download/info", methods=["POST"])
def download_info():
    data = request.get_json(silent=True) or {}
    ids = []
    for i in data.get("ids") or []:
        try:
            ids.append(int(i))
        except (TypeError, ValueError):
            continue
    ids = list(dict.fromkeys(ids))[:100000]
    if not ids:
        return jsonify({"count": 0, "bytes": 0, "size": "0 B"})
    conn = get_db()
    count = total = 0
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        marks = ",".join("?" * len(part))
        row = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM media WHERE id IN ({marks})", part
        ).fetchone()
        count += row[0]
        total += row[1]
    conn.close()
    return jsonify({"count": count, "bytes": total, "size": _human_size(total)})


@app.route("/download/bulk", methods=["POST"])
def download_bulk():
    """The selected items: one item is sent as itself, several as a zip."""
    ids = _parse_ids(request.form.get("ids"), 100000)
    if not ids:
        return Response("Nothing selected.", status=400, mimetype="text/plain")
    conn = get_db()
    rows = []
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        marks = ",".join("?" * len(part))
        rows += conn.execute(
            f"SELECT id, relpath, filename FROM media WHERE id IN ({marks})", part
        ).fetchall()
    conn.close()
    rows.sort(key=lambda r: ids.index(r["id"]) if len(ids) < 2000 else r["id"])
    entries = _entries_for_rows(rows)
    if len(entries) == 1:
        path, name = entries[0]
        return send_file(path, as_attachment=True, download_name=os.path.basename(name))
    return _zip_response(entries, "selection")


@app.route("/download/summary")
def download_summary():
    """Sizes for the Settings > Downloads section (loaded when it is opened)."""
    conn = get_db()
    cats = conn.execute(
        """
        SELECT c.id, c.name, COUNT(x.mid) AS n, COALESCE(SUM(x.sz), 0) AS bytes
        FROM categories c
        LEFT JOIN (
            SELECT DISTINCT a.category_id AS cid, m.id AS mid, m.size_bytes AS sz
            FROM media m
            JOIN media_albums ma ON ma.media_id = m.id
            JOIN albums a ON a.id = ma.album_id
        ) x ON x.cid = c.id
        GROUP BY c.id ORDER BY c.name COLLATE NOCASE
        """
    ).fetchall()
    unc = conn.execute(
        """
        SELECT COUNT(*) AS n, COALESCE(SUM(sz), 0) AS bytes FROM (
            SELECT DISTINCT m.id, m.size_bytes AS sz
            FROM media m
            JOIN media_albums ma ON ma.media_id = m.id
            JOIN albums a ON a.id = ma.album_id
            WHERE a.category_id IS NULL
        )
        """
    ).fetchone()
    allrow = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes FROM media").fetchone()
    conn.close()
    out = [{"id": c["id"], "name": c["name"], "count": c["n"], "size": _human_size(c["bytes"]), "bytes": c["bytes"]}
           for c in cats if c["n"]]
    if unc["n"]:
        out.append({"id": 0, "name": "Uncategorized channels", "count": unc["n"],
                    "size": _human_size(unc["bytes"]), "bytes": unc["bytes"]})
    return jsonify({"categories": out,
                    "all": {"count": allrow["n"], "size": _human_size(allrow["bytes"]), "bytes": allrow["bytes"]}})


@app.route("/download/category/<int:category_id>")
def download_category(category_id):
    """Every item in a category's channels, one folder per channel.
    category_id 0 means channels that have no category."""
    conn = get_db()
    if category_id:
        cat = conn.execute("SELECT name FROM categories WHERE id = ?", (category_id,)).fetchone()
        if not cat:
            conn.close()
            abort(404)
        label, cond, params = cat["name"], "a.category_id = ?", (category_id,)
    else:
        label, cond, params = "uncategorized", "a.category_id IS NULL", ()
    rows = conn.execute(
        f"""
        SELECT m.id, m.relpath, m.filename, a.name AS channel
        FROM media m
        JOIN media_albums ma ON ma.media_id = m.id
        JOIN albums a ON a.id = ma.album_id
        WHERE {cond}
        ORDER BY a.name COLLATE NOCASE, m.filename
        """,
        params,
    ).fetchall()
    conn.close()
    return _zip_response(_entries_for_rows(rows, folder_key="channel"), label)


@app.route("/download/all")
def download_all():
    """The whole library, keeping the folder layout it has on disk."""
    conn = get_db()
    rows = conn.execute("SELECT id, relpath, filename FROM media ORDER BY relpath").fetchall()
    conn.close()
    root = library_root()
    entries = []
    for r in rows:
        path = os.path.join(root, r["relpath"])
        if os.path.isfile(path):
            entries.append((path, r["relpath"].replace(os.sep, "/")))
    return _zip_response(entries, "library")


@app.route("/media/<int:media_id>/favorite", methods=["POST"])
def toggle_favorite(media_id):
    conn = get_db()
    row = conn.execute("SELECT favorite FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)
    new_val = 0 if row["favorite"] else 1
    conn.execute("UPDATE media SET favorite = ? WHERE id = ?", (new_val, media_id))
    conn.commit()
    conn.close()
    return jsonify({"favorite": bool(new_val)})


@app.route("/media/<int:media_id>/albums", methods=["POST"])
def add_to_album(media_id):
    # "Add to channel" now physically moves the file, same as Move.
    return move_media(media_id)
    album_id = request.form.get("album_id")
    new_album_name = request.form.get("new_album", "").strip()

    conn = get_db()
    if new_album_name:
        conn.execute(
            "INSERT OR IGNORE INTO albums (name, created_at) VALUES (?, ?)",
            (new_album_name, now_iso()),
        )
        row = conn.execute("SELECT id FROM albums WHERE name = ?", (new_album_name,)).fetchone()
        album_id = row["id"]

    if album_id:
        conn.execute(
            "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
            (media_id, album_id),
        )
        # Set as cover if the album has none yet
        conn.execute(
            "UPDATE albums SET cover_media_id = ? WHERE id = ? AND cover_media_id IS NULL",
            (media_id, album_id),
        )
    conn.commit()
    conn.close()
    return redirect(url_for("media_detail", media_id=media_id,
                            notice="Added to the channel. This only links it: the file stays where it is. Use Move to relocate the file."))


@app.route("/media/<int:media_id>/move", methods=["POST"])
def move_media(media_id):
    """Physically relocates the file into a collection's folder on disk.
    Unlike 'add to collection', a move replaces prior membership — the file
    now lives in exactly one place, same as moving a photo in a gallery app."""
    album_id = request.form.get("album_id") or None
    new_album_name = request.form.get("new_album", "").strip()

    conn = get_db()
    row = conn.execute("SELECT relpath, filename FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)

    album_name = None
    if new_album_name:
        conn.execute(
            "INSERT OR IGNORE INTO albums (name, created_at) VALUES (?, ?)",
            (new_album_name, now_iso()),
        )
        arow = conn.execute("SELECT id FROM albums WHERE name = ?", (new_album_name,)).fetchone()
        album_id = arow["id"]
        album_name = new_album_name
    elif album_id:
        arow = conn.execute("SELECT name FROM albums WHERE id = ?", (album_id,)).fetchone()
        album_name = arow["name"] if arow else None

    root = library_root()
    old_full = os.path.join(root, row["relpath"])
    cat_name = _album_category_name(conn, album_id)
    dest_dir = _album_dest_dir(conn, root, album_id, album_name, cat_name) if album_name else root
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.abspath(os.path.dirname(old_full)) == os.path.abspath(dest_dir):
        new_full = old_full
    else:
        new_full = _unique_path(dest_dir, row["filename"])

    try:
        if new_full != old_full:
            shutil.move(old_full, new_full)
    except OSError as e:
        conn.close()
        return redirect(url_for("media_detail", media_id=media_id,
                                notice=f"Could not move the file: {e.strerror or e}. Nothing was changed."))

    conn.execute(
        "UPDATE media SET relpath = ? WHERE id = ?",
        (os.path.relpath(new_full, root), media_id),
    )
    conn.execute("DELETE FROM media_albums WHERE media_id = ?", (media_id,))
    if album_id:
        conn.execute(
            "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
            (media_id, album_id),
        )
        conn.execute(
            "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ? AND cover_media_id IS NULL",
            (media_id, album_id),
        )
    conn.commit()
    conn.close()
    where = os.path.relpath(os.path.dirname(new_full), root) or "the library folder"
    return redirect(url_for("media_detail", media_id=media_id, album=album_id,
                            notice=f"Moved. The file is now in {where}"))


@app.route("/media/<int:media_id>/copy", methods=["POST"])
def copy_media(media_id):
    """Duplicates the file into another collection's folder, leaving the
    original where it is — unlike move, both copies exist afterward."""
    album_id = request.form.get("album_id") or None
    new_album_name = request.form.get("new_album", "").strip()

    conn = get_db()
    row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)

    album_name = None
    if new_album_name:
        conn.execute(
            "INSERT OR IGNORE INTO albums (name, created_at) VALUES (?, ?)",
            (new_album_name, now_iso()),
        )
        arow = conn.execute("SELECT id FROM albums WHERE name = ?", (new_album_name,)).fetchone()
        album_id = arow["id"]
        album_name = new_album_name
    elif album_id:
        arow = conn.execute("SELECT name FROM albums WHERE id = ?", (album_id,)).fetchone()
        album_name = arow["name"] if arow else None

    if not album_id:
        conn.close()
        return redirect(url_for("media_detail", media_id=media_id))

    root = library_root()
    old_full = os.path.join(root, row["relpath"])
    dest_dir = _album_dest_dir(conn, root, album_id, album_name, _album_category_name(conn, album_id))
    os.makedirs(dest_dir, exist_ok=True)
    new_full = _unique_path(dest_dir, row["filename"])

    try:
        shutil.copy2(old_full, new_full)
    except OSError as e:
        conn.close()
        return redirect(url_for("media_detail", media_id=media_id,
                                notice=f"Could not copy the file: {e.strerror or e}. Nothing was changed."))

    new_relpath = os.path.relpath(new_full, root)
    try:
        size_bytes = os.path.getsize(new_full)
    except OSError:
        size_bytes = row["size_bytes"]

    cur = conn.execute(
        "INSERT INTO media (filename, relpath, media_type, width, height, size_bytes, added_at, taken_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (os.path.basename(new_full), new_relpath, row["media_type"], row["width"], row["height"],
         size_bytes, now_iso(), row["taken_at"]),
    )
    new_id = cur.lastrowid
    conn.execute("INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)", (new_id, album_id))
    conn.execute(
        "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ? AND cover_media_id IS NULL",
        (new_id, album_id),
    )
    conn.commit()
    conn.close()

    ensure_thumbnail(new_id, new_relpath, row["media_type"])

    where = os.path.relpath(os.path.dirname(new_full), root) or "the library folder"
    return redirect(url_for("media_detail", media_id=new_id, album=album_id,
                            notice=f"Copied. The new file is in {where}"))


@app.route("/media/<int:media_id>/delete", methods=["POST"])
def delete_media(media_id):
    conn = get_db()
    row = conn.execute("SELECT relpath FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        conn.close()
        abort(404)

    ok = _trash_media(conn, media_id)   # goes to the Trash, restorable from Cleanup
    conn.commit()
    conn.close()
    if not ok:
        return redirect(url_for("media_detail", media_id=media_id,
                                notice="Could not move the file to the Trash. Nothing was changed."))
    back = request.form.get("back_url") or url_for("home")
    return redirect(back)


# ---------------------------------------------------------------------------
# Bulk actions. The browser sends the selected ids in small batches so a big
# selection shows progress and never hits one long request.
# ---------------------------------------------------------------------------
BULK_BATCH_MAX = 500


def _bulk_ids_from(data):
    ids = []
    for i in data.get("ids") or []:
        try:
            ids.append(int(i))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ids))[:BULK_BATCH_MAX]


def _bulk_resolve_album(conn, data):
    """Returns (album_id, album_name, category_name) for the target channel of
    a bulk add/move/copy, creating the channel if a new name was typed."""
    new_name = (data.get("new_album") or "").strip()
    album_id = data.get("album_id") or None
    if new_name:
        conn.execute(
            "INSERT OR IGNORE INTO albums (name, created_at) VALUES (?, ?)",
            (new_name, now_iso()),
        )
        album_id = conn.execute("SELECT id FROM albums WHERE name = ?", (new_name,)).fetchone()["id"]
    if not album_id:
        return None, None, None
    row = conn.execute(
        "SELECT a.name, c.name AS category_name FROM albums a "
        "LEFT JOIN categories c ON c.id = a.category_id WHERE a.id = ?",
        (album_id,),
    ).fetchone()
    if not row:
        return None, None, None
    return int(album_id), row["name"], row["category_name"]


@app.route("/bulk/ids")
def bulk_ids():
    """Every media id matching the current filters (search, type, starred,
    channel, date range), so 'select all' can cover more than one page."""
    conn = get_db()
    joins, join_params, where_sql, params, _ctx = _filtered_media(request.args)
    rows = conn.execute(
        f"SELECT m.id FROM media m {joins} {where_sql} "
        "ORDER BY COALESCE(m.taken_at, m.added_at) DESC",
        join_params + params,
    ).fetchall()
    conn.close()
    return jsonify({"ids": [r["id"] for r in rows], "total": len(rows)})


@app.route("/bulk/action", methods=["POST"])
def bulk_action():
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    ids = _bulk_ids_from(data)
    if action == "album":
        action = "move"   # the old link-only action now moves the file on disk
    if action not in ("favorite", "album", "move", "copy", "delete") or not ids:
        return jsonify({"error": "Nothing to do."}), 400

    conn = get_db()
    root = library_root()
    done = failed = 0
    album_id = None

    if action == "favorite":
        value = 1 if data.get("value", True) else 0
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            marks = ",".join("?" * len(chunk))
            cur = conn.execute(f"UPDATE media SET favorite = ? WHERE id IN ({marks})", [value] + chunk)
            done += cur.rowcount
        conn.commit()

    elif action == "delete":
        for mid in ids:
            if _trash_media(conn, mid):
                done += 1
            else:
                failed += 1
        conn.commit()

    else:  # album / move / copy all need a target channel
        album_id, album_name, category_name = _bulk_resolve_album(conn, data)
        if not album_id:
            conn.close()
            return jsonify({"error": "Pick a channel first."}), 400

        if action == "album":
            for mid in ids:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO media_albums (media_id, album_id) "
                    "SELECT id, ? FROM media WHERE id = ?",
                    (album_id, mid),
                )
                if cur.rowcount:
                    done += 1
            conn.execute(
                "UPDATE albums SET cover_media_id = ? WHERE id = ? AND cover_media_id IS NULL",
                (ids[0], album_id),
            )
            conn.commit()

        else:
            dest_dir = _album_dest_dir(conn, root, album_id, album_name, category_name)
            os.makedirs(dest_dir, exist_ok=True)
            new_thumbs = []
            for mid in ids:
                row = conn.execute("SELECT * FROM media WHERE id = ?", (mid,)).fetchone()
                if not row:
                    failed += 1
                    continue
                old_full = os.path.join(root, row["relpath"])
                if not os.path.exists(old_full):
                    failed += 1
                    continue

                if action == "move":
                    new_full = old_full
                    if os.path.abspath(os.path.dirname(old_full)) != os.path.abspath(dest_dir):
                        new_full = _unique_path(dest_dir, row["filename"])
                        try:
                            shutil.move(old_full, new_full)
                        except OSError:
                            failed += 1
                            continue
                    conn.execute(
                        "UPDATE media SET relpath = ? WHERE id = ?",
                        (os.path.relpath(new_full, root), mid),
                    )
                    conn.execute("DELETE FROM media_albums WHERE media_id = ?", (mid,))
                    conn.execute(
                        "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
                        (mid, album_id),
                    )
                    conn.execute(
                        "UPDATE albums SET cover_media_id = NULL WHERE cover_media_id = ? AND id != ?",
                        (mid, album_id),
                    )
                    conn.execute(
                        "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ?",
                        (mid, album_id),
                    )
                    done += 1

                else:  # copy
                    new_full = _unique_path(dest_dir, row["filename"])
                    try:
                        shutil.copy2(old_full, new_full)
                    except OSError:
                        failed += 1
                        continue
                    new_rel = os.path.relpath(new_full, root)
                    try:
                        size_bytes = os.path.getsize(new_full)
                    except OSError:
                        size_bytes = row["size_bytes"]
                    values = {k: row[k] for k in row.keys() if k != "id"}
                    values.update({
                        "filename": os.path.basename(new_full),
                        "relpath": new_rel,
                        "size_bytes": size_bytes,
                        "added_at": now_iso(),
                        "favorite": 0,
                    })
                    cols = ", ".join(values)
                    marks = ", ".join("?" * len(values))
                    cur = conn.execute(
                        f"INSERT INTO media ({cols}) VALUES ({marks})", list(values.values())
                    )
                    new_id = cur.lastrowid
                    conn.execute(
                        "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
                        (new_id, album_id),
                    )
                    conn.execute(
                        "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ?",
                        (new_id, album_id),
                    )
                    new_thumbs.append((new_id, new_rel, row["media_type"]))
                    done += 1
            conn.commit()
            for new_id, new_rel, mtype in new_thumbs:
                try:
                    ensure_thumbnail(new_id, new_rel, mtype)
                except Exception:
                    pass

    conn.close()
    dest_rel = ""
    if action in ("move", "copy") and album_id:
        try:
            dest_rel = os.path.relpath(dest_dir, root)
        except Exception:
            dest_rel = ""
    return jsonify({"done": done, "failed": failed, "album_id": album_id, "dest": dest_rel})


@app.route("/feed/setup")
def feed_setup():
    """Ported from the old Discord bot's !feed — pick any combination of
    channels (not just one, and not a whole category) and start a single
    feed pulling from all of them at once. Channels are grouped by category
    here purely so a whole category can be toggled with one click and so
    search can match a category name too — the actual /feed request this
    form submits is still just a flat list of ?album= ids."""
    conn = get_db()
    channels = _all_albums(conn)
    conn.close()

    groups_by_key = {}
    order = []
    for ch in channels:
        key = ch["category_id"] if ch["category_id"] is not None else "__none__"
        if key not in groups_by_key:
            groups_by_key[key] = {
                "id": ch["category_id"],
                "name": ch["category_name"] or "Uncategorized",
                "channels": [],
            }
            order.append(key)
        groups_by_key[key]["channels"].append(ch)

    groups = sorted(
        (groups_by_key[k] for k in order),
        key=lambda g: (g["id"] is None, g["name"].lower()),
    )
    return render_template("feed_setup.html", channels=channels, groups=groups)


@app.route("/feed")
def feed():
    """The 'Start Feed' experience — a full-screen, auto-advancing run through
    matching media, shuffled by default. Filterable by one collection, several
    hand-picked collections at once (?album=1&album=2...), a category, or
    favorites — so 'Feed this collection', 'Feed everything', and the
    multi-channel feed setup page all go through here."""
    conn = get_db()
    album_ids = [a for a in request.args.getlist("album") if a]
    category_id = request.args.get("category")
    favorites_only = request.args.get("fav") == "1"
    shuffle = request.args.get("shuffle", "1") == "1"

    clauses = []
    params = []
    if favorites_only:
        clauses.append("m.favorite = 1")

    joins = ""
    join_params = []
    album = None
    albums_selected = []
    category = None
    if len(album_ids) == 1:
        joins += " JOIN media_albums ma ON ma.media_id = m.id AND ma.album_id = ?"
        join_params.append(album_ids[0])
        album = conn.execute("SELECT * FROM albums WHERE id = ?", (album_ids[0],)).fetchone()
        if not album:
            conn.close()
            abort(404)
    elif len(album_ids) > 1:
        placeholders = ",".join("?" for _ in album_ids)
        joins += f" JOIN media_albums ma ON ma.media_id = m.id JOIN albums a2 ON a2.id = ma.album_id AND a2.id IN ({placeholders})"
        join_params.extend(album_ids)
        albums_selected = conn.execute(f"SELECT * FROM albums WHERE id IN ({placeholders})", album_ids).fetchall()
        if not albums_selected:
            conn.close()
            abort(404)
    elif category_id:
        joins += " JOIN media_albums ma ON ma.media_id = m.id JOIN albums a2 ON a2.id = ma.album_id AND a2.category_id = ?"
        join_params.append(category_id)
        category = conn.execute("SELECT * FROM categories WHERE id = ?", (category_id,)).fetchone()
        if not category:
            conn.close()
            abort(404)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    items = conn.execute(
        f"SELECT DISTINCT m.* FROM media m {joins} {where_sql} ORDER BY COALESCE(m.taken_at, m.added_at) ASC",
        join_params + params,
    ).fetchall()

    # "Look for" filter: keep only the best matches for some words (e.g. saree)
    look_q = (request.args.get("look") or "").strip()[:80]
    empty_msg = ""
    if look_q:
        look_n = min(max(request.args.get("n", type=int) or 200, 10), 3000)
        try:
            ranked = look.apply_fit(
                look.search(conn, look_q, media_ids=[i["id"] for i in items], top=look_n),
                request.args.get("fit", "balanced"))
            by_id = {i["id"]: i for i in items}
            items = [by_id[mid] for mid, _s in ranked if mid in by_id]
            if not items:
                empty_msg = "Nothing to match yet. Open Look and index your photos first."
        except RuntimeError as e:
            items = []
            empty_msg = "Look is not ready: " + str(e)
    conn.close()

    items_list = [{"id": i["id"], "type": i["media_type"], "filename": i["filename"]} for i in items]
    if shuffle:
        random.shuffle(items_list)

    if album:
        title = album["name"]
    elif albums_selected:
        title = f"{len(albums_selected)} channels"
    elif category:
        title = category["name"]
    else:
        title = "Favorites" if favorites_only else "Everything"

    if look_q:
        title = f'{title} \u00b7 "{look_q}"'

    if album:
        back_url = url_for("collection_detail", album_id=album["id"])
    elif albums_selected:
        back_url = url_for("feed_setup")
    elif category:
        back_url = url_for("category_detail", category_id=category["id"])
    else:
        back_url = url_for("browse", fav=1 if favorites_only else None)

    return render_template(
        "feed.html",
        title=title,
        album=album,
        category=category,
        items_json=json.dumps(items_list),
        has_items=len(items_list) > 0,
        back_url=back_url,
        empty_msg=empty_msg,
    )


@app.route("/feed/session", methods=["POST"])
def record_feed_session():
    """Called by the feed page when a session ends (Stop Feed, or the tab/page
    closing), so 'most fed' data survives even if you never revisit the app.
    Only a channel-scoped feed (one specific collection) earns points — a
    session only counts once it ran at least MIN_FEED_DURATION_FOR_POINTS
    seconds, which filters out instantly opening then leaving a feed."""
    # force=True: the browser may send this beacon as text/plain (iOS Safari
    # is picky about beacon content types), so parse the body as JSON anyway.
    data = request.get_json(silent=True, force=True) or request.form
    album_id = data.get("album_id")
    try:
        duration_seconds = float(data.get("duration_seconds") or 0)
        items_shown = int(data.get("items_shown") or 0)
        last_media_id = int(data.get("last_media_id") or 0) or None
    except (TypeError, ValueError):
        return jsonify({"recorded": False})

    if not album_id or duration_seconds < MIN_FEED_DURATION_FOR_POINTS:
        return jsonify({"recorded": False})

    points = POINTS_PER_SESSION + (duration_seconds / 60.0) * POINTS_PER_MINUTE + items_shown * POINTS_PER_ITEM
    points = round(points, 1)

    conn = get_db()
    row = conn.execute("SELECT id FROM albums WHERE id = ?", (album_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"recorded": False})

    if last_media_id and not conn.execute("SELECT 1 FROM media WHERE id = ?", (last_media_id,)).fetchone():
        last_media_id = None
    conn.execute(
        """
        INSERT INTO feed_stats (album_id, points, sessions, last_fed_at, seconds_fed, items_shown, last_media_id)
        VALUES (?, ?, 1, ?, ?, ?, ?)
        ON CONFLICT(album_id) DO UPDATE SET
            points = ROUND(points + excluded.points, 1),
            sessions = sessions + 1,
            last_fed_at = excluded.last_fed_at,
            seconds_fed = seconds_fed + excluded.seconds_fed,
            items_shown = items_shown + excluded.items_shown,
            last_media_id = COALESCE(excluded.last_media_id, last_media_id)
        """,
        (album_id, points, now_iso(), duration_seconds, items_shown, last_media_id),
    )
    conn.commit()
    conn.close()
    return jsonify({"recorded": True, "points": points})


@app.route("/leaderboard")
def leaderboard():
    """All-time 'most fed' channels, ported from the old Discord bot's
    !totalvotes points board — a personal stat, not multiplayer voting."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT a.id, a.name, a.category_id, c.name AS category_name,
               fs.points, fs.sessions, fs.last_fed_at, fs.seconds_fed, fs.items_shown
        FROM feed_stats fs
        JOIN albums a ON a.id = fs.album_id
        LEFT JOIN categories c ON c.id = a.category_id
        ORDER BY fs.points DESC
        LIMIT 50
        """
    ).fetchall()

    category_totals = conn.execute(
        """
        SELECT COALESCE(c.name, 'Uncategorized') AS name, SUM(fs.points) AS points,
               SUM(fs.seconds_fed) AS seconds_fed, SUM(fs.items_shown) AS items_shown
        FROM feed_stats fs
        JOIN albums a ON a.id = fs.album_id
        LEFT JOIN categories c ON c.id = a.category_id
        GROUP BY c.id
        ORDER BY points DESC
        """
    ).fetchall()
    agg = conn.execute(
        """
        SELECT COALESCE(SUM(points), 0) AS points, COALESCE(SUM(seconds_fed), 0) AS seconds,
               COALESCE(SUM(items_shown), 0) AS items, COALESCE(SUM(sessions), 0) AS sessions
        FROM feed_stats
        """
    ).fetchone()
    spotlight = None
    if rows:
        top = rows[0]
        cover = conn.execute(
            "SELECT cover_media_id FROM albums WHERE id = ?", (top["id"],)
        ).fetchone()
        last = conn.execute(
            "SELECT m.id, m.media_type, m.filename FROM feed_stats fs "
            "JOIN media m ON m.id = fs.last_media_id WHERE fs.album_id = ?",
            (top["id"],),
        ).fetchone()
        if not last and cover and cover["cover_media_id"]:
            last = conn.execute(
                "SELECT id, media_type, filename FROM media WHERE id = ?",
                (cover["cover_media_id"],),
            ).fetchone()
            spot_is_last = False
        else:
            spot_is_last = bool(last)
        spotlight = {"row": top, "media": last, "is_last_stop": spot_is_last}
    conn.close()

    totals = {
        "points": round(agg["points"], 1),
        "seconds": agg["seconds"],
        "viewed": agg["items"],
        "sessions": agg["sessions"],
    }
    return render_template("leaderboard.html", rows=rows, category_totals=category_totals,
                           totals=totals, spotlight=spotlight)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        conn = get_db()
        albums = _all_albums(conn)
        categories = _all_categories(conn)
        conn.close()
        return render_template("upload.html", albums=albums, categories=categories,
                               faces_ready=faces.engine_status()["ready"])

    files = request.files.getlist("files")
    album_id = request.form.get("album_id") or None
    new_album_name = request.form.get("new_album", "").strip()
    category_id_form = request.form.get("category_id") or None
    new_category_name = request.form.get("new_category", "").strip()

    root = library_root()
    os.makedirs(root, exist_ok=True)

    conn = get_db()
    album_name = None
    category_name = None
    if new_album_name:
        # A brand-new channel can be filed straight into a chosen (or
        # freshly typed) category, instead of always landing uncategorized.
        category_id = _get_or_create_category(conn, category_id_form, new_category_name)
        conn.execute(
            "INSERT OR IGNORE INTO albums (name, created_at, category_id) VALUES (?, ?, ?)",
            (new_album_name, now_iso(), category_id),
        )
        row = conn.execute("SELECT id FROM albums WHERE name = ?", (new_album_name,)).fetchone()
        album_id = row["id"]
        album_name = new_album_name
        if category_id:
            conn.execute(
                "UPDATE albums SET category_id = ? WHERE id = ? AND category_id IS NULL",
                (category_id, album_id),
            )
            cat_row = conn.execute("SELECT name FROM categories WHERE id = ?", (category_id,)).fetchone()
            category_name = cat_row["name"] if cat_row else None
    elif album_id:
        row = conn.execute(
            "SELECT a.name, c.name AS category_name FROM albums a "
            "LEFT JOIN categories c ON c.id = a.category_id WHERE a.id = ?",
            (album_id,),
        ).fetchone()
        album_name = row["name"] if row else None
        category_name = row["category_name"] if row else None
    conn.commit()

    # Files land wherever this channel's folder already lives on disk (or,
    # for a brand new channel, inside a fresh Category/Channel folder), so
    # what you see in Postmark actually matches how the library is laid out.
    dest_dir = _album_dest_dir(conn, root, album_id, album_name, category_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_relpaths = []
    for f in files:
        if not f or not f.filename:
            continue
        fname = secure_filename(f.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in PHOTO_EXTS and ext not in VIDEO_EXTS:
            continue

        dest = _unique_path(dest_dir, fname)
        f.save(dest)
        saved_relpaths.append(os.path.relpath(dest, root))

    conn.close()

    scan_library(source="upload")

    if album_id and saved_relpaths:
        # Actually attach the just-uploaded files to the collection (previously
        # this only ever set a cover image and never linked the media itself,
        # so uploads picked a collection but never showed up inside it).
        conn = get_db()
        placeholders = ",".join("?" for _ in saved_relpaths)
        rows = conn.execute(
            f"SELECT id FROM media WHERE relpath IN ({placeholders})", saved_relpaths
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
                (row["id"], album_id),
            )
        if rows:
            conn.execute(
                "UPDATE albums SET cover_media_id = COALESCE(cover_media_id, ?) WHERE id = ? AND cover_media_id IS NULL",
                (rows[0]["id"], album_id),
            )
        conn.commit()
        conn.close()

    if saved_relpaths:
        conn = get_db()
        placeholders = ",".join("?" for _ in saved_relpaths)
        new_ids = [r["id"] for r in conn.execute(
            f"SELECT id FROM media WHERE relpath IN ({placeholders}) ORDER BY id", saved_relpaths
        ).fetchall()]
        conn.close()
        look.index_async(new_ids)   # so "Look" searches can find them
        if album_id:
            # Uploaded straight into a channel: quietly learn their faces.
            faces.index_async(new_ids)
        elif new_ids and faces.engine_status()["ready"]:
            # No channel chosen: suggest one from the faces.
            return redirect(url_for("upload_review", ids=",".join(str(i) for i in new_ids[:300])))

    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Face matching: set-up page, channel suggestions for new uploads.
# ---------------------------------------------------------------------------
def _parse_ids(raw, limit):
    ids = []
    for part in (raw or "").split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return list(dict.fromkeys(ids))[:limit]


@app.route("/faces")
def faces_page():
    conn = get_db()
    st = faces.stats(conn)
    conn.close()
    return render_template("faces.html", status=faces.engine_status(), stats=st,
                           models_dir=faces.models_dir(),
                           model_urls=[(n, u) for n, u in faces.MODELS.values()])


@app.route("/faces/status")
def faces_status():
    conn = get_db()
    st = faces.stats(conn)
    conn.close()
    return jsonify({
        "engine": faces.engine_status(),
        "download": dict(faces.DOWNLOAD),
        "index": {k: v for k, v in faces.INDEX.items() if k != "stop"},
        "stats": st,
    })


@app.route("/faces/models", methods=["POST"])
def faces_models():
    faces.start_download()
    return jsonify({"started": True})


@app.route("/faces/index", methods=["POST"])
def faces_index():
    per_channel = request.form.get("per_channel", type=int) or 40
    per_channel = min(max(per_channel, 5), 300)
    started = faces.start_index(per_channel)
    return jsonify({"started": started})


@app.route("/faces/stop", methods=["POST"])
def faces_stop():
    faces.stop_index()
    return jsonify({"stopped": True})


@app.route("/faces/test", methods=["POST"])
def faces_test():
    """Send a picture, see which channels its face(s) look like. Saves nothing."""
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Choose a picture first."}), 400
    if not faces.engine_status()["ready"]:
        return jsonify({"error": faces.engine_status()["reason"]}), 503
    conn = get_db()
    try:
        return jsonify(faces.identify_bytes(conn, f.read()))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"error": "That file could not be read as a picture."}), 400
    finally:
        conn.close()


@app.route("/faces/selftest", methods=["POST"])
def faces_selftest():
    if not faces.engine_status()["ready"]:
        return jsonify({"error": faces.engine_status()["reason"]}), 503
    conn = get_db()
    try:
        return jsonify(faces.self_test(conn))
    finally:
        conn.close()


@app.route("/faces/suggest")
def faces_suggest():
    """Channel guesses for a few items at a time (the review page asks in
    small batches so it can show progress)."""
    ids = _parse_ids(request.args.get("ids"), 8)
    if not faces.engine_status()["ready"]:
        return jsonify({"error": faces.engine_status()["reason"]}), 503
    conn = get_db()
    results = {}
    try:
        matrix = faces._load_matrix(conn)
        for mid in ids:
            row = conn.execute(
                "SELECT relpath, media_type FROM media WHERE id = ?", (mid,)
            ).fetchone()
            if not row:
                continue
            try:
                found = faces.analyze(conn, mid, row["relpath"], row["media_type"])
            except Exception as e:
                results[mid] = {"status": "error", "error": str(e), "candidates": []}
                continue
            if not found:
                results[mid] = {"status": "noface", "candidates": []}
                continue
            cands = faces.suggest(conn, [v for _a, v in found], exclude_media=mid, matrix=matrix)
            results[mid] = {"status": "ok" if cands else "nomatch", "faces": len(found),
                            "candidates": cands}
    finally:
        conn.close()
    return jsonify({"results": results})


@app.route("/upload/review")
def upload_review():
    ids = _parse_ids(request.args.get("ids"), 300)
    conn = get_db()
    rows = []
    if ids:
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, filename, media_type FROM media WHERE id IN ({marks}) ORDER BY id", ids
        ).fetchall()
    conn.close()
    return render_template("upload_review.html", items=rows, status=faces.engine_status())


# ---------------------------------------------------------------------------
# Look: search and feed by what is in the picture ("saree", "beach"...).
# ---------------------------------------------------------------------------
@app.route("/look")
def look_page():
    conn = get_db()
    st = look.stats(conn)
    conn.close()
    return render_template("look.html", status=look.engine_status(), stats=st,
                           models_dir=look.models_dir(),
                           model_urls=[(n, u) for n, u in look.MODELS.values()])


@app.route("/look/status")
def look_status():
    conn = get_db()
    st = look.stats(conn)
    conn.close()
    idx = {k: v for k, v in look.INDEX.items() if k not in ("stop", "started")}
    if look.INDEX["running"] and look.INDEX["processed"]:
        idx["per_min"] = round(look.INDEX["processed"] / max((time.time() - look.INDEX["started"]) / 60.0, 0.01), 1)
    return jsonify({"engine": look.engine_status(), "download": dict(look.DOWNLOAD),
                    "index": idx, "stats": st, "tokenizer_ok": look.tokenizer_check()})


@app.route("/look/models", methods=["POST"])
def look_models():
    look.start_download()
    return jsonify({"started": True})


@app.route("/look/index", methods=["POST"])
def look_index():
    return jsonify({"started": look.start_index()})


@app.route("/look/stop", methods=["POST"])
def look_stop():
    look.stop_index()
    return jsonify({"stopped": True})


@app.route("/look/search")
def look_search():
    q = (request.args.get("q") or "").strip()[:80]
    n = min(max(request.args.get("n", type=int) or 300, 1), 1000)
    shown = min(max(request.args.get("show", type=int) or 60, 1), 300)
    if not q:
        return jsonify({"results": []})
    if not look.engine_status()["ready"]:
        return jsonify({"error": look.engine_status()["reason"]}), 503
    conn = get_db()
    try:
        ranked = look.apply_fit(look.search(conn, q, top=n), request.args.get("fit", "balanced"))
        matches = len(ranked)
        ranked = ranked[:shown]
        types = {}
        if ranked:
            marks = ",".join("?" * len(ranked))
            types = {r["id"]: r["media_type"] for r in conn.execute(
                f"SELECT id, media_type FROM media WHERE id IN ({marks})", [m for m, _ in ranked])}
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        conn.close()
    return jsonify({"matches": matches,
                    "results": [{"id": m, "score": round(s, 3), "type": types.get(m, "photo")} for m, s in ranked]})



# ---------------------------------------------------------------------------
# Exact duplicates: finds byte-identical files and lets the person delete extras.
# ---------------------------------------------------------------------------
DUP = {"running": False, "done": False, "error": "", "stage": "", "processed": 0, "total": 0,
       "groups": [], "stop": False}
_DUP_LOCK = threading.Lock()


def _hash_file(path, quick):
    h = hashlib.sha1()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if quick:
            # First, middle and last 256 KB: cheap way to rule out most non-matches.
            for off in (0, max(size // 2 - 131072, 0), max(size - 262144, 0)):
                f.seek(off)
                h.update(f.read(262144))
        else:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def _dup_job():
    try:
        root = library_root()
        conn = get_db()
        rows = conn.execute("SELECT id, relpath FROM media").fetchall()
        conn.close()
        DUP.update({"stage": "Comparing file sizes", "processed": 0, "total": len(rows)})
        by_size = {}
        for i, r in enumerate(rows):
            if DUP["stop"]:
                raise RuntimeError("Stopped.")
            try:
                sz = os.path.getsize(os.path.join(root, r["relpath"]))
            except OSError:
                continue
            if sz > 0:
                by_size.setdefault(sz, []).append(r["id"])
            DUP["processed"] = i + 1
        cands = {sz: ids for sz, ids in by_size.items() if len(ids) > 1}
        paths = {r["id"]: os.path.join(root, r["relpath"]) for r in rows}
        total = sum(len(v) for v in cands.values())
        DUP.update({"stage": "Checking possible matches", "processed": 0, "total": total})
        done = 0
        groups = []
        for sz, ids in cands.items():
            quick = {}
            for mid in ids:
                if DUP["stop"]:
                    raise RuntimeError("Stopped.")
                try:
                    quick.setdefault(_hash_file(paths[mid], True), []).append(mid)
                except OSError:
                    pass
                done += 1
                DUP["processed"] = done
            for qids in quick.values():
                if len(qids) < 2:
                    continue
                full = {}
                for mid in qids:
                    if DUP["stop"]:
                        raise RuntimeError("Stopped.")
                    try:
                        full.setdefault(_hash_file(paths[mid], False), []).append(mid)
                    except OSError:
                        pass
                for fids in full.values():
                    if len(fids) > 1:
                        groups.append({"size": sz, "ids": fids})
        conn = get_db()
        out = []
        for g in groups:
            marks = ",".join("?" * len(g["ids"]))
            items = conn.execute(
                f"SELECT id, filename, relpath, media_type, favorite, added_at FROM media WHERE id IN ({marks})",
                g["ids"]).fetchall()
            members = []
            for it in items:
                chans = [c["name"] for c in conn.execute(
                    "SELECT a.name FROM albums a JOIN media_albums ma ON ma.album_id = a.id WHERE ma.media_id = ?",
                    (it["id"],))]
                members.append({"id": it["id"], "filename": it["filename"],
                                "folder": os.path.dirname(it["relpath"]) or "(library root)",
                                "type": it["media_type"], "favorite": bool(it["favorite"]),
                                "channels": chans, "added": (it["added_at"] or "")[:10]})
            # Suggested keeper: starred first, then one that is in a channel, then the oldest.
            members.sort(key=lambda m: (not m["favorite"], not m["channels"], m["id"]))
            for n, m in enumerate(members):
                m["keep"] = (n == 0)
            out.append({"size": g["size"], "size_h": _human_size(g["size"]), "items": members})
        conn.close()
        out.sort(key=lambda g: -g["size"] * (len(g["items"]) - 1))
        DUP["groups"] = out
        DUP["error"] = ""
    except Exception as e:  # noqa: BLE001
        DUP["error"] = str(e)
    finally:
        DUP["running"] = False
        DUP["done"] = True
        DUP["stop"] = False


@app.route("/duplicates")
def duplicates_page():
    return redirect(url_for("cleanup_page", tab="dup"))


@app.route("/duplicates/status")
def duplicates_status():
    d = {k: v for k, v in DUP.items() if k != "stop"}
    d["groups"] = DUP["groups"][:150]
    d["group_count"] = len(DUP["groups"])
    d["extra_files"] = sum(len(g["items"]) - 1 for g in DUP["groups"])
    d["extra_bytes"] = sum(g["size"] * (len(g["items"]) - 1) for g in DUP["groups"])
    d["extra_size"] = _human_size(d["extra_bytes"])
    return jsonify(d)


@app.route("/duplicates/scan", methods=["POST"])
def duplicates_scan():
    with _DUP_LOCK:
        if DUP["running"]:
            return jsonify({"started": False})
        DUP.update({"running": True, "done": False, "error": "", "groups": [], "stop": False,
                    "processed": 0, "total": 0, "stage": "Starting"})
    threading.Thread(target=_dup_job, daemon=True).start()
    return jsonify({"started": True})


@app.route("/duplicates/stop", methods=["POST"])
def duplicates_stop():
    DUP["stop"] = True
    return jsonify({"ok": True})


@app.route("/duplicates/delete", methods=["POST"])
def duplicates_delete():
    """Deletes the ticked copies. Refuses to remove every copy in a group, and
    links the surviving copy to the channels the deleted ones were in."""
    data = request.get_json(silent=True) or {}
    ids = set(int(i) for i in data.get("ids", []) if str(i).isdigit())
    if not ids:
        return jsonify({"error": "Nothing selected."}), 400
    root = library_root()
    conn = get_db()
    deleted = freed = skipped = 0
    for g in DUP["groups"]:
        gids = [m["id"] for m in g["items"]]
        pick = [i for i in gids if i in ids]
        if not pick:
            continue
        keepers = [i for i in gids if i not in ids]
        if not keepers:
            skipped += len(pick)   # would delete every copy
            continue
        keeper = keepers[0]
        for mid in pick:
            row = conn.execute("SELECT relpath, favorite FROM media WHERE id = ?", (mid,)).fetchone()
            if not row:
                continue
            for a in conn.execute("SELECT album_id FROM media_albums WHERE media_id = ?", (mid,)).fetchall():
                conn.execute("INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
                             (keeper, a["album_id"]))
            if row["favorite"]:
                conn.execute("UPDATE media SET favorite = 1 WHERE id = ?", (keeper,))
            full = os.path.join(root, row["relpath"])
            try:
                size_here = os.path.getsize(full)
            except OSError:
                size_here = 0
            if _trash_media(conn, mid):
                freed += size_here
                deleted += 1
            else:
                skipped += 1
        g["items"] = [m for m in g["items"] if m["id"] not in ids]
    conn.commit()
    conn.close()
    DUP["groups"] = [g for g in DUP["groups"] if len(g["items"]) > 1]
    return jsonify({"deleted": deleted, "skipped": skipped, "freed": _human_size(freed)})


# ---------------------------------------------------------------------------
# Settings (tiny key/value store), Trash, and the background helpers
# ---------------------------------------------------------------------------
def _setting(conn, key, default=None):
    r = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return r["value"] if r and r["value"] is not None else default


def _set_setting(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))


def trash_root():
    d = os.path.join(data_root(), "trash")
    os.makedirs(d, exist_ok=True)
    return d


def _media_columns(conn):
    return {r["name"] for r in conn.execute("PRAGMA table_info(media)")}


def _trash_media(conn, mid):
    """Moves one item to the Trash: the file goes to .data/trash, the database
    row is saved so a restore is exact. Returns False if it could not be moved
    (nothing is lost in that case). Caller commits."""
    row = conn.execute("SELECT * FROM media WHERE id = ?", (mid,)).fetchone()
    if not row:
        return False
    src = os.path.join(library_root(), row["relpath"])
    albums = [r["album_id"] for r in conn.execute("SELECT album_id FROM media_albums WHERE media_id = ?", (mid,))]
    if os.path.exists(src):
        try:
            size = os.path.getsize(src)
        except OSError:
            size = row["size_bytes"] or 0
        cur = conn.execute(
            "INSERT INTO trash (media_id, filename, relpath, trash_path, size_bytes, media_type, row_json, album_ids, deleted_at) "
            "VALUES (?, ?, ?, '', ?, ?, ?, ?, ?)",
            (mid, row["filename"], row["relpath"], size, row["media_type"],
             json.dumps(dict(row)), json.dumps(albums), now_iso()),
        )
        tid = cur.lastrowid
        dest = os.path.join(trash_root(), f"{tid}_{os.path.basename(src)}")
        try:
            shutil.move(src, dest)
        except OSError:
            conn.execute("DELETE FROM trash WHERE id = ?", (tid,))
            return False
        conn.execute("UPDATE trash SET trash_path = ? WHERE id = ?", (dest, tid))
    for t in ("face_embeddings", "image_embeddings", "photo_stats"):
        try:
            conn.execute(f"DELETE FROM {t} WHERE media_id = ?", (mid,))
        except Exception:  # noqa: BLE001
            pass
    conn.execute("DELETE FROM media WHERE id = ?", (mid,))
    return True


def _restore_trash(conn, tid):
    """Puts a trashed item back where it was (its old channels too). (ok, message)."""
    t = conn.execute("SELECT * FROM trash WHERE id = ?", (tid,)).fetchone()
    if not t:
        return False, "Not in the Trash."
    src = t["trash_path"]
    if not src or not os.path.exists(src):
        conn.execute("DELETE FROM trash WHERE id = ?", (tid,))
        return False, "The file is gone."
    root = library_root()
    dest_full = os.path.join(root, t["relpath"])
    dest_dir = os.path.dirname(dest_full)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(dest_full):
            dest_full = _unique_path(dest_dir, os.path.basename(dest_full))
        shutil.move(src, dest_full)
    except OSError as e:
        return False, e.strerror or str(e)
    row = json.loads(t["row_json"] or "{}")
    row["relpath"] = os.path.relpath(dest_full, root)
    row["filename"] = os.path.basename(dest_full)
    if row.get("id") is not None and conn.execute("SELECT 1 FROM media WHERE id = ?", (row["id"],)).fetchone():
        row.pop("id", None)
    cols = _media_columns(conn)
    row = {k: v for k, v in row.items() if k in cols}
    conn.execute(f"INSERT INTO media ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    new_id = row.get("id") or conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for aid in json.loads(t["album_ids"] or "[]"):
        if conn.execute("SELECT 1 FROM albums WHERE id = ?", (aid,)).fetchone():
            conn.execute("INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)", (new_id, aid))
    conn.execute("DELETE FROM trash WHERE id = ?", (tid,))
    return True, new_id


def _purge_trash_row(conn, t):
    if t["trash_path"]:
        try:
            os.remove(t["trash_path"])
        except OSError:
            pass
    if t["media_id"]:
        for path in (thumbnail_path(t["media_id"]), preview_path(t["media_id"])):
            try:
                os.remove(path)
            except OSError:
                pass
    conn.execute("DELETE FROM trash WHERE id = ?", (t["id"],))


def _trash_days(conn):
    try:
        return int(_setting(conn, "trash_days", "30"))
    except ValueError:
        return 30


def _auto_purge(conn):
    """Deletes trash older than the chosen number of days (0 = keep forever)."""
    days = _trash_days(conn)
    if days <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute("SELECT * FROM trash WHERE deleted_at < ?", (cutoff,)).fetchall()
    for t in rows:
        _purge_trash_row(conn, t)
    if rows:
        conn.commit()
    return len(rows)


def _days_left(conn_days, deleted_at):
    if conn_days <= 0:
        return None
    try:
        d = datetime.fromisoformat(deleted_at)
    except (TypeError, ValueError):
        return None
    left = conn_days - (datetime.now(timezone.utc) - d).days
    return max(left, 0)


# ---------------------------------------------------------------------------
# Cleanup page
# ---------------------------------------------------------------------------
@app.route("/cleanup")
def cleanup_page():
    conn = get_db()
    albums = [{"id": a["id"], "name": a["name"]} for a in _all_albums(conn)]
    _auto_purge(conn)
    trash_count = conn.execute("SELECT COUNT(*) FROM trash").fetchone()[0]
    conn.close()
    st = look.engine_status()
    return render_template("cleanup.html", albums=albums, look_ready=st["ready"],
                           look_reason=st.get("reason") or "", trash_count=trash_count,
                           initial_tab=request.args.get("tab", "dup"))


@app.route("/cleanup/people")
def cleanup_people():
    st = look.engine_status()
    if not st["ready"]:
        return jsonify({"error": st["reason"]}), 503
    try:
        thr = float(request.args.get("thr", "0.25"))
    except ValueError:
        thr = 0.25
    kind = (request.args.get("kind") or "").strip()
    offset = max(request.args.get("offset", type=int) or 0, 0)
    conn = get_db()
    try:
        look.ensure_schema(conn)
        ranked = look.no_people(conn, thr)
        counts = {}
        for _m, _p, k in ranked:
            counts[k] = counts.get(k, 0) + 1
        if kind:
            ranked = [r for r in ranked if r[2] == kind]
        page = ranked[offset:offset + 60]
        left = conn.execute(
            "SELECT COUNT(*) FROM media WHERE media_type = 'photo' AND id NOT IN "
            "(SELECT media_id FROM image_embeddings)").fetchone()[0]
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        conn.close()
    return jsonify({"items": [{"id": m, "kind": k, "p": round(p, 3)} for m, p, k in page],
                    "total": len(ranked), "counts": counts, "left_to_index": left})


FACECHECK = {"running": False, "processed": 0, "total": 0, "error": "", "stop": False}


def _faceless_ids(conn, include_in_channel):
    faces.ensure_schema(conn)
    person = look.person_ids(conn)
    if not person:
        return [], []
    have = {r["media_id"]: (r["vec"] is None) for r in conn.execute("SELECT media_id, vec FROM face_embeddings")}
    in_channel = {r["media_id"] for r in conn.execute("SELECT DISTINCT media_id FROM media_albums")}
    exist = {r["id"] for r in conn.execute("SELECT id FROM media WHERE media_type = 'photo'")}
    faceless, unchecked = [], []
    for m in person:
        if m not in exist:
            continue
        if m not in have:
            unchecked.append(m)
        elif have[m] and (include_in_channel or m not in in_channel):
            faceless.append(m)
    return faceless, unchecked


@app.route("/cleanup/faceless")
def cleanup_faceless():
    st = look.engine_status()
    if not st["ready"]:
        return jsonify({"error": st["reason"]}), 503
    offset = max(request.args.get("offset", type=int) or 0, 0)
    include = request.args.get("all") == "1"
    conn = get_db()
    try:
        faceless, unchecked = _faceless_ids(conn, include)
        faceless.sort(reverse=True)
        page = faceless[offset:offset + 20]
        guesses = look.channel_guess(conn, page) if page else {}
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        conn.close()
    items = [{"id": m, "guess": [{"id": a, "name": n, "score": round(s, 3)} for a, n, s in guesses.get(m, [])]}
             for m in page]
    return jsonify({"items": items, "total": len(faceless), "unchecked": len(unchecked),
                    "faces_ready": faces.engine_status()["ready"]})


def _facecheck_job():
    conn = get_db()
    try:
        _faceless, unchecked = _faceless_ids(conn, True)
        FACECHECK["total"] = len(unchecked)
        for mid in unchecked:
            if FACECHECK["stop"]:
                break
            row = conn.execute("SELECT relpath, media_type FROM media WHERE id = ?", (mid,)).fetchone()
            if row:
                try:
                    faces.analyze(conn, mid, row["relpath"], row["media_type"])
                except Exception as e:  # noqa: BLE001
                    FACECHECK["error"] = str(e)
                    break
            FACECHECK["processed"] += 1
    finally:
        conn.close()
        FACECHECK["running"] = False


@app.route("/cleanup/facecheck", methods=["POST"])
def cleanup_facecheck():
    if not faces.engine_status()["ready"]:
        return jsonify({"error": faces.engine_status()["reason"]}), 503
    if FACECHECK["running"]:
        return jsonify({"started": False})
    FACECHECK.update({"running": True, "processed": 0, "total": 0, "error": "", "stop": False})
    threading.Thread(target=_facecheck_job, daemon=True).start()
    return jsonify({"started": True})


@app.route("/cleanup/facecheck/status")
def cleanup_facecheck_status():
    return jsonify({k: v for k, v in FACECHECK.items() if k != "stop"})


@app.route("/cleanup/facecheck/stop", methods=["POST"])
def cleanup_facecheck_stop():
    FACECHECK["stop"] = True
    return jsonify({"stopped": True})


# ---------------------------------------------------------------------------
# Photo analysis (one pass gives near-duplicate fingerprints AND blur/dark numbers)
# ---------------------------------------------------------------------------
STATS = {"running": False, "done": False, "error": "", "processed": 0, "total": 0, "stop": False}


def _photo_numbers(mid, relpath):
    """(dhash_hex, blur, brightness) from the cached thumbnail (fast)."""
    from PIL import Image, ImageFilter, ImageStat
    tp = thumbnail_path(mid)
    if not os.path.exists(tp):
        ensure_thumbnail(mid, relpath, "photo")
    src = tp if os.path.exists(tp) else os.path.join(library_root(), relpath)
    with Image.open(src) as im:
        g = im.convert("L")
        g.thumbnail((256, 256))
        small = g.resize((9, 8), Image.Resampling.BOX)
        px = list(small.getdata())
        bits = 0
        for r in range(8):
            for c in range(8):
                bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
        lap = g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
        blur = ImageStat.Stat(lap).var[0]
        bright = ImageStat.Stat(g).mean[0]
    return f"{bits:016x}", float(blur), float(bright)


def _stats_job():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT m.id, m.relpath FROM media m LEFT JOIN photo_stats p ON p.media_id = m.id "
            "WHERE m.media_type = 'photo' AND p.media_id IS NULL ORDER BY m.id").fetchall()
        STATS["total"] = len(rows)
        for r in rows:
            if STATS["stop"]:
                break
            try:
                res = _photo_numbers(r["id"], r["relpath"])
            except Exception:  # noqa: BLE001
                res = (None, None, None)   # unreadable: remembered so it is not retried forever
            conn.execute("INSERT OR REPLACE INTO photo_stats (media_id, dhash, blur, brightness, checked_at) "
                         "VALUES (?, ?, ?, ?, ?)", (r["id"],) + res + (now_iso(),))
            STATS["processed"] += 1
            if STATS["processed"] % 50 == 0:
                conn.commit()
        conn.commit()
        STATS["error"] = ""
    except Exception as e:  # noqa: BLE001
        STATS["error"] = str(e)
    finally:
        conn.close()
        STATS["running"] = False
        STATS["done"] = True
        STATS["stop"] = False


@app.route("/cleanup/scan", methods=["POST"])
def cleanup_scan():
    if STATS["running"]:
        return jsonify({"started": False})
    STATS.update({"running": True, "done": False, "error": "", "processed": 0, "total": 0, "stop": False})
    threading.Thread(target=_stats_job, daemon=True).start()
    return jsonify({"started": True})


@app.route("/cleanup/scan/status")
def cleanup_scan_status():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM media WHERE media_type = 'photo'").fetchone()[0]
    done = conn.execute("SELECT COUNT(*) FROM photo_stats").fetchone()[0]
    conn.close()
    d = {k: v for k, v in STATS.items() if k != "stop"}
    d.update({"photos": total, "analyzed": done, "left": max(total - done, 0)})
    return jsonify(d)


@app.route("/cleanup/scan/stop", methods=["POST"])
def cleanup_scan_stop():
    STATS["stop"] = True
    return jsonify({"stopped": True})


SIM_LEVELS = {"strict": 3, "normal": 6, "loose": 9}
_SIM_CACHE = {"key": None, "groups": []}


def _similar_groups(conn, maxd):
    """Groups of photos whose fingerprints are within maxd bits. Uses 8 bands of
    8 bits so only photos sharing a band are compared (fast on 15k photos)."""
    rows = conn.execute(
        "SELECT p.media_id AS id, p.dhash, m.width, m.height, m.size_bytes, m.favorite, m.filename, m.relpath "
        "FROM photo_stats p JOIN media m ON m.id = p.media_id WHERE p.dhash IS NOT NULL").fetchall()
    key = (len(rows), rows[-1]["id"] if rows else 0, maxd)
    if _SIM_CACHE["key"] == key:
        return _SIM_CACHE["groups"]
    info = {r["id"]: r for r in rows}
    vals = [(r["id"], int(r["dhash"], 16)) for r in rows]
    parent = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent.get(x, x)
        return root

    def ratio(r):
        return (r["width"] / r["height"]) if r["width"] and r["height"] else None

    buckets = [dict() for _ in range(8)]
    for mid, h in vals:
        for b in range(8):
            buckets[b].setdefault((h >> (8 * b)) & 0xFF, []).append((mid, h))
    for bk in buckets:
        for lst in bk.values():
            if len(lst) < 2 or len(lst) > 900:
                continue   # a huge bucket is flat/blank pictures, not real matches
            for i in range(len(lst)):
                ai, ah = lst[i]
                ra = ratio(info[ai])
                for j in range(i + 1, len(lst)):
                    bi, bh = lst[j]
                    if bin(ah ^ bh).count("1") > maxd:
                        continue
                    rb = ratio(info[bi])
                    if ra and rb and abs(ra - rb) > 0.15:
                        continue
                    x, y = find(ai), find(bi)
                    if x != y:
                        parent[x] = y
    comp = {}
    for mid, _h in vals:
        comp.setdefault(find(mid), []).append(mid)
    groups = []
    for ids in comp.values():
        if len(ids) < 2:
            continue
        ids.sort(key=lambda i: (-(info[i]["favorite"] or 0),
                                -((info[i]["width"] or 0) * (info[i]["height"] or 0)),
                                -(info[i]["size_bytes"] or 0), i))
        groups.append(ids)
    groups.sort(key=lambda g: (-len(g), g[0]))
    _SIM_CACHE.update({"key": key, "groups": groups})
    return groups


@app.route("/cleanup/similar")
def cleanup_similar():
    level = request.args.get("sens", "normal")
    maxd = SIM_LEVELS.get(level, 6)
    offset = max(request.args.get("offset", type=int) or 0, 0)
    conn = get_db()
    groups = _similar_groups(conn, maxd)
    page = groups[offset:offset + 15]
    ids = [i for g in page for i in g]
    meta, chans = {}, {}
    if ids:
        marks = ",".join("?" * len(ids))
        for r in conn.execute(f"SELECT id, filename, relpath, width, height, size_bytes, favorite FROM media WHERE id IN ({marks})", ids):
            meta[r["id"]] = r
        for r in conn.execute(f"SELECT ma.media_id, a.name FROM media_albums ma JOIN albums a ON a.id = ma.album_id WHERE ma.media_id IN ({marks})", ids):
            chans.setdefault(r["media_id"], []).append(r["name"])
    left = conn.execute("SELECT COUNT(*) FROM media m LEFT JOIN photo_stats p ON p.media_id = m.id "
                        "WHERE m.media_type = 'photo' AND p.media_id IS NULL").fetchone()[0]
    conn.close()
    out = []
    for g in page:
        out.append([{"id": i, "filename": meta[i]["filename"], "folder": os.path.dirname(meta[i]["relpath"]) or "(root)",
                     "dims": f'{meta[i]["width"]}x{meta[i]["height"]}' if meta[i]["width"] else "",
                     "size": _human_size(meta[i]["size_bytes"] or 0), "favorite": bool(meta[i]["favorite"]),
                     "channels": chans.get(i, []), "keep": n == 0}
                    for n, i in enumerate(g) if i in meta])
    return jsonify({"groups": out, "total": len(groups), "extra": sum(len(g) - 1 for g in groups), "unanalyzed": left})


QUAL_LEVELS = {
    "strict": {"blur": 25, "dark": 25, "bright": 240, "lowres": 100_000},
    "normal": {"blur": 60, "dark": 40, "bright": 225, "lowres": 300_000},
    "loose":  {"blur": 120, "dark": 60, "bright": 210, "lowres": 600_000},
}
QUAL_KINDS = {"blur": "Blurry", "dark": "Too dark", "bright": "Overexposed", "lowres": "Low resolution"}


def _quality_where(kind, lim):
    if kind == "blur":
        return "p.blur IS NOT NULL AND p.blur < ?", lim["blur"], "p.blur ASC"
    if kind == "dark":
        return "p.brightness IS NOT NULL AND p.brightness < ?", lim["dark"], "p.brightness ASC"
    if kind == "bright":
        return "p.brightness IS NOT NULL AND p.brightness > ?", lim["bright"], "p.brightness DESC"
    return "m.width IS NOT NULL AND m.height IS NOT NULL AND m.width * m.height < ?", lim["lowres"], "m.width * m.height ASC"


@app.route("/cleanup/quality")
def cleanup_quality():
    kind = request.args.get("kind", "blur")
    if kind not in QUAL_KINDS:
        kind = "blur"
    lim = QUAL_LEVELS.get(request.args.get("sens", "normal"), QUAL_LEVELS["normal"])
    offset = max(request.args.get("offset", type=int) or 0, 0)
    conn = get_db()
    counts = {}
    for k in QUAL_KINDS:
        w, val, _o = _quality_where(k, lim)
        counts[k] = conn.execute(
            f"SELECT COUNT(*) FROM media m LEFT JOIN photo_stats p ON p.media_id = m.id "
            f"WHERE m.media_type = 'photo' AND {w}", (val,)).fetchone()[0]
    w, val, order = _quality_where(kind, lim)
    rows = conn.execute(
        f"SELECT m.id, m.width, m.height, p.blur, p.brightness FROM media m LEFT JOIN photo_stats p ON p.media_id = m.id "
        f"WHERE m.media_type = 'photo' AND {w} ORDER BY {order} LIMIT 60 OFFSET ?", (val, offset)).fetchall()
    left = conn.execute("SELECT COUNT(*) FROM media m LEFT JOIN photo_stats p ON p.media_id = m.id "
                        "WHERE m.media_type = 'photo' AND p.media_id IS NULL").fetchone()[0]
    conn.close()

    def note(r):
        if kind == "blur":
            return f"sharpness {r['blur']:.0f}"
        if kind in ("dark", "bright"):
            return f"brightness {r['brightness']:.0f}"
        return f"{r['width']}x{r['height']}"
    return jsonify({"items": [{"id": r["id"], "note": note(r)} for r in rows],
                    "total": counts[kind], "counts": counts, "kinds": QUAL_KINDS, "unanalyzed": left})


# ---------------------------------------------------------------------------
# Trash routes
# ---------------------------------------------------------------------------
@app.route("/cleanup/trash")
def cleanup_trash():
    offset = max(request.args.get("offset", type=int) or 0, 0)
    conn = get_db()
    _auto_purge(conn)
    count, size = conn.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM trash").fetchone()
    days = _trash_days(conn)
    rows = conn.execute("SELECT * FROM trash ORDER BY id DESC LIMIT 60 OFFSET ?", (offset,)).fetchall()
    conn.close()
    return jsonify({
        "count": count, "size": _human_size(size), "days": days,
        "items": [{"id": t["id"], "media_id": t["media_id"], "filename": t["filename"], "type": t["media_type"],
                   "folder": os.path.dirname(t["relpath"] or "") or "(library root)", "size": _human_size(t["size_bytes"] or 0),
                   "deleted": t["deleted_at"], "left": _days_left(days, t["deleted_at"])} for t in rows]})


@app.route("/trash/thumb/<int:tid>")
def trash_thumb(tid):
    conn = get_db()
    t = conn.execute("SELECT media_id FROM trash WHERE id = ?", (tid,)).fetchone()
    conn.close()
    if not t or not t["media_id"]:
        abort(404)
    p = thumbnail_path(t["media_id"])
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


def _trash_ids_from(data, conn):
    if data.get("all"):
        return [r["id"] for r in conn.execute("SELECT id FROM trash")]
    return [int(i) for i in data.get("ids", []) if str(i).isdigit()]


@app.route("/cleanup/trash/restore", methods=["POST"])
def cleanup_trash_restore():
    data = request.get_json(silent=True) or {}
    conn = get_db()
    ok = failed = 0
    new_ids = []
    last = ""
    for tid in _trash_ids_from(data, conn):
        good, res = _restore_trash(conn, tid)
        if good:
            ok += 1
            new_ids.append(res)
        else:
            failed += 1
            last = res
    conn.commit()
    conn.close()
    if new_ids:
        threading.Thread(target=lambda: [ensure_thumbnail_safe(i) for i in new_ids], daemon=True).start()
    return jsonify({"restored": ok, "failed": failed, "error": last if failed and not ok else ""})


def ensure_thumbnail_safe(mid):
    try:
        conn = get_db()
        r = conn.execute("SELECT relpath, media_type FROM media WHERE id = ?", (mid,)).fetchone()
        conn.close()
        if r:
            ensure_thumbnail(mid, r["relpath"], r["media_type"])
    except Exception:  # noqa: BLE001
        pass


@app.route("/cleanup/trash/purge", methods=["POST"])
def cleanup_trash_purge():
    data = request.get_json(silent=True) or {}
    conn = get_db()
    freed = n = 0
    for tid in _trash_ids_from(data, conn):
        t = conn.execute("SELECT * FROM trash WHERE id = ?", (tid,)).fetchone()
        if t:
            freed += t["size_bytes"] or 0
            _purge_trash_row(conn, t)
            n += 1
    conn.commit()
    conn.close()
    return jsonify({"purged": n, "freed": _human_size(freed)})


@app.route("/cleanup/trash/days", methods=["POST"])
def cleanup_trash_days():
    days = (request.get_json(silent=True) or {}).get("days")
    if days not in (0, 7, 30, 90):
        return jsonify({"error": "Pick 7, 30, 90 or 0 (never)."}), 400
    conn = get_db()
    _set_setting(conn, "trash_days", days)
    conn.commit()
    conn.close()
    return jsonify({"days": days})


# ---------------------------------------------------------------------------
# Rescan log page, plus optional automatic rescans (for the backup bot)
# ---------------------------------------------------------------------------
@app.route("/rescans")
def rescans_page():
    return render_template("rescans.html")


@app.route("/rescans/data")
def rescans_data():
    conn = get_db()
    log = []
    for r in conn.execute("SELECT * FROM scan_log ORDER BY id DESC LIMIT 60"):
        try:
            d = json.loads(r["detail"] or "{}")
        except ValueError:
            d = {}
        log.append({"id": r["id"], "at": r["at"], "source": r["source"], "imported": r["imported"],
                    "removed": r["removed"], "seconds": r["seconds"], "channels": d.get("channels", []),
                    "unassigned": d.get("unassigned", 0), "sample_ids": d.get("sample_ids", [])})
    recent = [{"id": r["id"], "filename": r["filename"], "type": r["media_type"], "added": r["added_at"],
               "channel": r["channel"], "category": r["category"]}
              for r in conn.execute(
                  "SELECT m.id, m.filename, m.media_type, m.added_at, "
                  "(SELECT a.name FROM media_albums ma JOIN albums a ON a.id = ma.album_id WHERE ma.media_id = m.id LIMIT 1) AS channel, "
                  "(SELECT c.name FROM media_albums ma JOIN albums a ON a.id = ma.album_id JOIN categories c ON c.id = a.category_id "
                  " WHERE ma.media_id = m.id LIMIT 1) AS category "
                  "FROM media m ORDER BY m.added_at DESC, m.id DESC LIMIT 60")]
    try:
        mins = int(_setting(conn, "autoscan_min", "0") or 0)
    except ValueError:
        mins = 0
    conn.close()
    return jsonify({"log": log, "recent": recent, "autoscan_min": mins})


@app.route("/rescans/auto", methods=["POST"])
def rescans_auto():
    mins = (request.get_json(silent=True) or {}).get("minutes")
    if mins not in (0, 5, 15, 30, 60):
        return jsonify({"error": "Pick Off, 5, 15, 30 or 60 minutes."}), 400
    conn = get_db()
    _set_setting(conn, "autoscan_min", mins)
    conn.commit()
    conn.close()
    return jsonify({"minutes": mins})


def _background_loop():
    """Every 30 s: run an automatic rescan if it is due (off by default) and
    tidy the Trash about once an hour. Only runs when started from app.py."""
    last_scan = time.time()
    last_purge = 0.0
    while True:
        time.sleep(30)
        try:
            conn = get_db()
            try:
                mins = int(_setting(conn, "autoscan_min", "0") or 0)
            except ValueError:
                mins = 0
            if time.time() - last_purge > 3600:
                _auto_purge(conn)
                last_purge = time.time()
            conn.close()
            if mins > 0 and time.time() - last_scan >= mins * 60:
                scan_library(source="auto", wait=False)
                last_scan = time.time()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Dashboard "tap to reveal" pairs (posts that come with two images, written by
# the backup bot as pairs.json next to the images)
# ---------------------------------------------------------------------------
PAIR_CHANNEL = os.environ.get("POSTMARK_PAIR_CHANNEL", "thegirlthegape")


def _find_pairs_file(conn, album_id):
    root = library_root()
    counts = {}
    for r in conn.execute("SELECT m.relpath FROM media m JOIN media_albums ma ON ma.media_id = m.id WHERE ma.album_id = ?", (album_id,)):
        d = os.path.dirname(r["relpath"])
        counts[d] = counts.get(d, 0) + 1
    for d, _n in sorted(counts.items(), key=lambda kv: -kv[1]):
        p = os.path.join(root, d, "pairs.json")
        if os.path.isfile(p):
            return p, d
    for base, dirs, files in os.walk(root):
        if os.path.relpath(base, root).count(os.sep) >= 2:
            dirs[:] = []
        if os.path.basename(base).lower() == PAIR_CHANNEL.lower() and "pairs.json" in files:
            return os.path.join(base, "pairs.json"), os.path.relpath(base, root)
    return None, None


def _load_pairs_file(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    try:
        data = json.loads(txt)
    except ValueError:
        data = []
        for line in txt.splitlines():   # tolerate one JSON object per line
            try:
                data.append(json.loads(line))
            except ValueError:
                pass
    if isinstance(data, dict):
        data = data.get("pairs") or list(data.values())
    return [d for d in data if isinstance(d, dict) and isinstance(d.get("images"), list)]


@app.route("/api/pairs")
def api_pairs():
    conn = get_db()
    al = conn.execute("SELECT id, name FROM albums WHERE lower(name) = lower(?)", (PAIR_CHANNEL,)).fetchone()
    if not al:
        conn.close()
        return jsonify({"status": "no_channel", "channel": PAIR_CHANNEL})
    path, folder = _find_pairs_file(conn, al["id"])
    if not path:
        conn.close()
        return jsonify({"status": "no_file", "channel": al["name"]})
    try:
        entries = _load_pairs_file(path)
    except OSError as e:
        conn.close()
        return jsonify({"status": "unreadable", "channel": al["name"], "error": str(e)})
    prefix = (folder + os.sep) if folder else ""
    by_name = {}
    for r in conn.execute("SELECT id, relpath FROM media WHERE substr(relpath, 1, ?) = ?", (len(prefix), prefix)):
        if os.path.dirname(r["relpath"]) == folder:
            by_name[os.path.basename(r["relpath"])] = r["id"]
    conn.close()
    if all(isinstance(e.get("time"), str) and e.get("time") for e in entries):
        entries = sorted(entries, key=lambda e: e["time"], reverse=True)
    else:
        entries = list(reversed(entries))
    pairs, pending = [], 0
    for e in entries:
        ids = [by_name.get(os.path.basename(str(n))) for n in e["images"]]
        ids = [i for i in ids if i]
        if not ids:
            pending += 1      # in pairs.json but not scanned yet
            continue
        pairs.append({"id": e.get("id"), "time": e.get("time"), "images": ids})
    return jsonify({"status": "ok", "channel": al["name"], "album_id": al["id"], "total": len(pairs),
                    "pending": pending, "pairs": pairs[:300]})


@app.route("/scan", methods=["POST"])
def rescan():
    res = scan_library(source="manual")
    imported, removed = res
    return jsonify({"imported": imported, "removed": removed})


import autosync

_SYNC = dict(
    dest_dir=lambda conn, aid, name: _album_dest_dir(
        conn, library_root(), aid, name, _album_category_name(conn, aid)),
    after_new=lambda ids, aid: (look.index_async(ids), faces.index_async(ids)),
)


@app.route("/sync/instagram", methods=["POST"])
def sync_instagram():
    threading.Thread(target=autosync.run_once, args=(_SYNC,), daemon=True).start()
    return jsonify(autosync.STATE)


@app.route("/sync/instagram/status")
def sync_instagram_status():
    return jsonify(autosync.STATE)


if __name__ == "__main__":
    init_db()
    scan_library(source="startup")
    threading.Thread(target=_background_loop, daemon=True).start()
    autosync.start_loop(_SYNC)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
