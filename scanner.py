import os
import json
import time
import threading
import subprocess
import shutil
from PIL import Image, ExifTags

from db import get_db, now_iso

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

# Categories whose channels should show up directly (uncategorized) on the
# website. The files stay on disk in Category/Channel/ so the backup bot keeps
# working; only the website hides the category. Comma-separated, case-insensitive.
# Override with the POSTMARK_FLATTEN environment variable.
FLATTEN_CATEGORIES = {
    n.strip().lower()
    for n in os.environ.get("POSTMARK_FLATTEN", "tgtg").split(",")
    if n.strip()
}

SCAN_LOCK = threading.Lock()
LAST_SCAN = {"new_ids": [], "imported": 0, "removed": 0}

THUMB_SIZE = (400, 400)
# Bigger, higher-quality render used for the collection spotlight and media
# viewer, where a 400px grid thumbnail looks visibly soft.
PREVIEW_SIZE = (1080, 1080)


def library_root():
    """Where your actual media lives. Defaults to the bundled library/ folder,
    but set POSTMARK_LIBRARY to point at an existing folder instead (e.g. your
    Discord downloads, or ~/storage/dcim/Camera after termux-setup-storage) so
    files are read in place rather than duplicated."""
    custom = os.environ.get("POSTMARK_LIBRARY")
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")


def data_root():
    """Where the database, thumbnails, and previews are stored. Defaults to
    the app's own .data/ folder (Termux's internal/private storage), which
    can fill up fast — a 400x400 thumb + 1080x1080 preview per photo/video
    adds up to roughly 100-150KB per item. Set POSTMARK_DATA to redirect all
    of that onto your SD card instead (e.g. alongside POSTMARK_LIBRARY), so
    internal storage only holds the app itself."""
    custom = os.environ.get("POSTMARK_DATA")
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")


def thumbs_root():
    return os.path.join(data_root(), "thumbs")


def previews_root():
    return os.path.join(data_root(), "previews")


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _read_photo_taken_at(path):
    """Best-effort EXIF capture date; falls back to None if missing/unreadable."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}
            date_tag = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
            if date_tag and date_tag in exif:
                raw = exif[date_tag]
                # EXIF format: "YYYY:MM:DD HH:MM:SS"
                return raw.replace(":", "-", 2)
    except Exception:
        pass
    return None


def _read_photo_exif(path):
    """Best-effort camera/lens/exposure + GPS metadata. Returns a dict with
    only the keys actually found — missing ones are simply absent, and the
    caller treats every key as optional."""
    info = {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return info
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}

            def top(name):
                tag = tag_map.get(name)
                return exif.get(tag) if tag else None

            make, model = top("Make"), top("Model")
            if make:
                info["camera_make"] = str(make).strip().rstrip("\x00") or None
            if model:
                info["camera_model"] = str(model).strip().rstrip("\x00") or None

            # Lens/aperture/ISO/shutter/focal length live in the EXIF sub-IFD,
            # not the top-level tags.
            try:
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            except Exception:
                exif_ifd = {}

            def sub(name):
                tag = tag_map.get(name)
                return exif_ifd.get(tag) if tag else None

            lens = sub("LensModel")
            if lens:
                info["lens"] = str(lens).strip().rstrip("\x00") or None

            fnumber = sub("FNumber")
            if fnumber:
                try:
                    info["aperture"] = round(float(fnumber), 1)
                except Exception:
                    pass

            iso = sub("ISOSpeedRatings") or sub("PhotographicSensitivity")
            if iso:
                try:
                    info["iso"] = int(iso)
                except Exception:
                    pass

            exposure = sub("ExposureTime")
            if exposure:
                try:
                    exposure = float(exposure)
                    info["shutter_speed"] = f"1/{round(1 / exposure)}" if exposure < 1 else f"{exposure:g}s"
                except Exception:
                    pass

            focal = sub("FocalLength")
            if focal:
                try:
                    info["focal_length"] = f"{round(float(focal))}mm"
                except Exception:
                    pass

            # GPS lives in its own IFD, with its own tag names.
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except Exception:
                gps_ifd = {}
            if gps_ifd:
                gps_tag_map = {v: k for k, v in ExifTags.GPSTAGS.items()}

                def gps(name):
                    tag = gps_tag_map.get(name)
                    return gps_ifd.get(tag) if tag else None

                lat, lat_ref = gps("GPSLatitude"), gps("GPSLatitudeRef")
                lon, lon_ref = gps("GPSLongitude"), gps("GPSLongitudeRef")
                if lat and lon:
                    def to_deg(v):
                        d, m, s = v
                        return float(d) + float(m) / 60 + float(s) / 3600
                    try:
                        lat_deg, lon_deg = to_deg(lat), to_deg(lon)
                        if lat_ref == "S":
                            lat_deg = -lat_deg
                        if lon_ref == "W":
                            lon_deg = -lon_deg
                        info["gps_lat"] = round(lat_deg, 6)
                        info["gps_lon"] = round(lon_deg, 6)
                    except Exception:
                        pass
    except Exception:
        pass
    return info


def _photo_dimensions(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (None, None)


def _get_or_create_category(conn, name):
    row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO categories (name, created_at) VALUES (?, ?)",
        (name, now_iso()),
    )
    return cur.lastrowid


def _get_or_create_album(conn, name, category_name, category_id):
    """Finds or creates the channel (album) named `name`. If a channel with
    that exact name already exists under a *different* category, it's
    disambiguated by category rather than silently merging two unrelated
    on-disk folders into one channel (album names are globally unique)."""
    row = conn.execute("SELECT id, category_id FROM albums WHERE name = ?", (name,)).fetchone()
    if row is None:
        cur = conn.execute(
            "INSERT INTO albums (name, created_at, category_id) VALUES (?, ?, ?)",
            (name, now_iso(), category_id),
        )
        return cur.lastrowid
    if row["category_id"] is None and category_id is not None:
        conn.execute("UPDATE albums SET category_id = ? WHERE id = ?", (category_id, row["id"]))
        return row["id"]
    if category_id is not None and row["category_id"] not in (None, category_id):
        return _get_or_create_album(conn, f"{name} ({category_name})", category_name, category_id)
    return row["id"]


def _assign_channels(conn, rows):
    """Derives category/channel from each media row's on-disk folder path
    (top-level folder = category, next level down = channel — matching a
    Category/Channel/file.ext layout) and links it into that channel,
    creating the category/channel records as needed. A file with only one
    folder above it (Channel/file.ext) becomes an uncategorized channel. A
    file with no folder above it (loose in the library root) is left
    unassigned — there's no channel name to derive."""
    changed = False
    for row in rows:
        dir_parts = row["relpath"].split(os.sep)[:-1]
        if not dir_parts:
            continue
        if len(dir_parts) == 1:
            if dir_parts[0].lower() in FLATTEN_CATEGORIES:
                continue  # loose file directly in a hidden category: no channel name
            category_name, channel_name = None, dir_parts[0]
        else:
            category_name, channel_name = dir_parts[0], dir_parts[1]
            if category_name.lower() in FLATTEN_CATEGORIES:
                category_name = None  # show the channel directly, hide the category

        category_id = _get_or_create_category(conn, category_name) if category_name else None
        album_id = _get_or_create_album(conn, channel_name, category_name, category_id)
        conn.execute(
            "INSERT OR IGNORE INTO media_albums (media_id, album_id) VALUES (?, ?)",
            (row["id"], album_id),
        )
        changed = True
    if changed:
        conn.commit()


def _prune_missing(conn, root, thumbnail_path, preview_path):
    """Deleting a file straight from the file manager (off the SD card, say)
    doesn't tell Postmark anything — the DB row is left behind forever,
    showing up as a broken '?' tile. This checks every known file against
    disk and removes anything that's actually gone, plus its cached
    thumbnail/preview. Returns how many were removed."""
    rows = conn.execute("SELECT id, relpath FROM media").fetchall()
    missing_ids = [
        row["id"] for row in rows
        if not os.path.exists(os.path.join(root, row["relpath"]))
    ]
    if not missing_ids:
        return 0
    conn.executemany("DELETE FROM media WHERE id = ?", [(mid,) for mid in missing_ids])
    conn.commit()
    for mid in missing_ids:
        for path in (thumbnail_path(mid), preview_path(mid)):
            try:
                os.remove(path)
            except OSError:
                pass
    return len(missing_ids)


def flatten_categories(conn):
    """One-time-safe cleanup: channels that were already filed under a hidden
    category become uncategorized and the empty category is removed."""
    changed = False
    for cat in conn.execute("SELECT id, name FROM categories").fetchall():
        if cat["name"].lower() in FLATTEN_CATEGORIES:
            conn.execute("UPDATE albums SET category_id = NULL WHERE category_id = ?", (cat["id"],))
            conn.execute("DELETE FROM categories WHERE id = ?", (cat["id"],))
            changed = True
    if changed:
        conn.commit()


def _scan_library_impl(source="manual"):
    """Walks the library folder for new files not yet in the DB, imports them,
    generates thumbnails, prunes DB rows whose file was deleted outside the
    app, and assigns each surviving one to a category/channel derived from
    its on-disk folder path. Returns (imported_count, removed_count)."""
    started = time.time()
    root = library_root()
    os.makedirs(root, exist_ok=True)
    os.makedirs(thumbs_root(), exist_ok=True)
    os.makedirs(previews_root(), exist_ok=True)

    conn = get_db()
    flatten_categories(conn)
    existing = {row["relpath"] for row in conn.execute("SELECT relpath FROM media")}

    removed = _prune_missing(conn, root, thumbnail_path, preview_path)

    imported = 0
    new_media = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("._"):
                continue  # macOS AppleDouble sidecar file, not real media
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            ext = os.path.splitext(fname)[1].lower()

            if rel_path in existing:
                continue
            if ext in PHOTO_EXTS:
                media_type = "photo"
            elif ext in VIDEO_EXTS:
                media_type = "video"
            else:
                continue

            width = height = None
            taken_at = None
            exif_info = {}
            if media_type == "photo":
                width, height = _photo_dimensions(full_path)
                taken_at = _read_photo_taken_at(full_path)
                exif_info = _read_photo_exif(full_path)

            try:
                size_bytes = os.path.getsize(full_path)
            except OSError:
                size_bytes = None

            cur = conn.execute(
                "INSERT INTO media (filename, relpath, media_type, width, height, size_bytes, added_at, taken_at, "
                "camera_make, camera_model, lens, aperture, iso, shutter_speed, focal_length, gps_lat, gps_lon, exif_checked) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    fname, rel_path, media_type, width, height, size_bytes, now_iso(), taken_at,
                    exif_info.get("camera_make"), exif_info.get("camera_model"), exif_info.get("lens"),
                    exif_info.get("aperture"), exif_info.get("iso"), exif_info.get("shutter_speed"),
                    exif_info.get("focal_length"), exif_info.get("gps_lat"), exif_info.get("gps_lon"),
                ),
            )
            new_media.append((cur.lastrowid, rel_path, media_type))
            imported += 1

    conn.commit()

    # Remove any AppleDouble sidecar rows imported before the skip above
    # existed — these were already in the DB, so the skip alone doesn't
    # touch them, and ffmpeg will keep erroring on them every scan otherwise.
    ghost_rows = conn.execute(
        "SELECT id FROM media WHERE filename LIKE '._%'"
    ).fetchall()
    if ghost_rows:
        ghost_ids = [row["id"] for row in ghost_rows]
        conn.executemany(
            "DELETE FROM media WHERE id = ?", [(gid,) for gid in ghost_ids]
        )
        conn.commit()
        for gid in ghost_ids:
            for path in (thumbnail_path(gid), preview_path(gid)):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Backfill size_bytes for rows imported before this field existed.
    missing_size = conn.execute(
        "SELECT id, relpath FROM media WHERE size_bytes IS NULL"
    ).fetchall()
    for row in missing_size:
        full_path = os.path.join(root, row["relpath"])
        try:
            size_bytes = os.path.getsize(full_path)
            conn.execute("UPDATE media SET size_bytes = ? WHERE id = ?", (size_bytes, row["id"]))
        except OSError:
            pass
    if missing_size:
        conn.commit()

    # Backfill camera/lens/exposure/GPS metadata for photos imported before
    # this existed — a one-time cost per photo. exif_checked flips to 1
    # whether or not anything was actually found, so a photo with no EXIF
    # data isn't reopened and re-attempted on every future scan.
    unchecked = conn.execute(
        "SELECT id, relpath FROM media WHERE media_type = 'photo' AND exif_checked = 0"
    ).fetchall()
    for row in unchecked:
        full_path = os.path.join(root, row["relpath"])
        info = _read_photo_exif(full_path)
        conn.execute(
            """UPDATE media SET camera_make=?, camera_model=?, lens=?, aperture=?, iso=?,
               shutter_speed=?, focal_length=?, gps_lat=?, gps_lon=?, exif_checked=1 WHERE id=?""",
            (
                info.get("camera_make"), info.get("camera_model"), info.get("lens"),
                info.get("aperture"), info.get("iso"), info.get("shutter_speed"),
                info.get("focal_length"), info.get("gps_lat"), info.get("gps_lon"), row["id"],
            ),
        )
    if unchecked:
        conn.commit()

    # Assign any media not yet in a channel (new imports, plus anything
    # scanned before this existed) into a category/channel derived from its
    # folder path.
    unassigned = conn.execute(
        """
        SELECT m.id, m.relpath FROM media m
        LEFT JOIN media_albums ma ON ma.media_id = m.id
        WHERE ma.media_id IS NULL
        """
    ).fetchall()
    _assign_channels(conn, unassigned)

    # Pre-warm thumbnails only for what's genuinely new this scan. Every
    # other file already gets its thumbnail generated on first view via
    # /thumb (see app.py) — looping over the whole library here on every
    # scan/startup was pure duplicate work and the main cause of slow
    # restarts as the library grew.
    for media_id, rel_path, media_type in new_media:
        ensure_thumbnail(media_id, rel_path, media_type)

    new_ids = [m[0] for m in new_media]
    LAST_SCAN.update({"new_ids": new_ids, "imported": imported, "removed": removed})
    try:
        _write_scan_log(conn, source, imported, removed, time.time() - started, new_ids)
    except Exception:
        pass  # logging must never break a scan

    conn.close()
    return imported, removed


def _write_scan_log(conn, source, imported, removed, seconds, new_ids):
    """Records one scan: when, what triggered it, how many files, and which
    channels the new files landed in. Empty automatic scans are not logged."""
    if not (imported or removed or source == "manual"):
        return
    per = {}
    unassigned = 0
    for i in range(0, len(new_ids), 400):
        chunk = new_ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT m.id, a.name AS channel, c.name AS category FROM media m "
            f"LEFT JOIN media_albums ma ON ma.media_id = m.id "
            f"LEFT JOIN albums a ON a.id = ma.album_id "
            f"LEFT JOIN categories c ON c.id = a.category_id WHERE m.id IN ({marks})", chunk).fetchall()
        for r in rows:
            if r["channel"] is None:
                unassigned += 1
            else:
                k = (r["channel"], r["category"] or "")
                per[k] = per.get(k, 0) + 1
    detail = {
        "channels": [{"name": k[0], "category": k[1], "count": v}
                     for k, v in sorted(per.items(), key=lambda kv: -kv[1])],
        "unassigned": unassigned,
        "sample_ids": new_ids[-12:],
    }
    conn.execute(
        "INSERT INTO scan_log (at, source, imported, removed, seconds, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), source, imported, removed, round(seconds, 1), json.dumps(detail)),
    )
    conn.execute("DELETE FROM scan_log WHERE id NOT IN (SELECT id FROM scan_log ORDER BY id DESC LIMIT 300)")
    conn.commit()


def scan_library(source="manual", wait=True):
    """Scans the library (one scan at a time). source is logged: manual, auto,
    upload or startup. With wait=False a scan already running is not queued:
    returns None instead."""
    if not SCAN_LOCK.acquire(blocking=wait):
        return None
    try:
        return _scan_library_impl(source)
    finally:
        SCAN_LOCK.release()


def thumbnail_path(media_id):
    return os.path.join(thumbs_root(), f"{media_id}.jpg")


def ensure_thumbnail(media_id, relpath, media_type):
    out_path = thumbnail_path(media_id)
    if os.path.exists(out_path):
        return out_path

    src_path = os.path.join(library_root(), relpath)

    try:
        if media_type == "photo":
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE)
                img.save(out_path, "JPEG", quality=85)
        elif media_type == "video" and ffmpeg_available():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", src_path,
                    "-vf", "thumbnail,scale=400:-1",
                    "-frames:v", "1",
                    out_path,
                ],
                timeout=30,
                check=False,
            )
    except Exception:
        pass

    return out_path if os.path.exists(out_path) else None


def preview_path(media_id):
    return os.path.join(previews_root(), f"{media_id}.jpg")


def ensure_preview(media_id, relpath, media_type):
    """Larger, higher-quality render for the collection spotlight and the
    media viewer — the 400px grid thumbnail looks noticeably soft blown up."""
    out_path = preview_path(media_id)
    if os.path.exists(out_path):
        return out_path

    src_path = os.path.join(library_root(), relpath)

    try:
        if media_type == "photo":
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img.thumbnail(PREVIEW_SIZE)
                img.save(out_path, "JPEG", quality=92)
        elif media_type == "video" and ffmpeg_available():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", src_path,
                    "-vf", "thumbnail,scale=1080:-1",
                    "-frames:v", "1",
                    out_path,
                ],
                timeout=30,
                check=False,
            )
    except Exception:
        pass

    return out_path if os.path.exists(out_path) else None
