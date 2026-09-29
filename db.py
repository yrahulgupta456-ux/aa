import sqlite3
import os
from datetime import datetime, timezone


def _data_root():
    """Same POSTMARK_DATA variable scanner.py uses, kept independent here to
    avoid a circular import (scanner.py imports from this module)."""
    custom = os.environ.get("POSTMARK_DATA")
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")


DB_PATH = os.path.join(_data_root(), "postmark.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _column_names(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn):
    """Adds columns introduced after the initial release, without touching
    existing data. Safe to run every startup."""
    media_cols = _column_names(conn, "media")
    if "size_bytes" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN size_bytes INTEGER")

    album_cols = _column_names(conn, "albums")
    if "link" not in album_cols:
        conn.execute("ALTER TABLE albums ADD COLUMN link TEXT")
    if "category_id" not in album_cols:
        conn.execute("ALTER TABLE albums ADD COLUMN category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL")

    # Camera/lens/exposure + GPS metadata, best-effort read from each photo's
    # EXIF at scan time. exif_checked lets the scanner attempt this exactly
    # once per photo (a photo with no EXIF data isn't reopened every scan).
    if "camera_make" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN camera_make TEXT")
    if "camera_model" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN camera_model TEXT")
    if "lens" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN lens TEXT")
    if "aperture" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN aperture REAL")
    if "iso" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN iso INTEGER")
    if "shutter_speed" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN shutter_speed TEXT")
    if "focal_length" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN focal_length TEXT")
    if "gps_lat" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN gps_lat REAL")
    if "gps_lon" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN gps_lon REAL")
    if "exif_checked" not in media_cols:
        conn.execute("ALTER TABLE media ADD COLUMN exif_checked INTEGER NOT NULL DEFAULT 0")

    # Cumulative watch time and items shown per channel, added after the
    # leaderboard first shipped. Sessions recorded before this migration
    # keep their points but have 0 for these two.
    feed_cols = _column_names(conn, "feed_stats")
    if "seconds_fed" not in feed_cols:
        conn.execute("ALTER TABLE feed_stats ADD COLUMN seconds_fed REAL NOT NULL DEFAULT 0")
    if "items_shown" not in feed_cols:
        conn.execute("ALTER TABLE feed_stats ADD COLUMN items_shown INTEGER NOT NULL DEFAULT 0")
    if "last_media_id" not in feed_cols:
        # The item on screen when the channel was last fed (leaderboard card).
        conn.execute("ALTER TABLE feed_stats ADD COLUMN last_media_id INTEGER")

    conn.commit()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            relpath TEXT NOT NULL UNIQUE,
            media_type TEXT NOT NULL CHECK(media_type IN ('photo', 'video')),
            width INTEGER,
            height INTEGER,
            size_bytes INTEGER,
            added_at TEXT NOT NULL,
            taken_at TEXT,
            favorite INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            cover_media_id INTEGER REFERENCES media(id) ON DELETE SET NULL,
            link TEXT,
            category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS media_albums (
            media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            album_id INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
            PRIMARY KEY (media_id, album_id)
        );

        -- Cumulative "feed leaderboard" points per channel (collection), carried
        -- over forever. Mirrors the old Discord bot's all-time vote points: a
        -- session only counts once it runs at least MIN_FEED_DURATION_FOR_VOTE
        -- seconds, then scores base + per-minute + per-item.
        CREATE TABLE IF NOT EXISTS feed_stats (
            album_id INTEGER PRIMARY KEY REFERENCES albums(id) ON DELETE CASCADE,
            points REAL NOT NULL DEFAULT 0,
            sessions INTEGER NOT NULL DEFAULT 0,
            last_fed_at TEXT,
            seconds_fed REAL NOT NULL DEFAULT 0,
            items_shown INTEGER NOT NULL DEFAULT 0,
            last_media_id INTEGER
        );

        -- Trash: deleted items wait here (file kept in .data/trash) until restored
        -- or purged. row_json holds the old media row so a restore is exact.
        CREATE TABLE IF NOT EXISTS trash (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER,
            filename TEXT,
            relpath TEXT,
            trash_path TEXT,
            size_bytes INTEGER,
            media_type TEXT,
            row_json TEXT,
            album_ids TEXT,
            deleted_at TEXT NOT NULL
        );

        -- One line per library scan: what it found and which channels got it.
        CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at TEXT NOT NULL,
            source TEXT,
            imported INTEGER NOT NULL DEFAULT 0,
            removed INTEGER NOT NULL DEFAULT 0,
            seconds REAL,
            detail TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        -- Per-photo fingerprint (for near-duplicates) and quality numbers.
        CREATE TABLE IF NOT EXISTS photo_stats (
            media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
            dhash TEXT,
            blur REAL,
            brightness REAL,
            checked_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type);
        CREATE INDEX IF NOT EXISTS idx_media_favorite ON media(favorite);
        CREATE INDEX IF NOT EXISTS idx_albums_category ON albums(category_id);
        """
    )
    conn.commit()
    _migrate(conn)
    conn.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat()
