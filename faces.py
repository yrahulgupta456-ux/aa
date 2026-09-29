"""Face matching for Postmark: suggests which channel a new photo belongs to.

Everything runs on this device. It uses OpenCV's built-in face detector
(YuNet) and face recognizer (SFace), which need two small model files that
are downloaded once into .data/models/.

This module must import cleanly even when OpenCV or numpy are missing, so
the rest of Postmark keeps working; engine_status() reports what is missing.

How matching works: each channel is described by the faces found in its own
photos (the largest face in each photo). A new photo's faces are compared
with all of those, and a channel's score is the average of its 3 closest
faces (cosine similarity, 0 to 1 for real matches).
"""
import os
import threading
import time
import urllib.request

try:
    import numpy as np
except Exception:  # numpy not installed
    np = None

from db import get_db, now_iso
from scanner import data_root, library_root, thumbnail_path, preview_path

MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}
MIN_MODEL_BYTES = 100_000  # a Git-LFS pointer file or an error page is far smaller

# Detection
DET_SCORE = 0.85       # how sure the detector must be that it is a face
MIN_FACE_PX = 48       # ignore faces narrower than this (after downscaling)
MAX_SIDE = 960         # photos are shrunk to this before detection
MAX_QUERY_FACES = 3    # faces looked at per new photo

# Matching (starting points, tune with real data)
TOPK = 3               # a channel's score = mean of its closest TOPK faces
STRONG_MIN = 0.50      # "Strong match"
LIKELY_MIN = 0.38      # "Likely" (SFace's own same-person cut-off is 0.363)
WEAK_MIN = 0.28        # below this a channel is not even suggested
FEW_SAMPLES_PENALTY = 0.03  # per missing sample when a channel has < TOPK faces

_ENGINE = None
_ENGINE_LOCK = threading.RLock()


# ---------------------------------------------------------------- models

def models_dir():
    return os.path.join(data_root(), "models")


def model_path(key):
    return os.path.join(models_dir(), MODELS[key][0])


def models_present():
    for key in MODELS:
        p = model_path(key)
        if not (os.path.exists(p) and os.path.getsize(p) >= MIN_MODEL_BYTES):
            return False
    return True


DOWNLOAD = {"running": False, "error": "", "done": False}


def _download_job():
    try:
        os.makedirs(models_dir(), exist_ok=True)
        for key in MODELS:
            name, url = MODELS[key]
            dest = model_path(key)
            if os.path.exists(dest) and os.path.getsize(dest) >= MIN_MODEL_BYTES:
                continue
            part = dest + ".part"
            with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
            if os.path.getsize(part) < MIN_MODEL_BYTES:
                os.remove(part)
                raise RuntimeError(f"{name} downloaded but looks incomplete.")
            os.replace(part, dest)
        DOWNLOAD["done"] = True
    except Exception as e:  # network down, blocked, disk full...
        DOWNLOAD["error"] = f"Download failed: {e}"
    finally:
        DOWNLOAD["running"] = False


def start_download():
    if DOWNLOAD["running"]:
        return False
    DOWNLOAD.update({"running": True, "error": "", "done": False})
    threading.Thread(target=_download_job, daemon=True).start()
    return True


# ---------------------------------------------------------------- engine

class _Engine:
    """Real engine: OpenCV YuNet + SFace. faces(bgr) returns a list of
    (face_area, unit_vector), largest face first."""

    def __init__(self):
        import cv2
        self.det = cv2.FaceDetectorYN.create(model_path("yunet"), "", (320, 320), DET_SCORE, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(model_path("sface"), "")

    def faces(self, bgr):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, dets = self.det.detect(bgr)
        out = []
        if dets is None:
            return out
        for d in dets:
            if d[14] < DET_SCORE or d[2] < MIN_FACE_PX:
                continue
            aligned = self.rec.alignCrop(bgr, d)
            feat = np.array(self.rec.feature(aligned), dtype=np.float32).flatten()  # copy on purpose
            norm = float(np.linalg.norm(feat))
            if norm == 0:
                continue
            out.append((float(d[2] * d[3]), feat / norm))
        out.sort(key=lambda x: -x[0])
        return out


def set_engine(engine):
    """Swap in another engine (used by tests)."""
    global _ENGINE
    _ENGINE = engine


def engine_status():
    """{'ready': bool, 'reason': str} — reason says what to fix when not ready."""
    if _ENGINE is not None:
        return {"ready": True, "reason": "", "models": True}
    if np is None:
        return {"ready": False, "models": models_present(),
                "reason": "numpy is not installed. In Termux run: pkg install python-numpy"}
    try:
        import cv2
    except Exception:
        return {"ready": False, "models": models_present(),
                "reason": "OpenCV is not installed. In Termux run: pkg install python-opencv-python"}
    if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
        return {"ready": False, "models": models_present(),
                "reason": "This OpenCV is too old for face matching (needs 4.5.4 or newer)."}
    if not models_present():
        return {"ready": False, "models": False,
                "reason": "The two face model files (about 38 MB) have not been downloaded yet."}
    return {"ready": True, "reason": "", "models": True}


def _get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE, ""
        st = engine_status()
        if not st["ready"]:
            return None, st["reason"]
        try:
            _ENGINE = _Engine()
        except Exception as e:
            return None, f"Could not start the face engine: {e}"
        return _ENGINE, ""


# ---------------------------------------------------------------- storage

def ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
            vec BLOB,
            face_px REAL,
            checked_at TEXT
        )
        """
    )


def _image_to_bgr(im):
    from PIL import ImageOps
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def _load_bgr(media_id, relpath, media_type):
    """Decode a photo (or a video's still) to a downscaled BGR array."""
    from PIL import Image, ImageOps
    paths = [preview_path(media_id)]
    if media_type == "photo":
        paths.append(os.path.join(library_root(), relpath))
    paths.append(thumbnail_path(media_id))
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with Image.open(p) as im:
                try:
                    im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # fast JPEG downscale
                except Exception:
                    pass
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        except Exception:
            continue
    return None


def analyze(conn, media_id, relpath, media_type):
    """Detect faces in one item, remember the largest one, and return all of
    them as [(area, vec), ...]. Raises RuntimeError if the engine is not ready."""
    engine, why = _get_engine()
    if engine is None:
        raise RuntimeError(why)
    ensure_schema(conn)
    bgr = _load_bgr(media_id, relpath, media_type)
    found = []
    if bgr is not None:
        with _ENGINE_LOCK:
            try:
                found = engine.faces(bgr)
            except Exception:
                found = []
    if found:
        area, vec = found[0]
        conn.execute(
            "INSERT OR REPLACE INTO face_embeddings (media_id, vec, face_px, checked_at) VALUES (?, ?, ?, ?)",
            (media_id, np.asarray(vec, dtype=np.float32).tobytes(), area, now_iso()),
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO face_embeddings (media_id, vec, face_px, checked_at) VALUES (?, NULL, NULL, ?)",
            (media_id, now_iso()),
        )
    conn.commit()
    return found[:MAX_QUERY_FACES]


def identify_bytes(conn, data, top=5):
    """Test helper: who does the face in this picture look like? Nothing is
    saved. Returns {'faces': [{'area', 'candidates': [...]}]} or raises
    RuntimeError if the engine is not ready."""
    import io
    from PIL import Image
    engine, why = _get_engine()
    if engine is None:
        raise RuntimeError(why)
    with Image.open(io.BytesIO(data)) as im:
        bgr = _image_to_bgr(im)
    with _ENGINE_LOCK:
        found = engine.faces(bgr)[:MAX_QUERY_FACES]
    matrix = _load_matrix(conn)
    out = []
    for area, vec in found:
        out.append({"area": round(float(area)), "candidates": suggest(conn, [vec], top=top, matrix=matrix)})
    return {"faces": out}


def self_test(conn, samples=200):
    """Accuracy check on faces already learned: for a random sample of learned
    photos, hide the photo itself and ask which channel it would be sent to.
    Right = the top guess is a channel the photo is really in."""
    import random
    matrix = _load_matrix(conn)
    if matrix is None:
        return {"tested": 0}
    mat, mids, groups = matrix
    albums_of = {}
    for aid, s, e in groups:
        for mid in mids[s:e]:
            albums_of.setdefault(int(mid), set()).add(aid)
    row_of = {}
    for i, mid in enumerate(mids):
        row_of.setdefault(int(mid), i)
    # only channels with at least 2 learned faces can be judged fairly
    size = {aid: e - s for aid, s, e in groups}
    pool = [m for m, al in albums_of.items() if all(size[a] >= 2 for a in al)]
    random.shuffle(pool)
    pool = pool[:samples]
    res = {"tested": 0, "right": 0, "none": 0,
           "strong": {"n": 0, "right": 0}, "likely": {"n": 0, "right": 0}, "weak": {"n": 0, "right": 0}}
    for mid in pool:
        cands = suggest(conn, [mat[row_of[mid]]], exclude_media=mid, top=1, matrix=matrix)
        res["tested"] += 1
        if not cands:
            res["none"] += 1
            continue
        ok = cands[0]["album_id"] in albums_of[mid]
        res["right"] += ok
        b = res[cands[0]["tier"]]
        b["n"] += 1
        b["right"] += ok
    return res


# ---------------------------------------------------------------- matching

def _load_matrix(conn):
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT fe.media_id, fe.vec, ma.album_id FROM face_embeddings fe "
        "JOIN media_albums ma ON ma.media_id = fe.media_id WHERE fe.vec IS NOT NULL"
    ).fetchall()
    if not rows:
        return None
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    mids = np.array([r["media_id"] for r in rows])
    aids = np.array([r["album_id"] for r in rows])
    order = np.argsort(aids, kind="stable")
    mat, mids, aids = mat[order], mids[order], aids[order]
    bounds = np.flatnonzero(np.diff(aids)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(aids)]))
    groups = [(int(aids[s]), int(s), int(e)) for s, e in zip(starts, ends)]
    return mat, mids, groups


def tier(score):
    if score >= STRONG_MIN:
        return "strong"
    if score >= LIKELY_MIN:
        return "likely"
    return "weak"


def suggest(conn, query_vecs, exclude_media=None, top=3, matrix=None):
    """Best channels for the given face vectors: a list of
    {album_id, name, category, score, tier}, best first."""
    if not query_vecs:
        return []
    matrix = matrix or _load_matrix(conn)
    if matrix is None:
        return []
    mat, mids, groups = matrix
    best = {}
    for q in query_vecs:
        sims = mat @ np.asarray(q, dtype=np.float32)
        if exclude_media is not None:
            sims = np.where(mids == exclude_media, -1.0, sims)
        for aid, s, e in groups:
            part = sims[s:e]
            valid = part[part > -0.5]
            if valid.size == 0:
                continue
            k = min(TOPK, valid.size)
            score = float(np.partition(valid, valid.size - k)[valid.size - k:].mean())
            score -= FEW_SAMPLES_PENALTY * (TOPK - k)
            if score > best.get(aid, -1.0):
                best[aid] = score
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    ranked = [(a, s) for a, s in ranked if s >= WEAK_MIN][:top]
    out = []
    for aid, score in ranked:
        row = conn.execute(
            "SELECT a.name, c.name AS category FROM albums a "
            "LEFT JOIN categories c ON c.id = a.category_id WHERE a.id = ?",
            (aid,),
        ).fetchone()
        if not row:
            continue
        out.append({"album_id": aid, "name": row["name"], "category": row["category"] or "",
                    "score": round(score, 3), "tier": tier(score)})
    return out


# ---------------------------------------------------------------- learning jobs

INDEX = {"running": False, "processed": 0, "total": 0, "faces": 0, "error": "", "stop": False}


def _index_job(per_channel):
    conn = get_db()
    try:
        ensure_schema(conn)
        engine, why = _get_engine()
        if engine is None:
            INDEX["error"] = why
            return
        albums = [r["id"] for r in conn.execute("SELECT id FROM albums")]
        for _pass in range(4):
            have = {r["album_id"]: r["n"] for r in conn.execute(
                "SELECT ma.album_id, COUNT(*) AS n FROM face_embeddings fe "
                "JOIN media_albums ma ON ma.media_id = fe.media_id "
                "WHERE fe.vec IS NOT NULL GROUP BY ma.album_id")}
            todo, seen = [], set()
            for aid in albums:
                need = per_channel - have.get(aid, 0)
                if need <= 0:
                    continue
                rows = conn.execute(
                    "SELECT m.id, m.relpath, m.media_type FROM media m "
                    "JOIN media_albums ma ON ma.media_id = m.id "
                    "LEFT JOIN face_embeddings fe ON fe.media_id = m.id "
                    "WHERE ma.album_id = ? AND fe.media_id IS NULL "
                    "ORDER BY (m.media_type = 'photo') DESC, RANDOM() LIMIT ?",
                    (aid, need),
                ).fetchall()
                for r in rows:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        todo.append(r)
            if not todo:
                break
            INDEX["total"] += len(todo)
            for r in todo:
                if INDEX["stop"]:
                    return
                try:
                    found = analyze(conn, r["id"], r["relpath"], r["media_type"])
                    if found:
                        INDEX["faces"] += 1
                except Exception as e:
                    INDEX["error"] = str(e)
                    return
                INDEX["processed"] += 1
    finally:
        conn.close()
        INDEX["running"] = False


def start_index(per_channel=40):
    if INDEX["running"]:
        return False
    INDEX.update({"running": True, "processed": 0, "total": 0, "faces": 0, "error": "", "stop": False})
    threading.Thread(target=_index_job, args=(int(per_channel),), daemon=True).start()
    return True


def stop_index():
    INDEX["stop"] = True


def index_async(media_ids):
    """Learn the faces of freshly uploaded items in the background (used when
    they were uploaded straight into a channel). Silent if not set up."""
    if not media_ids or not engine_status()["ready"]:
        return

    def job():
        conn = get_db()
        try:
            for mid in media_ids:
                row = conn.execute("SELECT relpath, media_type FROM media WHERE id = ?", (mid,)).fetchone()
                if row:
                    try:
                        analyze(conn, mid, row["relpath"], row["media_type"])
                    except Exception:
                        return
        finally:
            conn.close()

    threading.Thread(target=job, daemon=True).start()


def stats(conn):
    ensure_schema(conn)
    faces = conn.execute("SELECT COUNT(*) FROM face_embeddings WHERE vec IS NOT NULL").fetchone()[0]
    checked = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
    channels = conn.execute(
        "SELECT COUNT(DISTINCT ma.album_id) FROM face_embeddings fe "
        "JOIN media_albums ma ON ma.media_id = fe.media_id WHERE fe.vec IS NOT NULL"
    ).fetchone()[0]
    total_channels = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
    return {"faces": faces, "checked": checked, "channels": channels, "total_channels": total_channels}
