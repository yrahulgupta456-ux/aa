"""Look: find photos by what is in them ("saree", "red dress", "beach").

Everything runs on this device. It uses CLIP (ViT-B/32, ONNX build by Xenova)
run with onnxruntime. Two things happen:

1. Indexing: every photo (and each video's still) is turned into a 512-number
   "fingerprint" once and stored in the image_embeddings table.
2. Searching: your words are turned into a fingerprint the same way, and the
   photos whose fingerprints are closest come first.

This module must import cleanly even when numpy or onnxruntime are missing,
so the rest of Postmark keeps working; engine_status() says what is missing.
Termux packages: python-numpy python-onnxruntime
"""
import json
import os
import re
import threading
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore", message=".*Unsupported platform.*")

try:
    import numpy as np
except Exception:  # numpy not installed
    np = None

from db import get_db, now_iso
from scanner import data_root, library_root, thumbnail_path, preview_path, ensure_thumbnail

_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
MODELS = {
    "vision": ("vision_model.onnx", _BASE + "onnx/vision_model.onnx"),
    "text": ("text_model.onnx", _BASE + "onnx/text_model.onnx"),
    "vocab": ("vocab.json", _BASE + "vocab.json"),
    "merges": ("merges.txt", _BASE + "merges.txt"),
}
MIN_BYTES = {"vision": 100_000_000, "text": 100_000_000, "vocab": 100_000, "merges": 100_000}

IMG_SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
CTX = 77                      # CLIP reads at most 77 tokens
BATCH = 8                     # photos per model call while indexing

_ENGINE = None
_ENGINE_LOCK = threading.RLock()


# ---------------------------------------------------------------- models

def models_dir():
    return os.path.join(data_root(), "models", "clip")


def model_path(key):
    return os.path.join(models_dir(), MODELS[key][0])


def models_present():
    for key in MODELS:
        p = model_path(key)
        if not (os.path.exists(p) and os.path.getsize(p) >= MIN_BYTES[key]):
            return False
    return True


DOWNLOAD = {"running": False, "error": "", "done": False, "file": "", "got": 0, "total": 0}


def _download_job():
    try:
        os.makedirs(models_dir(), exist_ok=True)
        for key in MODELS:
            name, url = MODELS[key]
            dest = model_path(key)
            if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES[key]:
                continue
            DOWNLOAD.update({"file": name, "got": 0, "total": 0})
            part = dest + ".part"
            req = urllib.request.Request(url, headers={"User-Agent": "Postmark"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                DOWNLOAD["total"] = int(resp.headers.get("Content-Length") or 0)
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    DOWNLOAD["got"] += len(chunk)
            if os.path.getsize(part) < MIN_BYTES[key]:
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
    DOWNLOAD.update({"running": True, "error": "", "done": False, "file": "", "got": 0, "total": 0})
    threading.Thread(target=_download_job, daemon=True).start()
    return True


# ---------------------------------------------------------------- tokenizer

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_TOKEN_RE = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+"
)


class Tokenizer:
    """CLIP's byte-pair tokenizer (same algorithm as the original)."""

    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [tuple(l.split()) for l in lines[1:] if l.strip()]
        self.ranks = {m: i for i, m in enumerate(merges) if len(m) == 2}
        self.byte_enc = _bytes_to_unicode()
        self.cache = {}
        self.bos = self.encoder.get("<|startoftext|>", 49406)
        self.eos = self.encoder.get("<|endoftext|>", 49407)

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        """[bos, ...tokens, eos], at most CTX long."""
        text = re.sub(r"\s+", " ", text or "").strip().lower()
        ids = []
        for tok in re.findall(_TOKEN_RE, text):
            tok = "".join(self.byte_enc[b] for b in tok.encode("utf-8"))
            for piece in self._bpe(tok).split(" "):
                if piece in self.encoder:
                    ids.append(self.encoder[piece])
        return [self.bos] + ids[:CTX - 2] + [self.eos]


# ---------------------------------------------------------------- engine

class _Engine:
    """Real engine: CLIP vision + text models under onnxruntime.
    images(list_of_PIL) -> (n, 512) unit vectors; text(str) -> (512,) unit vector."""

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        opts.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self.vis = ort.InferenceSession(model_path("vision"), opts, providers=prov)
        self.txt = ort.InferenceSession(model_path("text"), opts, providers=prov)
        self.tok = Tokenizer(model_path("vocab"), model_path("merges"))
        self.vis_in = self.vis.get_inputs()[0].name
        self.vis_out = self._pick(self.vis, "image_embeds")
        self.txt_in = [i.name for i in self.txt.get_inputs()]
        self.txt_out = self._pick(self.txt, "text_embeds")
        # Fail here (with a clear message) rather than half way through indexing.
        probe = self.text("a photo of a cat")
        if probe.shape != (512,):
            raise RuntimeError(f"Unexpected text output shape {probe.shape}.")

    @staticmethod
    def _pick(sess, preferred):
        names = [o.name for o in sess.get_outputs()]
        return preferred if preferred in names else names[0]

    def images(self, pil_images):
        batch = np.stack([_prep(im) for im in pil_images]).astype(np.float32)
        out = self.vis.run([self.vis_out], {self.vis_in: batch})[0]
        out = np.asarray(out, dtype=np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def _one_text(self, s):
        ids = self.tok.encode(s)
        n = len(ids)
        ids = ids + [self.tok.eos] * (CTX - n)
        feed = {}
        for name in self.txt_in:
            if "input_ids" in name:
                feed[name] = np.array([ids], dtype=np.int64)
            elif "attention_mask" in name:
                feed[name] = np.array([[1] * n + [0] * (CTX - n)], dtype=np.int64)
        out = self.txt.run([self.txt_out], feed)[0]
        v = np.asarray(out, dtype=np.float32).reshape(-1)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def text(self, query):
        """Average of the plain words and 'a photo of ...' (steadier results)."""
        v = self._one_text(query) + self._one_text("a photo of " + query)
        return v / max(float(np.linalg.norm(v)), 1e-9)


def _prep(im):
    """PIL image -> (3, 224, 224) float array. The whole picture is kept
    (padded to a square) so a full-length outfit is not cropped away."""
    from PIL import Image
    im = im.convert("RGB")
    w, h = im.size
    scale = IMG_SIZE / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.BICUBIC)
    pad = tuple(int(round(m * 255)) for m in MEAN)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), pad)
    canvas.paste(im, ((IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return arr.transpose(2, 0, 1)


def set_engine(engine):
    """Swap in another engine (used by tests)."""
    global _ENGINE
    _ENGINE = engine


def engine_status():
    if _ENGINE is not None:
        return {"ready": True, "reason": "", "models": True}
    if np is None:
        return {"ready": False, "models": models_present(),
                "reason": "numpy is not installed. In Termux run: pkg install python-numpy"}
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return {"ready": False, "models": models_present(),
                "reason": "onnxruntime is not installed. In Termux run: pkg install python-onnxruntime"}
    if not models_present():
        return {"ready": False, "models": False,
                "reason": "The Look model files (about 600 MB) have not been downloaded yet."}
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
            return None, f"Could not start the Look engine: {e}"
        return _ENGINE, ""


def tokenizer_check():
    """Known CLIP ids for 'a photo of a cat'. None if not loaded yet."""
    eng = _ENGINE
    tok = getattr(eng, "tok", None)
    if tok is None:
        return None
    return tok.encode("a photo of a cat") == [49406, 320, 1125, 539, 320, 2368, 49407]


# ---------------------------------------------------------------- storage

def ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_embeddings (
            media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
            vec BLOB,
            checked_at TEXT
        )
        """
    )


def _open_for_embedding(media_id, relpath, media_type):
    """The small thumbnail is plenty for CLIP and much faster than the original."""
    from PIL import Image
    paths = []
    try:
        t = ensure_thumbnail(media_id, relpath, media_type)
        if t:
            paths.append(t)
    except Exception:
        pass
    paths += [thumbnail_path(media_id), preview_path(media_id)]
    if media_type == "photo":
        paths.append(os.path.join(library_root(), relpath))
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with Image.open(p) as im:
                try:
                    im.draft("RGB", (640, 640))
                except Exception:
                    pass
                im.load()
                return im.convert("RGB")
        except Exception:
            continue
    return None


def _store(conn, media_id, vec):
    blob = None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO image_embeddings (media_id, vec, checked_at) VALUES (?, ?, ?)",
        (media_id, blob, now_iso()),
    )


def embed_rows(conn, rows):
    """Fingerprint a list of media rows and save them. Returns how many got one."""
    engine, why = _get_engine()
    if engine is None:
        raise RuntimeError(why)
    ensure_schema(conn)
    good = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        imgs, keep = [], []
        for r in chunk:
            im = _open_for_embedding(r["id"], r["relpath"], r["media_type"])
            if im is None:
                _store(conn, r["id"], None)      # nothing to look at (e.g. no ffmpeg still)
            else:
                imgs.append(im)
                keep.append(r["id"])
        if imgs:
            with _ENGINE_LOCK:
                vecs = engine.images(imgs)
            for mid, v in zip(keep, vecs):
                _store(conn, mid, v)
                good += 1
        conn.commit()
    return good


# ---------------------------------------------------------------- indexing job

INDEX = {"running": False, "processed": 0, "total": 0, "error": "", "stop": False, "started": 0.0}


def _index_job():
    conn = get_db()
    try:
        ensure_schema(conn)
        engine, why = _get_engine()
        if engine is None:
            INDEX["error"] = why
            return
        rows = conn.execute(
            "SELECT m.id, m.relpath, m.media_type FROM media m "
            "LEFT JOIN image_embeddings ie ON ie.media_id = m.id "
            "WHERE ie.media_id IS NULL ORDER BY (m.media_type = 'photo') DESC, m.id DESC"
        ).fetchall()
        INDEX["total"] = len(rows)
        for i in range(0, len(rows), BATCH):
            if INDEX["stop"]:
                return
            chunk = rows[i:i + BATCH]
            try:
                embed_rows(conn, chunk)
            except Exception as e:
                INDEX["error"] = str(e)
                return
            INDEX["processed"] += len(chunk)
    finally:
        conn.close()
        INDEX["running"] = False


def start_index():
    if INDEX["running"]:
        return False
    INDEX.update({"running": True, "processed": 0, "total": 0, "error": "", "stop": False,
                  "started": time.time()})
    threading.Thread(target=_index_job, daemon=True).start()
    return True


def stop_index():
    INDEX["stop"] = True


def index_async(media_ids):
    """Fingerprint freshly uploaded items in the background. Silent if not set up."""
    if not media_ids or INDEX["running"] or not engine_status()["ready"]:
        return

    def job():
        conn = get_db()
        try:
            marks = ",".join("?" * len(media_ids))
            rows = conn.execute(
                f"SELECT id, relpath, media_type FROM media WHERE id IN ({marks})", list(media_ids)
            ).fetchall()
            embed_rows(conn, rows)
        except Exception:
            pass
        finally:
            conn.close()

    threading.Thread(target=job, daemon=True).start()


def stats(conn):
    ensure_schema(conn)
    done = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE vec IS NOT NULL").fetchone()[0]
    skipped = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE vec IS NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    return {"indexed": done, "skipped": skipped, "total": total, "left": max(total - done - skipped, 0)}


# ---------------------------------------------------------------- searching

_MATRIX = {"n": -1, "mat": None, "ids": None}
_TEXT_CACHE = {}


def _matrix(conn):
    ensure_schema(conn)
    n = conn.execute("SELECT COUNT(*) FROM image_embeddings WHERE vec IS NOT NULL").fetchone()[0]
    if n == _MATRIX["n"] and _MATRIX["mat"] is not None:
        return _MATRIX["mat"], _MATRIX["ids"]
    rows = conn.execute("SELECT media_id, vec FROM image_embeddings WHERE vec IS NOT NULL").fetchall()
    if not rows:
        _MATRIX.update({"n": n, "mat": None, "ids": None})
        return None, None
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    ids = np.array([r["media_id"] for r in rows])
    _MATRIX.update({"n": n, "mat": mat, "ids": ids})
    return mat, ids


def search(conn, query, media_ids=None, top=200):
    """[(media_id, score)] best first. media_ids limits the search to those items.
    Raises RuntimeError if the engine is not ready."""
    query = (query or "").strip()
    if not query:
        return []
    engine, why = _get_engine()
    if engine is None:
        raise RuntimeError(why)
    key = query.lower()
    q = _TEXT_CACHE.get(key)
    if q is None:
        with _ENGINE_LOCK:
            q = engine.text(query)
        if len(_TEXT_CACHE) > 200:
            _TEXT_CACHE.clear()
        _TEXT_CACHE[key] = q
    mat, ids = _matrix(conn)
    if mat is None:
        return []
    sims = mat @ q
    if media_ids is not None:
        keep = np.isin(ids, np.fromiter(media_ids, dtype=ids.dtype, count=len(media_ids)))
        sims = np.where(keep, sims, -2.0)
    top = max(1, min(int(top), len(sims)))
    idx = np.argpartition(-sims, top - 1)[:top]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(ids[i]), float(sims[i])) for i in idx if sims[i] > -1.5]


# How far below the best match a picture may score and still be shown.
# Starting points, not tuned on real photos: compare with the scores on the Look page.
FIT_MARGIN = {"strict": 0.03, "balanced": 0.06, "loose": 0.10, "all": None}


def apply_fit(ranked, fit):
    """Drop matches that score much lower than the best one."""
    margin = FIT_MARGIN.get(fit, FIT_MARGIN["balanced"])
    if not ranked or margin is None:
        return ranked
    floor = ranked[0][1] - margin
    return [(m, s) for m, s in ranked if s >= floor]


# ---------------------------------------------------------------- cleanup helpers
# Zero-shot sorting: compare every fingerprint with a few descriptions.
PERSON_PROMPTS = [
    "a photo of a person", "a selfie", "a photo of a woman", "a photo of a man",
    "a group of people", "a close-up of a person's face", "a person posing", "a person's body",
]
OTHER_PROMPTS = {
    "Food": ["a photo of food", "a plate of food", "a drink in a glass"],
    "Landscape": ["a landscape", "a photo of nature", "a beach", "a sky with clouds", "mountains"],
    "Buildings": ["a building", "a street", "the interior of a room"],
    "Animals": ["a photo of an animal", "a pet"],
    "Objects": ["a photo of an object", "a product", "a car", "a gadget"],
    "Screenshots": ["a screenshot", "a document with text", "a meme with text"],
}
_CLS = {"n": -1, "ids": None, "p": None, "kind": None}


def _text_vec(query):
    engine, why = _get_engine()
    if engine is None:
        raise RuntimeError(why)
    key = query.lower()
    q = _TEXT_CACHE.get(key)
    if q is None:
        with _ENGINE_LOCK:
            q = engine.text(query)
        _TEXT_CACHE[key] = q
    return q


def classify(conn):
    """(ids, p_person, kind_names) for every indexed item. p_person is 0..1, how
    strongly the picture looks like it shows a person versus the other descriptions."""
    mat, ids = _matrix(conn)
    if mat is None:
        return None, None, None
    if _CLS["n"] == _MATRIX["n"] and _CLS["p"] is not None:
        return _CLS["ids"], _CLS["p"], _CLS["kind"]
    names = list(OTHER_PROMPTS)
    prompts = list(PERSON_PROMPTS)
    owner = [-1] * len(PERSON_PROMPTS)
    for gi, name in enumerate(names):
        for p in OTHER_PROMPTS[name]:
            prompts.append(p)
            owner.append(gi)
    T = np.stack([_text_vec(p) for p in prompts])
    logits = 100.0 * (mat @ T.T)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    pr = e / e.sum(axis=1, keepdims=True)
    owner = np.array(owner)
    p_person = pr[:, owner == -1].sum(axis=1)
    per_kind = np.stack([pr[:, owner == g].sum(axis=1) for g in range(len(names))], axis=1)
    kind = np.array(names)[per_kind.argmax(axis=1)]
    _CLS.update({"n": _MATRIX["n"], "ids": ids, "p": p_person, "kind": kind})
    return ids, p_person, kind


def no_people(conn, max_person=0.25):
    """[(media_id, p_person, kind)] most clearly person-free first."""
    ids, p, kind = classify(conn)
    if ids is None:
        return []
    sel = np.where(p <= max_person)[0]
    sel = sel[np.argsort(p[sel])]
    return [(int(ids[i]), float(p[i]), str(kind[i])) for i in sel]


def person_ids(conn, min_person=0.6):
    ids, p, _k = classify(conn)
    if ids is None:
        return []
    return [int(i) for i in ids[p >= min_person]]


def channel_guess(conn, media_ids, min_members=5, top=3):
    """For photos with no readable face: which channels do they look most like
    overall (outfit, place, style)? {media_id: [(album_id, name, score), ...]}.
    Much weaker than face matching, so only ever a suggestion."""
    mat, ids = _matrix(conn)
    if mat is None or not media_ids:
        return {}
    pos = {int(m): i for i, m in enumerate(ids)}
    rows = conn.execute(
        "SELECT ma.album_id, ma.media_id, a.name FROM media_albums ma JOIN albums a ON a.id = ma.album_id"
    ).fetchall()
    members, names = {}, {}
    for r in rows:
        i = pos.get(int(r["media_id"]))
        if i is not None:
            members.setdefault(r["album_id"], []).append(i)
            names[r["album_id"]] = r["name"]
    aids = [a for a, v in members.items() if len(v) >= min_members]
    if not aids:
        return {}
    C = np.stack([mat[members[a]].mean(axis=0) for a in aids])
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    out = {}
    want = [m for m in media_ids if int(m) in pos]
    Q = mat[[pos[int(m)] for m in want]]
    sims = Q @ C.T
    for row, m in zip(sims, want):
        best = np.argsort(-row)[:top]
        out[int(m)] = [(int(aids[j]), names[aids[j]], float(row[j])) for j in best]
    return out
