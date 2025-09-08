# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple, Dict
import io, math, urllib.request, json
import numpy as np
from PIL import Image

app = FastAPI(title="ContrastCheck API", version="1.7.0")

# ========= Tunables =========
MAX_SAMPLE_PX = 25000           # per-region sampling cap
MAX_IMAGE_PX  = 1_200_000       # downscale huge images before ring split (~1.2MP)
FAST_PASS_MEAN_MARGIN = 0.8     # fast-pass if mean >= base - 0.2? (negative margin means stricter)
FAST_PASS_FAILCOV_MAX = 0.10    # fast-pass only if failing coverage <= 10%

# ========= Utilities =========

def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_py(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    thresh = 0.04045
    low = c <= thresh
    out = np.zeros_like(c, dtype=np.float64)
    out[low] = c[low] / 12.92
    high = ~low
    out[high] = ((c[high] + 0.055) / 1.055) ** 2.4
    return out

def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [float(x) for x in rgb]
    lin = srgb_to_linear(np.array([r, g, b]))
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

def contrast_ratio(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    la = relative_luminance(a)
    lb = relative_luminance(b)
    L1, L2 = (max(la, lb), min(la, lb))
    return (L1 + 0.05) / (L2 + 0.05)

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def pil_from_url(url: str) -> Image.Image:
    try:
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ContrastCheck/1.7 (+fastapi)", "Accept": "image/*,*/*;q=0.8"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

def sample_region_pixels(img: Image.Image, box: Tuple[int, int, int, int], max_px=MAX_SAMPLE_PX) -> np.ndarray:
    x, y, w, h = box
    crop = img.crop((x, y, x + w, y + h)).convert("RGB")
    arr = np.array(crop)  # HxWx3
    H, W, _ = arr.shape
    N = H * W
    if N > max_px:
        stride = int(math.sqrt(N / max_px)) + 1
        arr = arr[::stride, ::stride, :]
    return arr.reshape(-1, 3)

def maybe_downscale_array(arr: np.ndarray, max_px=MAX_IMAGE_PX) -> np.ndarray:
    H, W = arr.shape[:2]
    N = H * W
    if N <= max_px: return arr
    stride = int(math.sqrt(N / max_px)) + 1
    return arr[::stride, ::stride, :]

def kmeans_palette(pixels: np.ndarray, k_min=2, k_max=8, iters=8, tol=1e-3) -> List[Dict]:
    if pixels.size == 0:
        return []
    X = pixels.astype(np.float64)

    def run_kmeans(k):
        rng = np.random.default_rng(42 + k)
        idx = rng.choice(X.shape[0], size=min(k, X.shape[0]), replace=False)
        centers = X[idx].copy()
        last_inertia = None
        for _ in range(iters):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            new_centers = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                for i in range(centers.shape[0])
            ])
            inertia = float(((X - new_centers[labels]) ** 2).sum())
            if last_inertia is not None and abs(last_inertia - inertia) < tol * (last_inertia + 1e-9):
                centers = new_centers
                break
            centers = new_centers
            last_inertia = inertia
        counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.float64)
        percents = counts / counts.sum()
        order = np.argsort(-percents)
        centers = centers[order]
        percents = percents[order]
        return centers, percents, last_inertia or 0.0

    best = None
    last_inertia = None
    for k in range(k_min, k_max + 1):
        centers, percents, inertia = run_kmeans(k)
        if best is None:
            best = (centers, percents, inertia); last_inertia = inertia; continue
        improvement = (last_inertia - inertia) / (last_inertia + 1e-9)
        if improvement < 0.10:
            break
        best = (centers, percents, inertia); last_inertia = inertia

    centers, percents, _ = best
    out = []
    for c, p in zip(centers, percents):
        rgb = [int(round(v)) for v in c.clip(0, 255)]
        out.append({"rgb": rgb, "percent": float(round(p, 4))})
    out = [c for c in out if c["percent"] >= 0.05] or out[:3]
    return out[:8]

def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def split_border_inner(arr: np.ndarray, border_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W, _ = arr.shape
    t = max(1, int(round(border_pct * min(W, H))))
    if t * 2 >= min(W, H):
        flat = arr.reshape(-1, 3)
        return flat, flat
    top = arr[:t, :, :]
    bottom = arr[H - t:, :, :]
    left = arr[:, :t, :]
    right = arr[:, W - t:, :]
    border = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0
    )
    inner = arr[t:H - t, t:W - t, :].reshape(-1, 3)
    return border, inner

# Alpha-safe cutout loaders
def load_cutout_rgb_from_bytes(data: bytes, bg=(255, 255, 255)) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA"):
        base = Image.new("RGBA", im.size, (*bg, 255))
        im = Image.alpha_composite(base, im.convert("RGBA")).convert("RGB")
    else:
        im = im.convert("RGB")
    return im

def load_cutout_rgb_from_url(url: str, bg=(255, 255, 255)) -> Image.Image:
    raw = pil_from_url(url)
    buf = io.BytesIO()
    fmt = "PNG" if (raw.mode in ("RGBA", "LA")) else "JPEG"
    raw.save(buf, format=fmt)
    buf.seek(0)
    return load_cutout_rgb_from_bytes(buf.getvalue(), bg)

# ---- Leniency / overall-look helpers ----
def coverage_adjusted_threshold(pct: float, base: float, floor: float = 2.2) -> float:
    pct = float(pct or 0.0)
    if pct >= 0.25:
        return base
    elif pct >= 0.10:
        return max(floor, base - 0.4)
    else:
        return max(floor, base - 0.8)

def weighted_contrast_stats(g_rgb: Tuple[int,int,int], design_palette: List[Dict], threshold: float) -> Dict[str, float]:
    ratios, weights = [], []
    for c in design_palette:
        r = contrast_ratio(g_rgb, tuple(c["rgb"]))
        ratios.append(r)
        weights.append(float(c.get("percent", 0.0)) or 0.0)
    if not ratios:
        return {"weighted_mean": 0.0, "weighted_p25": 0.0, "fail_coverage": 1.0, "ratios": []}
    r = np.asarray(ratios, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() or 1.0)
    weighted_mean = float((w * r).sum())
    order = np.argsort(r); r_sorted = r[order]; w_sorted = w[order]; cw = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cw, 0.25)); idx = min(max(idx, 0), len(r_sorted) - 1)
    weighted_p25 = float(r_sorted[idx])
    fail_coverage = float(w[r < threshold].sum())
    return {"weighted_mean": round(weighted_mean, 3), "weighted_p25": round(weighted_p25, 3),
            "fail_coverage": round(fail_coverage, 3), "ratios": [float(round(x, 3)) for x in r.tolist()]}

def overall_contrast_verdict(g_vs_d: List[Dict], stats: Dict[str, float], base_min: float) -> Tuple[str, List[str]]:
    notes = []
    MEAN_MARGIN_FAIL = 0.6
    P25_MARGIN_FAIL  = 0.4
    FAIL_COVERAGE_MAX = 0.40
    EXTREME_LOW = max(2.0, base_min - 1.0)
    extreme = [p for p in g_vs_d if p["ratio"] < EXTREME_LOW and float(p.get("coverage", 0) or 0) >= 0.10]
    if extreme:
        notes.append("One or more sizable colors are extremely low in contrast.")
    overall_fail = (
        stats["weighted_mean"] < (base_min - MEAN_MARGIN_FAIL) or
        stats["weighted_p25"]  < (base_min - P25_MARGIN_FAIL) or
        stats["fail_coverage"] > FAIL_COVERAGE_MAX or
        len(extreme) > 0
    )
    any_miss = any(p["ratio"] < p.get("required", base_min) for p in g_vs_d)
    any_border = any(p.get("borderline") for p in g_vs_d)
    if overall_fail: verdict = "fail"
    elif any_miss or any_border: verdict = "warn"
    else: verdict = "pass"
    return verdict, notes

# ========= Models =========
class Box(BaseModel):
    x: int; y: int; w: int; h: int

class Location(BaseModel):
    location_id: str = Field(..., description="e.g., 'front', 'back', or 'loc1'")
    design_box: Box
    garment_box: Box
    text_boxes: Optional[List[Box]] = Field(default_factory=list)

class Thresholds(BaseModel):
    min_garment_vs_design: float = 3.0
    warn_garment_vs_design: float = 3.4
    min_intra_design: float = 2.5
    min_text_vs_garment: float = 3.4
    warn_text_vs_garment: float = 4.0

class RequestBoxMode(BaseModel):
    image_url: str
    locations: List[Location]
    thresholds: Optional[Thresholds] = Thresholds()
    @field_validator("locations")
    @classmethod
    def non_empty(cls, v):
        if not v: raise ValueError("locations cannot be empty")
        return v

class CutoutRequest(BaseModel):
    cutout_url: str
    cutout_id: Optional[str] = None
    border_pct: float = 0.12
    thresholds: Optional[Thresholds] = Thresholds()

class CutoutBatchItem(BaseModel):
    cutout_url: str
    cutout_id: Optional[str] = None
    border_pct: Optional[float] = None

class CutoutBatchRequest(BaseModel):
    items: List[CutoutBatchItem]
    thresholds: Optional[Thresholds] = Thresholds()
    @field_validator("items")
    @classmethod
    def _non_empty(cls, v):
        if not v: raise ValueError("items cannot be empty")
        return v

# ========= Shared helpers =========
def build_checks(g_rgb, g_hex, design_palette, thresholds):
    for c in design_palette:
        c["hex"] = to_hex(tuple(c["rgb"]))
        c["luminance"] = float(round(relative_luminance(tuple(c["rgb"])), 6))
    g_vs_d = []
    for c in design_palette:
        ratio = float(round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3))
        required = coverage_adjusted_threshold(c.get("percent", 0.0), thresholds.min_garment_vs_design)
        warn_at = max(required, thresholds.warn_garment_vs_design)
        g_vs_d.append({
            "designHex": c["hex"], "designRGB": c["rgb"], "ratio": ratio,
            "required": float(round(required, 3)),
            "pass": bool(ratio >= required),
            "borderline": bool(ratio >= required and ratio < warn_at),
            "coverage": c.get("percent", 0.0),
        })
    stats = weighted_contrast_stats(g_rgb, design_palette, thresholds.min_garment_vs_design)
    return g_vs_d, stats

def text_checks(img: Image.Image, boxes: List[Box], g_rgb: Tuple[int,int,int], thresholds: Thresholds, W: int, H: int):
    out = []
    for tb in (boxes or []):
        tx, ty, tw, th = clamp_box(tb.x, tb.y, tb.w, tb.h, W, H)
        text_pixels = sample_region_pixels(img, (tx, ty, tw, th))
        if text_pixels.size == 0: continue
        text_palette = kmeans_palette(text_pixels, k_min=1, k_max=5)
        for c in text_palette:
            c["hex"] = to_hex(tuple(c["rgb"]))
            ratio = float(round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3))
            out.append({
                "textHex": c["hex"], "textRGB": c["rgb"], "ratio": ratio,
                "pass": bool(ratio >= thresholds.min_text_vs_garment),
                "borderline": bool(ratio >= thresholds.min_text_vs_garment and ratio < thresholds.warn_text_vs_garment),
            })
    return out

# ========= Routes =========

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/contrastcheck_upload", "/contrastcheck_cutout",
                                      "/contrastcheck", "/contrastcheck_upload_batch",
                                      "/contrastcheck_cutout_batch", "/docs"]}

# ---- BOX MODE (image URL + boxes) ----
@app.post("/contrastcheck")
def contrastcheck(req: RequestBoxMode):
    img = pil_from_url(req.image_url).convert("RGB")
    W, H = img.size
    results = []
    for loc in req.locations:
        dx, dy, dw, dh = clamp_box(loc.design_box.x, loc.design_box.y, loc.design_box.w, loc.design_box.h, W, H)
        gx, gy, gw, gh = clamp_box(loc.garment_box.x, loc.garment_box.y, loc.garment_box.w, loc.garment_box.h, W, H)

        garment_pixels = sample_region_pixels(img, (gx, gy, gw, gh))
        if garment_pixels.size == 0:
            raise HTTPException(400, f"Empty garment region for {loc.location_id}")
        garment_palette = kmeans_palette(garment_pixels, k_min=1, k_max=3)
        g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb)
        g_lum = relative_luminance(g_rgb)

        design_pixels = sample_region_pixels(img, (dx, dy, dw, dh))
        if design_pixels.size == 0:
            raise HTTPException(400, f"Empty design region for {loc.location_id}")
        design_palette = kmeans_palette(design_pixels, k_min=2, k_max=8)

        g_vs_d, stats = build_checks(g_rgb, g_hex, design_palette, req.thresholds)

        # Fast pass: obvious overall OK
        if (stats["weighted_mean"] >= (req.thresholds.min_garment_vs_design - FAST_PASS_MEAN_MARGIN) and
            stats["fail_coverage"] <= FAST_PASS_FAILCOV_MAX):
            text_results = text_checks(img, loc.text_boxes, g_rgb, req.thresholds, W, H)
            text_fail = [p for p in text_results if not p["pass"]]
            verdict = "fail" if text_fail else "pass"
        else:
            # intra (notes only)
            text_results = text_checks(img, loc.text_boxes, g_rgb, req.thresholds, W, H)
            text_fail = [p for p in text_results if not p["pass"]]
            verdict, _ = overall_contrast_verdict(g_vs_d, stats, req.thresholds.min_garment_vs_design)
            if text_fail: verdict = "fail"

        # suggestions
        misses = [p for p in g_vs_d if p["ratio"] < p["required"]]
        suggestions = [
            f"Improve contrast for {p['designHex']} vs {g_hex} (ratio {p['ratio']:.3f} < {p['required']:.1f})"
            for p in sorted(misses, key=lambda x: x["ratio"])[:3]
        ]
        if text_fail:
            for p in text_fail[:3]:
                suggestions.append(f"TEXT {p['textHex']} vs {g_hex} too low (ratio {p['ratio']:.3f} < {req.thresholds.min_text_vs_garment})")

        notes = []
        if suggestions: notes.append("Suggestions: " + "; ".join(suggestions))
        notes.append(f"Overall stats: mean={stats['weighted_mean']}, p25={stats['weighted_p25']}, failing_coverage={stats['fail_coverage']}")

        results.append({
            "location_id": loc.location_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": float(round(g_lum, 6))},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "overallStats": stats, "textChecks": text_results},
            "thresholdsUsed": req.thresholds.dict(),
            "contrastVerdict": verdict,
            "notes": notes,
        })
    return to_py({"contrastcheck": {"image_url": req.image_url, "results": results}})

# ---- CUTOUT MODE (URL, garment from border ring) ----
@app.post("/contrastcheck_cutout")
def contrastcheck_cutout(req: CutoutRequest):
    try:
        img = load_cutout_rgb_from_url(req.cutout_url, bg=(255, 255, 255))
        arr = maybe_downscale_array(np.array(img))
        border_px, inner_px = split_border_inner(arr, req.border_pct)

        garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
        if not garment_palette: raise HTTPException(400, "Unable to derive garment palette from border")
        g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb); g_lum = relative_luminance(g_rgb)

        design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
        if not design_palette: raise HTTPException(400, "Unable to derive design palette from inner area")

        g_vs_d, stats = build_checks(g_rgb, g_hex, design_palette, req.thresholds)

        if (stats["weighted_mean"] >= (req.thresholds.min_garment_vs_design - FAST_PASS_MEAN_MARGIN) and
            stats["fail_coverage"] <= FAST_PASS_FAILCOV_MAX):
            verdict = "pass"
        else:
            verdict, _ = overall_contrast_verdict(g_vs_d, stats, req.thresholds.min_garment_vs_design)

        misses = [p for p in g_vs_d if p["ratio"] < p["required"]]
        suggestions = [f"Improve contrast for {p['designHex']} vs {g_hex} (ratio {p['ratio']:.3f} < {p['required']:.1f})"
                       for p in sorted(misses, key=lambda x: x["ratio"])[:3]]

        notes = []
        if suggestions: notes.append("Suggestions: " + "; ".join(suggestions))
        notes.append(f"Overall stats: mean={stats['weighted_mean']}, p25={stats['weighted_p25']}, failing_coverage={stats['fail_coverage']}")

        result = {
            "cutout_id": req.cutout_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": float(round(g_lum, 6))},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "overallStats": stats},
            "thresholdsUsed": req.thresholds.dict(),
            "contrastVerdict": verdict,
            "notes": notes,
        }
        return to_py({"contrastcheck": {"cutout_url": req.cutout_url, "results": [result]}})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")

# ---- UPLOAD MODE (single file) ----
@app.post("/contrastcheck_upload")
async def contrastcheck_upload(
    file: UploadFile | None = File(None),
    cutout: UploadFile | None = File(None),
    cutout_id: Optional[str] = Form(None),
    border_pct: float = Form(0.12),
    thresholds_json: Optional[str] = Form(None),
):
    try:
        upload = file or cutout
        if upload is None:
            raise HTTPException(400, "No file uploaded. Expected field named 'file' or 'cutout'.")

        t = {
            "min_garment_vs_design": 3.0, "warn_garment_vs_design": 3.4, "min_intra_design": 2.5,
            "min_text_vs_garment": 3.4, "warn_text_vs_garment": 4.0,
        }
        if thresholds_json:
            try: t.update(json.loads(thresholds_json))
            except Exception as e: raise HTTPException(400, f"Invalid thresholds_json: {e}")

        data = await upload.read()
        img = load_cutout_rgb_from_bytes(data, bg=(255, 255, 255))
        arr = maybe_downscale_array(np.array(img))
        border_px, inner_px = split_border_inner(arr, float(border_pct))

        garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
        if not garment_palette: raise HTTPException(400, "Unable to derive garment palette from border")
        g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb); g_lum = relative_luminance(g_rgb)

        design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
        if not design_palette: raise HTTPException(400, "Unable to derive design palette from inner area")

        class TObj:
            min_garment_vs_design=float(t["min_garment_vs_design"])
            warn_garment_vs_design=float(t.get("warn_garment_vs_design", t["min_garment_vs_design"]))
            min_intra_design=float(t["min_intra_design"])
            min_text_vs_garment=float(t["min_text_vs_garment"])
            warn_text_vs_garment=float(t["warn_text_vs_garment"])

        g_vs_d, stats = build_checks(g_rgb, g_hex, design_palette, TObj)

        if (stats["weighted_mean"] >= (TObj.min_garment_vs_design - FAST_PASS_MEAN_MARGIN) and
            stats["fail_coverage"] <= FAST_PASS_FAILCOV_MAX):
            verdict = "pass"
        else:
            verdict, _ = overall_contrast_verdict(g_vs_d, stats, TObj.min_garment_vs_design)

        misses = [p for p in g_vs_d if p["ratio"] < p["required"]]
        suggestions = [f"Improve contrast for {p['designHex']} vs {g_hex} (ratio {p['ratio']:.3f} < {p['required']:.1f})"
                       for p in sorted(misses, key=lambda x: x["ratio"])[:3]]

        notes = []
        if suggestions: notes.append("Suggestions: " + "; ".join(suggestions))
        notes.append(f"Overall stats: mean={stats['weighted_mean']}, p25={stats['weighted_p25']}, failing_coverage={stats['fail_coverage']}")

        result = {
            "cutout_id": cutout_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": float(round(g_lum, 6))},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "overallStats": stats},
            "thresholdsUsed": t,
            "contrastVerdict": verdict,
            "notes": notes,
        }
        return to_py({"contrastcheck": {"results": [result]}})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")

# ---- UPLOAD MODE (batch: multiple files) ----
@app.post("/contrastcheck_upload_batch")
async def contrastcheck_upload_batch(
    files: List[UploadFile] = File(..., description="Repeat 'files' field for each cutout"),
    cutout_ids_json: Optional[str] = Form(None),
    border_pct: float = Form(0.12),
    thresholds_json: Optional[str] = Form(None),
):
    try:
        t = {
            "min_garment_vs_design": 3.0, "warn_garment_vs_design": 3.4, "min_intra_design": 2.5,
            "min_text_vs_garment": 3.4, "warn_text_vs_garment": 4.0,
        }
        if thresholds_json:
            try: t.update(json.loads(thresholds_json))
            except Exception as e: raise HTTPException(400, f"Invalid thresholds_json: {e}")

        cutout_ids = []
        if cutout_ids_json:
            try:
                cutout_ids = list(json.loads(cutout_ids_json))
            except Exception as e:
                raise HTTPException(400, f"Invalid cutout_ids_json: {e}")

        class TObj:
            min_garment_vs_design=float(t["min_garment_vs_design"])
            warn_garment_vs_design=float(t.get("warn_garment_vs_design", t["min_garment_vs_design"]))
            min_intra_design=float(t["min_intra_design"])
            min_text_vs_garment=float(t["min_text_vs_garment"])
            warn_text_vs_garment=float(t["warn_text_vs_garment"])

        out_results = []
        for idx, upload in enumerate(files):
            data = await upload.read()
            img = load_cutout_rgb_from_bytes(data, bg=(255, 255, 255))
            arr = maybe_downscale_array(np.array(img))
            border_px, inner_px = split_border_inner(arr, float(border_pct))
            garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
            if not garment_palette: raise HTTPException(400, "Unable to derive garment palette from border")
            g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb); g_lum = relative_luminance(g_rgb)
            design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
            if not design_palette: raise HTTPException(400, "Unable to derive design palette from inner area")
            g_vs_d, stats = build_checks(g_rgb, g_hex, design_palette, TObj)
            if (stats["weighted_mean"] >= (TObj.min_garment_vs_design - FAST_PASS_MEAN_MARGIN) and
                stats["fail_coverage"] <= FAST_PASS_FAILCOV_MAX):
                verdict = "pass"
            else:
                verdict, _ = overall_contrast_verdict(g_vs_d, stats, TObj.min_garment_vs_design)
            misses = [p for p in g_vs_d if p["ratio"] < p["required"]]
            suggestions = [f"Improve contrast for {p['designHex']} vs {g_hex} (ratio {p['ratio']:.3f} < {p['required']:.1f})"
                           for p in sorted(misses, key=lambda x: x["ratio"])[:3]]
            notes = []
            if suggestions: notes.append("Suggestions: " + "; ".join(suggestions))
            notes.append(f"Overall stats: mean={stats['weighted_mean']}, p25={stats['weighted_p25']}, failing_coverage={stats['fail_coverage']}")
            out_results.append({
                "cutout_id": cutout_ids[idx] if idx < len(cutout_ids) else None,
                "filename": upload.filename,
                "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": float(round(g_lum, 6))},
                "designPalette": design_palette,
                "contrast": {"garmentVsDesign": g_vs_d, "overallStats": stats},
                "thresholdsUsed": t,
                "contrastVerdict": verdict,
                "notes": notes,
            })
        return to_py({"contrastcheck": {"results": out_results}})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")

# ---- CUTOUT MODE (batch: multiple URLs) ----
@app.post("/contrastcheck_cutout_batch")
def contrastcheck_cutout_batch(req: CutoutBatchRequest):
    try:
        out_results = []
        for it in req.items:
            img = load_cutout_rgb_from_url(it.cutout_url, bg=(255, 255, 255))
            arr = maybe_downscale_array(np.array(img))
            border_px, inner_px = split_border_inner(arr, it.border_pct if it.border_pct is not None else req.thresholds and req.thresholds.min_garment_vs_design or 0.12)  # default 0.12 if not set
            garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
            if not garment_palette: raise HTTPException(400, "Unable to derive garment palette from border")
            g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb); g_lum = relative_luminance(g_rgb)
            design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
            if not design_palette: raise HTTPException(400, "Unable to derive design palette from inner area")
            g_vs_d, stats = build_checks(g_rgb, g_hex, design_palette, req.thresholds)
            if (stats["weighted_mean"] >= (req.thresholds.min_garment_vs_design - FAST_PASS_MEAN_MARGIN) and
                stats["fail_coverage"] <= FAST_PASS_FAILCOV_MAX):
                verdict = "pass"
            else:
                verdict, _ = overall_contrast_verdict(g_vs_d, stats, req.thresholds.min_garment_vs_design)
            misses = [p for p in g_vs_d if p["ratio"] < p["required"]]
            suggestions = [f"Improve contrast for {p['designHex']} vs {g_hex} (ratio {p['ratio']:.3f} < {p['required']:.1f})"
                           for p in sorted(misses, key=lambda x: x["ratio"])[:3]]
            notes = []
            if suggestions: notes.append("Suggestions: " + "; ".join(suggestions))
            notes.append(f"Overall stats: mean={stats['weighted_mean']}, p25={stats['weighted_p25']}, failing_coverage={stats['fail_coverage']}")
            out_results.append({
                "cutout_id": it.cutout_id,
                "cutout_url": it.cutout_url,
                "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": float(round(g_lum, 6))},
                "designPalette": design_palette,
                "contrast": {"garmentVsDesign": g_vs_d, "overallStats": stats},
                "thresholdsUsed": req.thresholds.dict(),
                "contrastVerdict": verdict,
                "notes": notes,
            })
        return to_py({"contrastcheck": {"results": out_results}})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")
