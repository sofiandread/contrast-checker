# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict
import io, math, urllib.request, json
import numpy as np
from PIL import Image

app = FastAPI(title="ContrastCheck API", version="1.2.0")

# ============= Helpers =============

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    thresh = 0.04045
    low = c <= thresh
    high = ~low
    out = np.zeros_like(c, dtype=np.float64)
    out[low] = c[low] / 12.92
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
    """Robust image fetcher: upgrades http→https, sets UA, handles timeouts; returns RGB PIL image."""
    try:
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ContrastCheck/1.2 (+fastapi)",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

def sample_region_pixels(img: Image.Image, box: Tuple[int, int, int, int], max_px=40000) -> np.ndarray:
    """Return Nx3 uint8 pixels from region; downsample if region is large."""
    x, y, w, h = box
    crop = img.crop((x, y, x + w, y + h))
    arr = np.array(crop)  # HxWx3
    H, W, _ = arr.shape
    N = H * W
    if N > max_px:
        stride = int(math.sqrt(N / max_px)) + 1
        arr = arr[::stride, ::stride, :]
    return arr.reshape(-1, 3)

def kmeans_palette(pixels: np.ndarray, k_min=2, k_max=8, iters=10, tol=1e-3) -> List[Dict]:
    """Small, deterministic-ish k-means to get a palette. Prunes <5% colors. Caps at 8."""
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
    """Return (border_pixels, inner_pixels) from an HxWx3 RGB array using a ring border."""
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

# ============= Models =============

class Box(BaseModel):
    x: int; y: int; w: int; h: int

class Location(BaseModel):
    location_id: str = Field(..., description="e.g., 'front', 'back', or 'loc1'")
    design_box: Box
    garment_box: Box

class Thresholds(BaseModel):
    min_garment_vs_design: float = 3.0
    warn_garment_vs_design: float = 3.4
    min_intra_design: float = 2.5

class RequestBoxMode(BaseModel):
    image_url: str
    locations: List[Location]
    thresholds: Optional[Thresholds] = Thresholds()
    @validator("locations")
    def non_empty(cls, v):
        if not v:
            raise ValueError("locations cannot be empty")
        return v

class CutoutRequest(BaseModel):
    cutout_url: str
    cutout_id: Optional[str] = None
    border_pct: float = 0.12
    thresholds: Optional[Thresholds] = Thresholds()

# ============= Routes =============

@app.get("/")
def root():
    return {
        "ok": True,
        "endpoints": ["/contrastcheck_upload", "/contrastcheck_cutout", "/contrastcheck", "/docs"]
    }

# ---- BOX MODE (image URL + boxes) ----
@app.post("/contrastcheck")
def contrastcheck(req: RequestBoxMode):
    img = pil_from_url(req.image_url)
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
        for c in design_palette:
            c["hex"] = to_hex(tuple(c["rgb"]))
            c["luminance"] = round(relative_luminance(tuple(c["rgb"])), 6)

        g_vs_d = []
        for c in design_palette:
            ratio = round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3)
            g_vs_d.append({
                "designHex": c["hex"], "designRGB": c["rgb"], "ratio": ratio,
                "pass": ratio >= req.thresholds.min_garment_vs_design,
                "borderline": (ratio >= req.thresholds.min_garment_vs_design and ratio < req.thresholds.warn_garment_vs_design),
            })

        intra = []
        dp = design_palette
        for i in range(len(dp)):
            for j in range(i + 1, len(dp)):
                a = tuple(dp[i]["rgb"]); b = tuple(dp[j]["rgb"])
                ratio = round(contrast_ratio(a, b), 3)
                intra.append({
                    "a": dp[i]["hex"], "b": dp[j]["hex"], "ratio": ratio,
                    "pass": ratio >= req.thresholds.min_intra_design,
                })

        failing_garment_pairs = [p for p in g_vs_d if not p["pass"]]
        intra_fail = [p for p in intra if not p["pass"]]

        verdict = "pass"
        if failing_garment_pairs or intra_fail:
            verdict = "fail"
        elif any(p["borderline"] for p in g_vs_d):
            verdict = "warn"

        notes = []
        for p in failing_garment_pairs:
            notes.append(f"Design {p['designHex']} vs garment {g_hex} too low ({p['ratio']}<={req.thresholds.min_garment_vs_design})")
        for p in intra_fail:
            notes.append(f"Design colors {p['a']} vs {p['b']} too low ({p['ratio']}<={req.thresholds.min_intra_design})")

        results.append({
            "location_id": loc.location_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": round(g_lum, 6)},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "intraDesignPairs": intra},
            "thresholdsUsed": req.thresholds.dict(),
            "contrastVerdict": verdict,
            "notes": notes,
        })

    return {"contrastcheck": {"image_url": req.image_url, "results": results}}

# ---- CUTOUT MODE (URL, garment from border ring) ----
@app.post("/contrastcheck_cutout")
def contrastcheck_cutout(req: CutoutRequest):
    try:
        img = pil_from_url(req.cutout_url).convert("RGB")
        arr = np.array(img)
        border_px, inner_px = split_border_inner(arr, req.border_pct)

        garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
        if not garment_palette:
            raise HTTPException(400, "Unable to derive garment palette from border")
        g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb)
        g_lum = relative_luminance(g_rgb)

        design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
        if not design_palette:
            raise HTTPException(400, "Unable to derive design palette from inner area")
        for c in design_palette:
            c["hex"] = to_hex(tuple(c["rgb"]))
            c["luminance"] = round(relative_luminance(tuple(c["rgb"])), 6)

        g_vs_d, intra = [], []
        for c in design_palette:
            ratio = round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3)
            g_vs_d.append({
                "designHex": c["hex"], "designRGB": c["rgb"], "ratio": ratio,
                "pass": ratio >= req.thresholds.min_garment_vs_design,
                "borderline": (ratio >= req.thresholds.min_garment_vs_design and ratio < req.thresholds.warn_garment_vs_design),
            })

        for i in range(len(design_palette)):
            for j in range(i + 1, len(design_palette)):
                a = tuple(design_palette[i]["rgb"]); b = tuple(design_palette[j]["rgb"])
                ratio = round(contrast_ratio(a, b), 3)
                intra.append({
                    "a": design_palette[i]["hex"], "b": design_palette[j]["hex"], "ratio": ratio,
                    "pass": ratio >= req.thresholds.min_intra_design,
                })

        failing_g = [p for p in g_vs_d if not p["pass"]]
        intra_fail = [p for p in intra if not p["pass"]]
        verdict = "fail" if (failing_g or intra_fail) else ("warn" if any(p["borderline"] for p in g_vs_d) else "pass")

        notes = []
        for p in failing_g:
            notes.append(f"Design {p['designHex']} vs garment {g_hex} too low ({p['ratio']}<={req.thresholds.min_garment_vs_design})")
        for p in intra_fail:
            notes.append(f"Design colors {p['a']} vs {p['b']} too low ({p['ratio']}<={req.thresholds.min_intra_design})")

        result = {
            "cutout_id": req.cutout_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": round(g_lum, 6)},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "intraDesignPairs": intra},
            "thresholdsUsed": req.thresholds.dict(),
            "contrastVerdict": verdict,
            "notes": notes,
        }
        return {"contrastcheck": {"cutout_url": req.cutout_url, "results": [result]}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")

# ---- UPLOAD MODE (multipart cutout PNG) ----
@app.post("/contrastcheck_upload")
async def contrastcheck_upload(
    file: UploadFile = File(...),
    cutout_id: Optional[str] = Form(None),
    border_pct: float = Form(0.12),
    thresholds_json: Optional[str] = Form(None),
):
    try:
        # thresholds (with safe defaults)
        t = {"min_garment_vs_design": 3.0, "warn_garment_vs_design": 3.4, "min_intra_design": 2.5}
        if thresholds_json:
            try:
                t.update(json.loads(thresholds_json))
            except Exception as e:
                raise HTTPException(400, f"Invalid thresholds_json: {e}")

        # read image
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)

        # ring split
        border_px, inner_px = split_border_inner(arr, float(border_pct))

        garment_palette = kmeans_palette(border_px, k_min=1, k_max=3)
        if not garment_palette:
            raise HTTPException(400, "Unable to derive garment palette from border")
        g_rgb = tuple(garment_palette[0]["rgb"]); g_hex = to_hex(g_rgb)
        g_lum = relative_luminance(g_rgb)

        design_palette = kmeans_palette(inner_px, k_min=2, k_max=8)
        if not design_palette:
            raise HTTPException(400, "Unable to derive design palette from inner area")
        for c in design_palette:
            c["hex"] = to_hex(tuple(c["rgb"]))
            c["luminance"] = round(relative_luminance(tuple(c["rgb"])), 6)

        g_vs_d, intra = [], []
        for c in design_palette:
            ratio = round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3)
            g_vs_d.append({
                "designHex": c["hex"], "designRGB": c["rgb"], "ratio": ratio,
                "pass": ratio >= t["min_garment_vs_design"],
                "borderline": (ratio >= t["min_garment_vs_design"] and ratio < t["warn_garment_vs_design"]),
            })

        for i in range(len(design_palette)):
            for j in range(i + 1, len(design_palette)):
                a = tuple(design_palette[i]["rgb"]); b = tuple(design_palette[j]["rgb"])
                ratio = round(contrast_ratio(a, b), 3)
                intra.append({
                    "a": design_palette[i]["hex"], "b": design_palette[j]["hex"], "ratio": ratio,
                    "pass": ratio >= t["min_intra_design"],
                })

        failing_g = [p for p in g_vs_d if not p["pass"]]
        intra_fail = [p for p in intra if not p["pass"]]
        verdict = "fail" if (failing_g or intra_fail) else ("warn" if any(p["borderline"] for p in g_vs_d) else "pass")

        notes = []
        for p in failing_g:
            notes.append(f"Design {p['designHex']} vs garment {g_hex} too low ({p['ratio']}<={t['min_garment_vs_design']})")
        for p in intra_fail:
            notes.append(f"Design colors {p['a']} vs {p['b']} too low ({p['ratio']}<={t['min_intra_design']})")

        result = {
            "cutout_id": cutout_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": round(g_lum, 6)},
            "designPalette": design_palette,
            "contrast": {"garmentVsDesign": g_vs_d, "intraDesignPairs": intra},
            "thresholdsUsed": t,
            "contrastVerdict": verdict,
            "notes": notes,
        }
        return {"contrastcheck": {"results": [result]}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")
