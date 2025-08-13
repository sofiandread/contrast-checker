from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict
import io, math, urllib.request
import numpy as np
from PIL import Image

app = FastAPI(title="ContrastCheck API", version="1.0.0")

# --------- Helpers ---------
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    thresh = 0.04045
    low = c <= thresh
    high = ~low
    out = np.zeros_like(c, dtype=np.float64)
    out[low] = c[low] / 12.92
    out[high] = ((c[high] + 0.055) / 1.055) ** 2.4
    return out

def relative_luminance(rgb: Tuple[int,int,int]) -> float:
    r, g, b = [float(x) for x in rgb]
    lin = srgb_to_linear(np.array([r, g, b]))
    return 0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]

def contrast_ratio(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    la = relative_luminance(a)
    lb = relative_luminance(b)
    L1, L2 = (max(la, lb), min(la, lb))
    return (L1 + 0.05) / (L2 + 0.05)

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def pil_from_url(url: str) -> Image.Image:
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(400, f"Failed to load image: {e}")

def sample_region_palette(img: Image.Image, box: Tuple[int,int,int,int], max_px=40000) -> np.ndarray:
    """
    Return pixels from region as Nx3 array (uint8). Downsamples if too big.
    """
    x, y, w, h = box
    crop = img.crop((x, y, x+w, y+h))
    arr = np.array(crop)  # HxWx3
    H, W, _ = arr.shape
    N = H*W
    if N > max_px:
        # uniform grid downsample
        stride = int(math.sqrt(N / max_px)) + 1
        arr = arr[::stride, ::stride, :]
    return arr.reshape(-1, 3)

def kmeans_palette(pixels: np.ndarray, k_min=2, k_max=8, iters=10, tol=1e-3) -> List[Dict]:
    """
    Lightweight k-means; adapt k by diminishing returns on inertia.
    Returns list of dicts: {rgb:[r,g,b], percent:float}
    """
    if pixels.shape[0] == 0:
        return []
    # Pre-normalize
    X = pixels.astype(np.float64)

    def run_kmeans(k):
        # init: random sample of points
        rng = np.random.default_rng(42 + k)  # stable-ish
        idx = rng.choice(X.shape[0], size=min(k, X.shape[0]), replace=False)
        centers = X[idx].copy()
        last_inertia = None
        for _ in range(iters):
            # assign
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            # update
            new_centers = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(centers.shape[0])])
            inertia = float(((X - new_centers[labels])**2).sum())
            if last_inertia is not None and abs(last_inertia - inertia) < tol*last_inertia:
                centers = new_centers
                break
            centers = new_centers
            last_inertia = inertia
        # final stats
        counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.float64)
        percents = counts / counts.sum()
        # sort by percent desc
        order = np.argsort(-percents)
        centers = centers[order]
        percents = percents[order]
        return centers, percents, inertia if last_inertia is not None else 0.0

    # try different k, stop when improvement < 10%
    best = None
    last_inertia = None
    for k in range(k_min, k_max+1):
        centers, percents, inertia = run_kmeans(k)
        if best is None:
            best = (centers, percents, inertia)
            last_inertia = inertia
            continue
        improvement = (last_inertia - inertia) / (last_inertia + 1e-9)
        if improvement < 0.10:
            break
        best = (centers, percents, inertia)
        last_inertia = inertia

    centers, percents, _ = best
    out = []
    for c, p in zip(centers, percents):
        rgb = [int(round(v)) for v in c.clip(0,255)]
        out.append({"rgb": rgb, "percent": float(round(p, 4))})
    # prune tiny colors (<5%)
    out = [c for c in out if c["percent"] >= 0.05] or out[:3]
    return out[:8]

def to_hex(rgb: Tuple[int,int,int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

# --------- Models ---------
class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int

class Location(BaseModel):
    location_id: str = Field(..., description="e.g., 'front', 'back', or 'loc1'")
    design_box: Box
    garment_box: Box

class Thresholds(BaseModel):
    min_garment_vs_design: float = 3.0
    warn_garment_vs_design: float = 3.4
    min_intra_design: float = 2.5

class Request(BaseModel):
    image_url: str
    locations: List[Location]
    thresholds: Optional[Thresholds] = Thresholds()

    @validator("locations")
    def non_empty(cls, v):
        if not v:
            raise ValueError("locations cannot be empty")
        return v

# --------- Core ---------
@app.post("/contrastcheck")
def contrastcheck(req: Request):
    img = pil_from_url(req.image_url)
    W, H = img.size
    results = []
    for loc in req.locations:
        # clamp boxes
        dx, dy, dw, dh = clamp_box(loc.design_box.x, loc.design_box.y, loc.design_box.w, loc.design_box.h, W, H)
        gx, gy, gw, gh = clamp_box(loc.garment_box.x, loc.garment_box.y, loc.garment_box.w, loc.garment_box.h, W, H)

        # sample garment color from garment_box
        garment_pixels = sample_region_palette(img, (gx, gy, gw, gh))
        if garment_pixels.size == 0:
            raise HTTPException(400, f"Empty garment region for {loc.location_id}")
        garment_palette = kmeans_palette(garment_pixels, k_min=1, k_max=3)
        # choose dominant garment color (first)
        g_rgb = tuple(garment_palette[0]["rgb"])
        g_hex = to_hex(g_rgb)
        g_lum = relative_luminance(g_rgb)

        # design palette
        design_pixels = sample_region_palette(img, (dx, dy, dw, dh))
        if design_pixels.size == 0:
            raise HTTPException(400, f"Empty design region for {loc.location_id}")
        design_palette = kmeans_palette(design_pixels, k_min=2, k_max=8)
        # enrich palette with hex & luminance
        for c in design_palette:
            c["hex"] = to_hex(tuple(c["rgb"]))
            c["luminance"] = round(relative_luminance(tuple(c["rgb"])), 6)

        # garment vs design pairs
        g_vs_d = []
        for c in design_palette:
            ratio = round(contrast_ratio(g_rgb, tuple(c["rgb"])), 3)
            pass_min = ratio >= req.thresholds.min_garment_vs_design
            borderline = (not pass_min) and (ratio >= req.thresholds.min_garment_vs_design)  # kept for clarity
            g_vs_d.append({
                "designHex": c["hex"],
                "designRGB": c["rgb"],
                "ratio": ratio,
                "pass": pass_min,
                "borderline": (ratio >= req.thresholds.min_garment_vs_design and ratio < req.thresholds.warn_garment_vs_design)
            })

        # intra-design pairs
        intra = []
        dp = design_palette
        for i in range(len(dp)):
            for j in range(i+1, len(dp)):
                a = tuple(dp[i]["rgb"]); b = tuple(dp[j]["rgb"])
                ratio = round(contrast_ratio(a, b), 3)
                intra.append({
                    "a": dp[i]["hex"], "b": dp[j]["hex"], "ratio": ratio,
                    "pass": ratio >= req.thresholds.min_intra_design
                })

        # verdicts
        failing_garment_pairs = [p for p in g_vs_d if not p["pass"]]
        intra_fail = [p for p in intra if not p["pass"]]

        verdict = "pass"
        notes = []
        if failing_garment_pairs or intra_fail:
            verdict = "fail"
        elif any(p["borderline"] for p in g_vs_d):
            verdict = "warn"

        for p in failing_garment_pairs:
            notes.append(f"Design {p['designHex']} vs garment {g_hex} too low ({p['ratio']}<={req.thresholds.min_garment_vs_design})")
        for p in intra_fail:
            notes.append(f"Design colors {p['a']} vs {p['b']} too low ({p['ratio']}<={req.thresholds.min_intra_design})")

        results.append({
            "location_id": loc.location_id,
            "garment": {"rgb": list(g_rgb), "hex": g_hex, "luminance": round(g_lum, 6)},
            "designPalette": design_palette,
            "contrast": {
                "garmentVsDesign": g_vs_d,
                "intraDesignPairs": intra
            },
            "thresholdsUsed": req.thresholds.dict(),
            "contrastVerdict": verdict,
            "notes": notes
        })

    # top-level structure you can pass downstream
    return {
        "contrastcheck": {
            "image_url": req.image_url,
            "results": results
        }
    }
