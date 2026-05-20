import cv2
import numpy as np
from PIL import Image
import os

OUTPUT_DIR = "./farm_walkway_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def find_vanishing_point(img):
    """Detect vanishing point via Hough lines on the lower half."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[h // 2:, :]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=80, maxLineGap=20)
    if lines is None:
        return w // 2, h // 3

    candidates = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        y1 += h // 2; y2 += h // 2          # shift back to full image coords
        if abs(x2 - x1) < 5:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        candidates.append((slope, intercept))

    # pairwise intersections → cluster
    pts = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            s1, b1 = candidates[i]
            s2, b2 = candidates[j]
            if abs(s1 - s2) < 0.01:
                continue
            x = (b2 - b1) / (s1 - s2)
            y = s1 * x + b1
            if 0 < x < w and 0 < y < h:
                pts.append((x, y))

    if not pts:
        return w // 2, h // 3

    pts = np.array(pts)
    vp = np.median(pts, axis=0)
    return int(vp[0]), int(vp[1])


def build_ground_quad(img, vp_x, vp_y):
    """Define a quadrilateral covering the visible dirt path."""
    h, w = img.shape[:2]

    # bottom edge: full width near bottom of frame
    bl = (int(w * 0.05), int(h * 0.97))
    br = (int(w * 0.95), int(h * 0.97))

    # top edge: converges toward vanishing point
    spread = 0.12
    tl = (int(vp_x - w * spread), int(vp_y + (h - vp_y) * 0.18))
    tr = (int(vp_x + w * spread), int(vp_y + (h - vp_y) * 0.18))

    return np.array([tl, tr, br, bl], dtype=np.float32)


def make_herringbone_texture(tw=800, th=800):
    """Generate a beige/sandy herringbone paver texture."""
    tex = np.ones((th, tw, 3), dtype=np.uint8)
    tex[:] = (185, 195, 210)   # base sandy-beige (BGR)

    bw, bh = 40, 20            # brick width / height in pixels
    mortar = 2
    colors = [
        (175, 190, 205),
        (190, 200, 215),
        (180, 195, 210),
        (170, 185, 200),
    ]
    rng = np.random.default_rng(42)

    def draw_brick(canvas, x, y, w, h, horizontal):
        bw2, bh2 = (w, h) if horizontal else (h, w)
        cx, cy = int(x), int(y)
        col = colors[rng.integers(0, len(colors))]
        cv2.rectangle(canvas, (cx + mortar, cy + mortar),
                       (cx + bw2 - mortar, cy + bh2 - mortar),
                       col, -1)
        # subtle highlight / shadow
        cv2.line(canvas, (cx + mortar, cy + mortar),
                  (cx + bw2 - mortar, cy + mortar),
                  tuple(min(255, c + 18) for c in col), 1)
        cv2.line(canvas, (cx + mortar, cy + bh2 - mortar),
                  (cx + bw2 - mortar, cy + bh2 - mortar),
                  tuple(max(0, c - 18) for c in col), 1)

    # herringbone: alternating H and V pairs in 2×bh × 2×bw tiles
    tile_w, tile_h = bw * 2, bh * 2
    for ty in range(-tile_h, th + tile_h, tile_h):
        for tx in range(-tile_w, tw + tile_w, tile_w):
            # two horizontal bricks stacked
            draw_brick(tex, tx,      ty,      bw, bh, True)
            draw_brick(tex, tx,      ty + bh, bw, bh, True)
            # two vertical bricks beside them
            draw_brick(tex, tx + bw, ty,      bw, bh, False)
            draw_brick(tex, tx + bw, ty + bh, bw, bh, False)

    # add slight noise for realism
    noise = rng.integers(-12, 12, tex.shape, dtype=np.int16)
    tex = np.clip(tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return tex


def apply_paver_overlay(img, quad_pts):
    """Perspective-warp herringbone texture onto the ground quad."""
    h, w = img.shape[:2]
    tex = make_herringbone_texture(800, 800)

    dst = quad_pts.astype(np.float32)
    src = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

    # mask for the quad region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad_pts.astype(np.int32)], 255)

    # ── lighting match: sample brightness from original ground ──────────────
    ground_region = cv2.bitwise_and(img, img, mask=mask)
    orig_mean = cv2.mean(ground_region, mask=mask)[:3]
    warp_mean = cv2.mean(warped, mask=mask)[:3]

    scale = np.array([
        orig_mean[c] / max(warp_mean[c], 1) for c in range(3)
    ], dtype=np.float32)
    scale = np.clip(scale, 0.5, 1.8)

    warped_f = warped.astype(np.float32)
    for c in range(3):
        warped_f[:, :, c] *= scale[c]
    warped_adj = np.clip(warped_f, 0, 255).astype(np.uint8)

    # ── shadow / luminance transfer from original ────────────────────────────
    orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    warp_lab = cv2.cvtColor(warped_adj, cv2.COLOR_BGR2LAB).astype(np.float32)

    # blend L channel: 40 % original luminance, 60 % paver
    mask_f = (mask / 255.0)[:, :, np.newaxis]
    warp_lab[:, :, 0] = warp_lab[:, :, 0] * 0.60 + orig_lab[:, :, 0] * 0.40
    warped_lit = cv2.cvtColor(
        np.clip(warp_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # composite
    result = img.copy()
    mask3 = np.stack([mask] * 3, axis=-1) > 0
    result[mask3] = warped_lit[mask3]
    return result, mask


def add_curb_borders(img, quad_pts, mask):
    """Draw concrete curb strips along the two side edges of the quad."""
    result = img.copy()
    h, w = img.shape[:2]

    tl, tr, br, bl = quad_pts.astype(np.int32)
    curb_w = max(6, int(w * 0.012))
    curb_color = (200, 205, 205)   # light concrete gray (BGR)
    shadow_color = (140, 145, 145)

    def draw_curb(p1, p2, p3, p4):
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(result, [pts], curb_color)
        # darker bottom edge for depth
        cv2.line(result, p4, p3, shadow_color, 2)

    # left curb: between left edge of quad and inset
    def offset_toward(a, b, d):
        vec = np.array(b, dtype=float) - np.array(a, dtype=float)
        norm = vec / (np.linalg.norm(vec) + 1e-9)
        perp = np.array([norm[1], -norm[0]])
        return (int(a[0] + perp[0] * d), int(a[1] + perp[1] * d))

    left_tl = offset_toward(tl, bl, -curb_w)
    left_bl = offset_toward(bl, tl, -curb_w)
    draw_curb(tl, bl, left_bl, left_tl)

    right_tr = offset_toward(tr, br, curb_w)
    right_br = offset_toward(br, tr, curb_w)
    draw_curb(tr, br, right_br, right_tr)

    return result


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return

    print(f"Processing {os.path.basename(path)}  {img.shape[1]}×{img.shape[0]}")

    vp_x, vp_y = find_vanishing_point(img)
    print(f"  Vanishing point: ({vp_x}, {vp_y})")

    quad = build_ground_quad(img, vp_x, vp_y)

    result, mask = apply_paver_overlay(img, quad)
    result = add_curb_borders(result, quad, mask)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_pavers.jpg")
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out_path}")

    # side-by-side comparison
    h, w = img.shape[:2]
    divider = np.full((h, 6, 3), 255, dtype=np.uint8)
    side_by_side = np.hstack([img, divider, result])

    # add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(side_by_side, "BEFORE", (20, 50),
                font, 1.4, (255, 255, 255), 3)
    cv2.putText(side_by_side, "AFTER", (w + 26, 50),
                font, 1.4, (255, 255, 255), 3)

    comp_path = os.path.join(OUTPUT_DIR, f"{base}_comparison.jpg")
    cv2.imwrite(comp_path, side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Saved: {comp_path}")


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        # default: process every image in the working directory
        for f in sorted(os.listdir(".")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f)

    for p in paths:
        process_image(p)

    print("Done. Results in", OUTPUT_DIR)
