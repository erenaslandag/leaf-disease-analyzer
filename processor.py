from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class LeafAnalysisResult:
    filtered_rgb: np.ndarray
    leaf_mask: np.ndarray
    disease_mask: np.ndarray
    overlay_rgb: np.ndarray
    total_leaf_pixels: int
    diseased_pixels: int
    disease_ratio: float
    severity_label: str

def analyze_leaf(bgr_img, filter_method="median", filter_kernel=5, morph_kernel=7, hue_low=8, hue_high=34, min_sat=45):
    k = max(3, int(filter_kernel) | 1)
    denoised = cv2.GaussianBlur(bgr_img, (k, k), 0) if filter_method == "gaussian" else cv2.medianBlur(bgr_img, k)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    leaf_score = cv2.max(255 - lab[:, :, 1], lab[:, :, 2])
    _, otsu = cv2.threshold(leaf_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mk = max(3, int(morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    leaf_mask = np.where(labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))), 255, 0).astype(np.uint8) if num_labels > 1 else cleaned

    fill_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(leaf_mask, fill_contours, -1, 255, thickness=cv2.FILLED)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array([hue_low, min_sat, 30], np.uint8), np.array([hue_high, 255, 255], np.uint8))

    v_channel = hsv[:, :, 2]
    h_channel = hsv[:, :, 0]
    leaf_v_values = v_channel[leaf_mask > 0]
    dark_mask = np.zeros_like(leaf_mask)
    if leaf_v_values.size > 0:
        dark_thresh = int(np.median(leaf_v_values) * 0.60)
        not_green = (h_channel < 36) | (h_channel > 95)
        dark_mask = np.where((v_channel >= 40) & (v_channel < dark_thresh) & not_green, 255, 0).astype(np.uint8)

    inner_leaf = cv2.erode(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    combined = cv2.bitwise_or(color_mask, dark_mask)
    disease_mask = cv2.bitwise_and(combined, inner_leaf)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    total_px = int(np.count_nonzero(leaf_mask))
    diseased_px = int(np.count_nonzero(disease_mask))
    ratio = (diseased_px / total_px) * 100.0 if total_px > 0 else 0.0
    label = "Sağlıklı (< %5)" if ratio < 5.0 else "Erken Evre (%5–%20)" if ratio < 20 else "Orta Şiddet (%20–%45)" if ratio < 45 else "Kritik (%45+)"

    overlay = bgr_img.copy()
    highlight = overlay.copy()
    highlight[disease_mask > 0] = (0, 60, 220)
    overlay = cv2.addWeighted(highlight, 0.45, overlay, 0.55, 0)
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    return LeafAnalysisResult(
        filtered_rgb=cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB),
        leaf_mask=leaf_mask,
        disease_mask=disease_mask,
        overlay_rgb=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        total_leaf_pixels=total_px,
        diseased_pixels=diseased_px,
        disease_ratio=ratio,
        severity_label=label,
    )