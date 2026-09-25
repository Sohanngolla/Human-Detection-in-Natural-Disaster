#!/usr/bin/env python3
"""
Human Detection with Activity Analysis for Natural Disaster Response
====================================================================

Detects people in an image, classifies each person's activity from pose
keypoints (Lying down / Sitting / Arms raised / Using device / Walking /
Standing), and writes an annotated image plus a text report.

Built for disaster / search-and-rescue scenes where a single aerial or
ground photo needs a fast read on how many people are present and who may
be immobile (e.g. lying down) versus active.

Models:
  - yolov8n.pt        -> person detection (COCO class 0)
  - yolov8n-pose.pt   -> 17-keypoint pose estimation for activity analysis

Usage:
  python detect_humans.py path/to/image.jpg
  python detect_humans.py scene.heic --out results/ --conf 0.25

Supported input: JPG, PNG, HEIC, AVIF, WEBP, BMP, TIFF (auto-converted).
"""

import os
import argparse

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import pillow_heif
pillow_heif.register_heif_opener()
try:
    import pillow_avif  # noqa: F401  (registers AVIF support if installed)
except Exception:
    pass


# 17 COCO keypoint indices used by the pose model
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 13, 14

SKELETON = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12),  # torso
    (11, 12),          # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def convert_to_supported_format(image_path):
    """Convert any image format to JPG for YOLO compatibility."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    ext = image_path.lower().split(".")[-1]
    if ext in ("jpg", "jpeg", "png"):
        return image_path

    print(f"Converting {ext.upper()} to JPG...")
    img = Image.open(image_path)
    jpg_path = image_path.rsplit(".", 1)[0] + "_converted.jpg"
    rgb_img = img.convert("RGB") if img.mode in ("RGBA", "LA", "P") else img
    rgb_img.save(jpg_path, "JPEG", quality=95)
    print(f"Converted to: {jpg_path}")
    return jpg_path


def analyze_activity_from_keypoints(keypoints):
    """Classify activity from 17 COCO pose keypoints (x, y, confidence)."""
    if keypoints is None or len(keypoints) < 17:
        return "Person detected"

    nose = keypoints[NOSE]
    l_shoulder, r_shoulder = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
    l_wrist, r_wrist = keypoints[L_WRIST], keypoints[R_WRIST]
    l_hip, r_hip = keypoints[L_HIP], keypoints[R_HIP]
    l_knee, r_knee = keypoints[L_KNEE], keypoints[R_KNEE]

    # Need a reasonable number of visible keypoints to reason about pose
    if sum(1 for kp in keypoints if kp[2] > 0.3) < 5:
        return "Person detected"

    shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_center_y = (l_hip[1] + r_hip[1]) / 2
    body_height = abs(shoulder_center_y - hip_center_y)

    # Lying down: shoulders and hips at nearly the same height
    if body_height < 30:
        return "Lying down"

    # Sitting: knees visible and roughly level with the hips
    if l_knee[2] > 0.3 and r_knee[2] > 0.3:
        if (l_knee[1] + r_knee[1]) / 2 < hip_center_y + 20:
            return "Sitting"

    # Arms raised: a wrist tracked well above its shoulder
    if l_wrist[2] > 0.3 and l_wrist[1] < l_shoulder[1] - 30:
        return "Arms raised"
    if r_wrist[2] > 0.3 and r_wrist[1] < r_shoulder[1] - 30:
        return "Arms raised"

    # Using device: a hand near the face
    if l_wrist[2] > 0.3 and abs(l_wrist[1] - nose[1]) < 80:
        return "Using device"
    if r_wrist[2] > 0.3 and abs(r_wrist[1] - nose[1]) < 80:
        return "Using device"

    # Walking: noticeable vertical separation between the knees
    if l_knee[2] > 0.3 and r_knee[2] > 0.3 and abs(l_knee[1] - r_knee[1]) > 60:
        return "Walking"

    return "Standing"


def draw_results(image, bbox, activity, person_num, keypoints=None):
    """Draw the bounding box, pose skeleton, keypoints, and activity label."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if keypoints is not None and len(keypoints) >= 17:
        for a, b in SKELETON:
            p1, p2 = keypoints[a], keypoints[b]
            if p1[2] > 0.3 and p2[2] > 0.3:
                cv2.line(image, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), (255, 0, 0), 3)
        for p in keypoints:
            if p[2] > 0.3:
                cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

    label = f"Person {person_num}: {activity}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
    tx, ty = x1, max(th + 20, y1 - 15)
    cv2.rectangle(image, (tx - 10, ty - th - 10),
                  (tx + tw + 10, ty + base + 10), (0, 0, 0), -1)
    cv2.putText(image, label, (tx, ty), font, scale, (0, 255, 0), thick)
    return image


def load_image(file_path):
    """Load an image with OpenCV, falling back to PIL if needed."""
    img = cv2.imread(file_path)
    if img is None:
        pil_img = Image.open(file_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def run(image_path, out_dir, conf):
    os.makedirs(out_dir, exist_ok=True)
    file_path = convert_to_supported_format(image_path)
    if file_path is None:
        return

    img = load_image(file_path)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    print("Step 1: detecting persons...")
    detection_model = YOLO("yolov8n.pt")
    result = detection_model(file_path, conf=conf, verbose=False)[0]
    boxes = result.boxes
    person_indices = [i for i, cls in enumerate(boxes.cls) if int(cls) == 0]
    person_count = len(person_indices)
    print(f"Found {person_count} person(s).")

    annotated = img.copy()
    activity_results = []

    if person_count:
        print("Step 2: analyzing poses and activities...")
        pose_model = YOLO("yolov8n-pose.pt")
        pose_result = pose_model(file_path, conf=conf, verbose=False)[0]

        for idx, i in enumerate(person_indices):
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())

            keypoints = None
            if hasattr(pose_result, "keypoints") and len(pose_result.keypoints) > idx:
                try:
                    xy = pose_result.keypoints.xy[idx].cpu().numpy()
                    kc = pose_result.keypoints.conf[idx].cpu().numpy()
                    keypoints = np.column_stack([xy, kc.reshape(-1, 1)])
                except Exception:
                    keypoints = None

            activity = analyze_activity_from_keypoints(keypoints)
            activity_results.append(
                {"person_id": idx + 1, "activity": activity, "confidence": confidence}
            )
            annotated = draw_results(annotated, bbox, activity, idx + 1, keypoints)
            print(f"Person {idx + 1}: {activity} ({confidence:.2%})")

    image_out = os.path.join(out_dir, "result_with_activities.jpg")
    cv2.imwrite(image_out, annotated)

    report_out = os.path.join(out_dir, "activity_report.txt")
    with open(report_out, "w") as f:
        f.write("=" * 60 + "\nHUMAN ACTIVITY DETECTION REPORT\n" + "=" * 60 + "\n\n")
        f.write(f"Source file: {image_path}\n")
        f.write(f"Size: {w}x{h}\n")
        f.write(f"Total persons detected: {person_count}\n\n")
        if activity_results:
            f.write("INDIVIDUAL ANALYSIS:\n" + "-" * 40 + "\n")
            for r in activity_results:
                f.write(f"\nPerson {r['person_id']}:\n")
                f.write(f"  Activity: {r['activity']}\n")
                f.write(f"  Detection Confidence: {r['confidence']:.2%}\n")
        else:
            f.write("No persons detected.\n")

    print(f"\nDone. Annotated image -> {image_out}\n       Report        -> {report_out}")


def main():
    ap = argparse.ArgumentParser(description="Human detection + activity analysis")
    ap.add_argument("image", help="path to input image (jpg/png/heic/avif/webp/...)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    args = ap.parse_args()
    run(args.image, args.out, args.conf)


if __name__ == "__main__":
    main()
