#!/usr/bin/env python3
"""
Lumbarization Detection Pipeline for AP Spine X-rays.

Workflow:
1. Run YOLO segmentation to detect all vertebrae
2. Find D12 as anchor point (trust D12 detection)
3. Identify vertebrae below D12 (should be L1-L5, S1)
4. Reorder them correctly based on spatial position (top to bottom)
5. Detect lumbarization: 6 vertebrae after D12 instead of 5
6. Generate diagnostic visualization

Usage:
    # Single image
    python3 lumbarization_detection.py --image /path/to/xray.jpg

    # Batch processing
    python3 lumbarization_detection.py --image /path/to/folder/ --batch

    # Custom output directory
    python3 lumbarization_detection.py --image /path/to/xray.jpg --output-dir /path/to/results

Author: AI Assistant
Date: 2026-05-08
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Default paths
DEFAULT_MODEL = Path(
    "/root/SPINE_PATHOLOGIES/old_build_models/SPINE/PATHOLOGY/"
    "ALL_DORSAL_LUMBAR_runs/yolo11m_seg2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = Path("/root/SPINE_PATHOLOGIES/LUMBARIZATION/results")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Expected vertebra sequence after D12 (top to bottom in AP view)
# We assign labels based on COUNT, not model predictions:
# 5 vertebrae → L1, L2, L3, L4, L5 (normal)
# 6 vertebrae → L1, L2, L3, L4, L5, S1 (lumbarization)
EXPECTED_SEQUENCE_5 = ["L1", "L2", "L3", "L4", "L5"]
EXPECTED_SEQUENCE_6 = ["L1", "L2", "L3", "L4", "L5", "S1"]


def extract_min_area_rect(mask: np.ndarray, polygon_xy: List) -> Dict:
    """
    Extract minimum area rectangle from segmentation mask/polygon.
    
    Args:
        mask: Binary segmentation mask
        polygon_xy: Polygon points [[x1, y1], [x2, y2], ...]
        
    Returns:
        Dictionary with cx, cy, width, height, angle_deg_opencv
    """
    if polygon_xy and len(polygon_xy) >= 5:
        points = np.array(polygon_xy, dtype=np.float32)
        rect = cv2.minAreaRect(points)
        (cx, cy), (w, h), angle = rect
        
        # Get 4 corners
        box = cv2.boxPoints(rect)
        box = np.asarray(box, dtype=np.float32)
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        # Sort by y first, then by x
        sorted_by_y = sorted(box, key=lambda p: p[1])
        top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
        corners_ordered = top_two + bottom_two[::-1]  # tl, tr, br, bl
        
        return {
            "cx": float(cx),
            "cy": float(cy),
            "width": float(w),
            "height": float(h),
            "angle_deg_opencv": float(angle),
            "corners": [{"x": float(pt[0]), "y": float(pt[1])} for pt in corners_ordered]
        }
    
    return None


def run_segmentation(
    model: YOLO, 
    image_path: Path,
    conf_threshold: float = 0.5
) -> List[Dict]:
    """
    Run YOLO segmentation and extract detections.
    
    Returns:
        List of detection dictionaries with class_name, confidence, min_area_rect, etc.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False
    )
    
    if not results or len(results) == 0:
        return []
    
    result = results[0]
    detections = []
    
    if not hasattr(result, 'boxes') or result.boxes is None:
        return []
    
    # Process each detection
    for idx in range(len(result.boxes)):
        box = result.boxes[idx]
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "original_class_name": class_name,
            "corrected": False
        }
        
        # Extract mask polygon if available
        if hasattr(result, 'masks') and result.masks is not None:
            mask = result.masks[idx]
            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                polygon = mask.xy[0].tolist()
                detection["polygon_xy"] = polygon
                
                # Calculate min area rect
                rect_info = extract_min_area_rect(None, polygon)
                if rect_info:
                    detection["min_area_rect"] = rect_info
        
        # Fallback: use bounding box if no mask
        if "min_area_rect" not in detection:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            detection["min_area_rect"] = {
                "cx": float(cx),
                "cy": float(cy),
                "width": float(w),
                "height": float(h),
                "angle_deg_opencv": 0.0,
                "corners": [
                    {"x": float(x1), "y": float(y1)},
                    {"x": float(x2), "y": float(y1)},
                    {"x": float(x2), "y": float(y2)},
                    {"x": float(x1), "y": float(y2)}
                ]
            }
        
        detections.append(detection)
    
    return detections


def find_d12_anchor(detections: List[Dict]) -> Optional[Dict]:
    """
    Find D12 vertebra to use as anchor point.
    
    Returns:
        D12 detection or None if not found
    """
    d12_candidates = [d for d in detections if d["class_name"] == "D12"]
    
    if not d12_candidates:
        return None
    
    # If multiple D12, take the one with highest confidence
    d12_candidates.sort(key=lambda d: d["confidence"], reverse=True)
    return d12_candidates[0]


def get_vertebrae_below_d12(
    detections: List[Dict],
    d12: Dict,
    y_tolerance: float = 50.0
) -> List[Dict]:
    """
    Find all vertebrae below D12 (higher y-coordinate in image).
    
    Args:
        detections: All detections
        d12: D12 vertebra detection
        y_tolerance: Minimum y-distance below D12 to consider
        
    Returns:
        List of vertebrae below D12
    """
    d12_cy = d12["min_area_rect"]["cy"]
    
    below = []
    for det in detections:
        if det is d12:
            continue
        
        cy = det["min_area_rect"]["cy"]
        
        # Must be below D12 (higher y coordinate)
        if cy > d12_cy + y_tolerance:
            below.append(det)
    
    return below


def remove_spatial_duplicates(
    vertebrae: List[Dict],
    cy_threshold: float = 90.0,
) -> List[Dict]:
    """
    Merge detections on the same vertebral level (duplicate boxes).

    YOLO often predicts two boxes at one level (e.g. L3 and L1 at cy=753).
    Those must count as ONE vertebra before L1–L5 / L1–L5–S1 labeling.

    Args:
        vertebrae: Detections below D12
        cy_threshold: Max |cy1 - cy2| to treat as same level (pixels)

    Returns:
        One detection per distinct spinal level (highest confidence kept)
    """
    if not vertebrae:
        return []

    # Process cranial → caudal; keep best box per level band
    sorted_verts = sorted(
        vertebrae,
        key=lambda d: (d["min_area_rect"]["cy"], d["min_area_rect"]["cx"]),
    )

    levels: List[Dict] = []
    for det in sorted_verts:
        cy = det["min_area_rect"]["cy"]
        merged = False
        for i, kept in enumerate(levels):
            if abs(cy - kept["min_area_rect"]["cy"]) <= cy_threshold:
                if det.get("confidence", 0) > kept.get("confidence", 0):
                    levels[i] = det
                merged = True
                break
        if not merged:
            levels.append(det)

    return sorted(levels, key=lambda d: d["min_area_rect"]["cy"])


def reorder_lumbar_vertebrae(vertebrae: List[Dict]) -> List[Dict]:
    """
    Reorder ALL vertebrae from top to bottom and assign labels BASED ON COUNT.
    
    Logic:
    - 5 vertebrae detected → Assign L1, L2, L3, L4, L5 (NORMAL)
    - 6 vertebrae detected → Assign L1, L2, L3, L4, L5, S1 (LUMBARIZATION)
    
    Args:
        vertebrae: List of ALL vertebrae below D12
        
    Returns:
        Reordered list with corrected class names
    """
    # Sort by y-coordinate (top to bottom)
    sorted_verts = sorted(vertebrae, key=lambda d: d["min_area_rect"]["cy"])
    
    num_vertebrae = len(sorted_verts)
    
    # Choose sequence based on count
    if num_vertebrae == 5:
        sequence = EXPECTED_SEQUENCE_5  # L1-L5
    elif num_vertebrae == 6:
        sequence = EXPECTED_SEQUENCE_6  # L1-L5-S1
    else:
        # For other counts, use longest sequence available
        sequence = EXPECTED_SEQUENCE_6
    
    # Assign labels based on position
    corrected = []
    for idx, vert in enumerate(sorted_verts):
        new_vert = dict(vert)
        
        if idx < len(sequence):
            expected_label = sequence[idx]
            
            if new_vert["class_name"] != expected_label:
                new_vert["original_class_name"] = new_vert["class_name"]
                new_vert["class_name"] = expected_label
                new_vert["corrected"] = True
        else:
            # Extra vertebrae beyond expected
            new_vert["corrected"] = False
            print(f"  ⚠ Warning: Extra vertebra #{idx+1} beyond expected count")
        
        corrected.append(new_vert)
    
    return corrected


def analyze_lumbarization(
    d12: Dict,
    lumbar_vertebrae: List[Dict]
) -> Dict:
    """
    Analyze for lumbarization: 6 vertebrae after D12 instead of 5.
    
    Normal anatomy: D12 → L1, L2, L3, L4, L5 (5 vertebrae)
    Lumbarization: D12 → L1, L2, L3, L4, L5, S1 (6 vertebrae)
    
    Returns:
        Dictionary with analysis results
    """
    num_vertebrae = len(lumbar_vertebrae)
    
    # Check S1 presence
    s1_present = any(v["class_name"] == "S1" for v in lumbar_vertebrae)
    
    # Simple logic: 6 vertebrae = Lumbarization, 5 vertebrae = Normal
    if num_vertebrae == 6:
        lumbarization_detected = True
        if s1_present:
            diagnosis = "Lumbarization: 6 vertebrae below D12 (L1-L5-S1)"
        else:
            diagnosis = "Lumbarization: 6 vertebrae below D12 (S1 may be mislabeled)"
        severity = "positive"
    elif num_vertebrae == 5:
        lumbarization_detected = False
        diagnosis = "Normal: 5 vertebrae below D12 (L1-L5)"
        severity = "normal"
    elif num_vertebrae < 5:
        lumbarization_detected = False
        diagnosis = f"Incomplete: Only {num_vertebrae} vertebrae detected below D12"
        severity = "incomplete"
    else:
        # More than 6
        lumbarization_detected = True
        diagnosis = f"Anomalous: {num_vertebrae} vertebrae detected below D12 (expected max 6)"
        severity = "anomalous"
    
    return {
        "lumbarization_detected": lumbarization_detected,
        "diagnosis": diagnosis,
        "severity": severity,
        "vertebrae_count": num_vertebrae,
        "s1_present": s1_present,
        "expected_normal": 5,
        "expected_lumbarization": 6,
        "d12_position_cy": d12["min_area_rect"]["cy"]
    }


def create_visualization(
    image_path: Path,
    d12: Dict,
    lumbar_vertebrae: List[Dict],
    analysis: Dict,
    output_path: Path
) -> bool:
    """
    Create diagnostic visualization showing D12, reordered vertebrae, and diagnosis.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return False
    
    # Colors (BGR)
    COLOR_D12 = (0, 255, 0)      # Green
    COLOR_NORMAL = (255, 150, 0)  # Blue
    COLOR_L6 = (0, 0, 255)        # Red - lumbarization
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_BG = (0, 0, 0)          # Black
    
    # Draw D12
    d12_rect = d12["min_area_rect"]
    cx = int(d12_rect["cx"])
    cy = int(d12_rect["cy"])
    w = int(d12_rect["width"])
    h = int(d12_rect["height"])
    angle = d12_rect["angle_deg_opencv"]
    
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.asarray(box, dtype=np.int32)
    
    cv2.drawContours(image, [box], 0, COLOR_D12, 3)
    cv2.circle(image, (cx, cy), 5, COLOR_D12, -1)
    
    # Label D12
    cv2.putText(
        image,
        "D12 (anchor)",
        (cx - 60, cy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_D12,
        2,
        cv2.LINE_AA
    )
    
    # Draw centerline through vertebrae
    centers = [(int(d12_rect["cx"]), int(d12_rect["cy"]))]
    for vert in lumbar_vertebrae:
        vr = vert["min_area_rect"]
        centers.append((int(vr["cx"]), int(vr["cy"])))
    
    for i in range(len(centers) - 1):
        cv2.line(image, centers[i], centers[i+1], (255, 255, 0), 2, cv2.LINE_AA)
    
    # Draw lumbar vertebrae
    for vert in lumbar_vertebrae:
        vr = vert["min_area_rect"]
        cx = int(vr["cx"])
        cy = int(vr["cy"])
        w = int(vr["width"])
        h = int(vr["height"])
        angle = vr["angle_deg_opencv"]
        
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.asarray(box, dtype=np.int32)
        
        # Color based on vertebra type
        if vert["class_name"] == "S1":
            color = COLOR_L6  # Red for S1 in lumbarization
            label = "S1"
        elif vert["class_name"] in ["L1", "L2", "L3", "L4", "L5"]:
            color = COLOR_NORMAL
            label = vert["class_name"]
        else:
            color = COLOR_OTHER
            label = vert["class_name"]
        
        # Don't show corrections in visualization - keep it clean
        
        cv2.drawContours(image, [box], 0, color, 3)
        cv2.circle(image, (cx, cy), 5, color, -1)
        
        # Label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy - 12
        
        # Background for text
        cv2.rectangle(
            image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            COLOR_BG,
            -1
        )
        
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
    
    # Add diagnosis banner at top
    lumbarization_detected = analysis["lumbarization_detected"]
    diagnosis = analysis["diagnosis"]
    lumbar_count = analysis["vertebrae_count"]
    
    banner_height = 120
    banner = np.zeros((banner_height, image.shape[1], 3), dtype=np.uint8)
    
    if lumbarization_detected:
        banner[:] = (0, 0, 100)  # Dark red
        status_text = "⚠️ LUMBARIZATION DETECTED"
    else:
        banner[:] = (0, 100, 0)  # Dark green
        status_text = "✓ NORMAL ANATOMY"
    
    cv2.putText(
        banner,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        COLOR_TEXT,
        2,
        cv2.LINE_AA
    )
    
    expected_text = "Normal: 5 (L1-L5) | Lumbarization: 6 (L1-L5-S1)"
    cv2.putText(
        banner,
        f"Vertebrae below D12: {lumbar_count} | {expected_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
        cv2.LINE_AA
    )
    
    # Combine banner and image
    result = np.vstack([banner, image])
    
    # Add legend at bottom
    legend_height = 150
    legend = np.zeros((legend_height, image.shape[1], 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)
    
    legend_items = [
        (COLOR_D12, "D12 (Anchor)"),
        (COLOR_NORMAL, "L1-L5 (Lumbar Vertebrae)"),
        (COLOR_L6, "S1 (Sacrum - indicates lumbarization)")
    ]
    
    y_pos = 30
    for color, text in legend_items:
        cv2.rectangle(legend, (20, y_pos - 10), (50, y_pos + 10), color, -1)
        cv2.putText(
            legend,
            text,
            (60, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
            cv2.LINE_AA
        )
        y_pos += 45
    
    # Combine all
    result = np.vstack([result, legend])
    
    # Save
    cv2.imwrite(str(output_path), result)
    
    return True


def process_single_image(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float = 0.5
) -> bool:
    """
    Process a single image for lumbarization detection.
    
    Returns:
        True if successful
    """
    print(f"\nProcessing: {image_path.name}")
    
    # Create output directory
    image_stem = image_path.stem
    image_output_dir = output_dir / image_stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run segmentation
    print("  → Running vertebra segmentation...")
    detections = run_segmentation(model, image_path, conf_threshold)
    
    if not detections:
        print("  ✗ No vertebrae detected")
        return False
    
    print(f"  ✓ Detected {len(detections)} vertebrae")
    
    # Find D12
    print("  → Finding D12 anchor...")
    d12 = find_d12_anchor(detections)
    
    if d12 is None:
        print("  ✗ D12 not found - cannot perform lumbarization analysis")
        return False
    
    print(f"  ✓ D12 found (confidence: {d12['confidence']:.3f})")
    
    # Find vertebrae below D12
    print("  → Identifying vertebrae below D12...")
    below_d12 = get_vertebrae_below_d12(detections, d12)
    
    if not below_d12:
        print("  ✗ No vertebrae found below D12")
        return False
    
    n_raw = len(below_d12)
    print(f"  ✓ Found {n_raw} raw detections below D12")

    # Drop duplicate boxes on the same level before counting / labeling
    below_d12 = remove_spatial_duplicates(below_d12)
    n_removed = n_raw - len(below_d12)
    if n_removed > 0:
        print(f"  ✓ After dedup: {len(below_d12)} distinct levels ({n_removed} duplicate box(es) removed)")
    else:
        print(f"  ✓ {len(below_d12)} distinct levels below D12")

    # Reorder vertebrae
    print("  → Reordering lumbar vertebrae...")
    lumbar_vertebrae = reorder_lumbar_vertebrae(below_d12)
    
    num_corrections = sum(1 for v in lumbar_vertebrae if v["corrected"])
    print(f"  ✓ Reordered vertebrae (corrected {num_corrections} labels)")
    
    # Analyze for lumbarization
    print("  → Analyzing for lumbarization...")
    analysis = analyze_lumbarization(d12, lumbar_vertebrae)
    
    print(f"  ✓ {analysis['diagnosis']}")
    
    # Save results to JSON
    results = {
        "image": str(image_path),
        "model": str(DEFAULT_MODEL),
        "output_dir": str(image_output_dir),
        "total_detections": len(detections),
        "d12_anchor": d12,
        "lumbar_vertebrae": lumbar_vertebrae,
        "lumbarization_analysis": analysis
    }
    
    json_path = image_output_dir / "lumbarization_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Results saved: {json_path}")
    
    # Create visualization
    print("  → Creating visualization...")
    viz_path = image_output_dir / "lumbarization_visualization.jpg"
    
    if create_visualization(image_path, d12, lumbar_vertebrae, analysis, viz_path):
        print(f"  ✓ Visualization saved: {viz_path}")
    else:
        print("  ✗ Visualization failed")
    
    return True


def process_batch(
    input_dir: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float = 0.5
) -> Tuple[int, int]:
    """
    Process all images in a directory.
    
    Returns:
        (successful_count, total_count)
    """
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0
    
    print(f"\nFound {len(image_files)} images to process")
    print("=" * 60)
    
    successful = 0
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        try:
            if process_single_image(image_path, model, output_dir, conf_threshold):
                successful += 1
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
    
    return successful, len(image_files)


def main():
    parser = argparse.ArgumentParser(
        description="Lumbarization detection in AP spine X-rays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python3 lumbarization_detection.py --image /path/to/xray.jpg
  
  # Batch processing
  python3 lumbarization_detection.py --image /path/to/folder/ --batch
  
  # Custom output directory
  python3 lumbarization_detection.py --image /path/to/xray.jpg --output-dir /path/to/results
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image file or directory (if --batch)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to YOLO model weights"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image path does not exist: {image_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model weights not found: {model_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    print("✓ Model loaded")
    
    # Process
    if args.batch:
        if not image_path.is_dir():
            print(f"Error: --batch requires a directory, got: {image_path}")
            sys.exit(1)
        
        successful, total = process_batch(image_path, model, output_dir, args.conf)
        
        print("\n" + "=" * 60)
        print(f"Batch processing complete: {successful}/{total} successful")
        print(f"Results saved to: {output_dir}")
        
    else:
        if not image_path.is_file():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        success = process_single_image(image_path, model, output_dir, args.conf)
        
        if success:
            print("\n✓ Processing complete")
            print(f"Results saved to: {output_dir}")
        else:
            print("\n✗ Processing failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
