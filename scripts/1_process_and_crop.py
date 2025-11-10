import cv2
import numpy as np
import os
import glob
import sys
import re

# --- Sentinel Color Definitions (BGR format) ---
SENTINEL_BLUE = np.array([255, 0, 0], dtype=np.uint8)         
SENTINEL_MAGENTA = np.array([255, 0, 255], dtype=np.uint8)    
SENTINEL_CYAN = np.array([255, 255, 0], dtype=np.uint8)       
SENTINEL_YELLOW = np.array([0, 255, 255], dtype=np.uint8)     
TRUE_BLACK = np.array([0, 0, 0], dtype=np.uint8)

# --- Path Configuration ---
def get_paths():
    base_dir = os.path.dirname(os.getcwd()) 
    input_annotated_dir = os.path.join(base_dir, 'annotations')
    
    # *** CORRECTED INPUT DIRECTORY: Reverting to ./color_pics/ and looking for JPG ***
    input_original_rgb_dir = os.path.join(base_dir, 'color_pics')
    
    output_dir = os.path.join(base_dir, 'outputs', 'final_plant_crops') 
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(input_original_rgb_dir):
        print(f"CRITICAL: Required input directory not found: {input_original_rgb_dir}. Please create it or adjust path.")
        sys.exit(1)
        
    return input_annotated_dir, input_original_rgb_dir, output_dir

# --- Helper Function for Clean Mask Generation (Contour Subtraction) ---
def create_exclusive_mask(image, color):
    """Creates an exclusive mask of an enclosed area defined by a single color."""
    raw_mask = cv2.inRange(image, color, color)
    contours, _ = cv2.findContours(raw_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    inclusive_mask = np.zeros_like(raw_mask)
    cv2.drawContours(inclusive_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    exclusive_mask = cv2.subtract(inclusive_mask, raw_mask)
    exclusive_mask[exclusive_mask > 0] = 255
    
    return exclusive_mask, raw_mask

def consolidate_and_crop_final():
    input_annotated_dir, input_original_rgb_dir, output_dir = get_paths()
    
    print(f"Starting Final Consolidation: Cropping and Saving Required Assets.")
    print("-" * 70)

    for annotated_path in glob.glob(os.path.join(input_annotated_dir, "p*_*.tif")):
        filename = os.path.basename(annotated_path)
        output_basename = filename.replace('.tif', '')
            
        annotated_image = cv2.imread(annotated_path)
        if annotated_image is None: continue
        
        print(f"\n--- Processing: {filename}")
        
        # --- PHASE 1: Isolation and Component Identification ---
        
        # Isolation: Get the plant component mask (Logic from previous successful steps)
        interior_mask_exclusive, blue_mask_raw = create_exclusive_mask(annotated_image, SENTINEL_BLUE)
        if interior_mask_exclusive is None:
            print("    [1] WARNING: Could not generate exclusive blue ring mask. Skipping.")
            continue
            
        temp_isolated_image = np.full_like(annotated_image, TRUE_BLACK, dtype=np.uint8)
        mask_3d = np.stack([interior_mask_exclusive] * 3, axis=2)
        temp_isolated_image = np.where(mask_3d > 0, annotated_image, temp_isolated_image).astype(np.uint8)
        
        blue_mask_inverted_3d = np.stack([cv2.bitwise_not(blue_mask_raw)] * 3, axis=2)
        temp_isolated_image = np.where(blue_mask_inverted_3d > 0, temp_isolated_image, TRUE_BLACK).astype(np.uint8)
        
        # Component Finding and Bounding Box
        gray_image = cv2.cvtColor(temp_isolated_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
        if num_labels <= 1: continue
            
        max_area = 0
        largest_label = 1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                largest_label = i
                
        x, y, w, h, area = stats[largest_label]
        bbox = (x, y, w, h)
        final_plant_mask_raw = (labels == largest_label).astype(np.uint8) * 255
        print(f"    [1] Plant BBox identified: {bbox}")

        
        # --- PHASE 2: Cropping and Saving (Final Requirements) ---
        
        # CORRECTION 1: Load p007.jpg from ./color_pics/
        match = re.match(r"(p\d+)", output_basename)
        if not match: continue
            
        original_file_prefix = match.group(1)
        original_color_path = os.path.join(input_original_rgb_dir, f"{original_file_prefix}.jpg")
        original_color_image = cv2.imread(original_color_path)
        
        if original_color_image is None:
            print(f"    [2] CRITICAL: Original RGB image not found at {original_color_path}. Skipping save.")
            continue
            
        # Define crop coordinates: (y_start:y_end, x_start:x_end)
        x_start, y_start, width, height = bbox
        x_end, y_end = x_start + width, y_start + height
        
        # Get cropped versions of the two primary inputs
        cropped_plant_mask_raw = final_plant_mask_raw[y_start:y_end, x_start:x_end]
        cropped_original_rgb = original_color_image[y_start:y_end, x_start:x_end]


        # Output A: p007_01_crop_plant_mask.tif (Required)
        cv2.imwrite(os.path.join(output_dir, f"{output_basename}_crop_plant_mask.tif"), cropped_plant_mask_raw)
        print("    [3] Saved A: Cropped Plant Mask (Binary).")


        # Output B: p007_01_crop_rgb.tif (Required, Isolated RGB)
        # Apply the binary mask (cropped_plant_mask_raw) to the cropped original RGB image.
        mask_3d_cropped = np.stack([cropped_plant_mask_raw] * 3, axis=2)
        
        isolated_rgb = np.full_like(cropped_original_rgb, TRUE_BLACK, dtype=np.uint8)
        isolated_rgb = np.where(mask_3d_cropped > 0, cropped_original_rgb, isolated_rgb).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"{output_basename}_crop_rgb.tif"), isolated_rgb)
        print("    [4] Saved B: Cropped Original RGB Image (Isolated).")


        # Output C: p007_01_crop_sentinel_mask.tif (Required, Retaining Sentinel Color)
        
        # 1. Isolate the M, C, Y colors from the original annotated image
        magenta_mask = cv2.inRange(annotated_image, SENTINEL_MAGENTA, SENTINEL_MAGENTA)
        cyan_mask = cv2.inRange(annotated_image, SENTINEL_CYAN, SENTINEL_CYAN)
        yellow_mask = cv2.inRange(annotated_image, SENTINEL_YELLOW, SENTINEL_YELLOW)
        
        # 2. Combine all sentinel masks to get a full M/C/Y binary mask
        sentinel_binary_mask = cv2.bitwise_or(magenta_mask, cyan_mask)
        sentinel_binary_mask = cv2.bitwise_or(sentinel_binary_mask, yellow_mask)
        
        # 3. Apply the final plant mask to ensure only sentinels on the plant are considered
        sentinel_on_plant_mask = cv2.bitwise_and(sentinel_binary_mask, final_plant_mask_raw)
        
        # 4. Create the final color image (BLACK background)
        sentinel_color_output = np.full_like(annotated_image, TRUE_BLACK, dtype=np.uint8)
        
        # 5. Only copy the sentinel pixels from the original annotation where the mask is TRUE
        sentinel_mask_3d = np.stack([sentinel_on_plant_mask] * 3, axis=2)
        sentinel_color_output = np.where(
            sentinel_mask_3d > 0, 
            annotated_image, # Use the original annotated image to get the color (Magenta, Cyan, Yellow)
            sentinel_color_output
        ).astype(np.uint8)

        # 6. Crop and save the color sentinel image
        cropped_sentinel_color = sentinel_color_output[y_start:y_end, x_start:x_end]
        cv2.imwrite(os.path.join(output_dir, f"{output_basename}_crop_sentinel_mask.tif"), cropped_sentinel_color)
        print("    [5] Saved C: Cropped Sentinel Mask (M/C/Y, Retaining Color).")


        # Output D: (p007_01_crop_annotated_isolated.tif is now skipped as requested)
        # print("    [6] Skipped annotated isolated image.")


    print("-" * 70)
    print("All final assets saved successfully.")

if __name__ == "__main__":
    consolidate_and_crop_final()