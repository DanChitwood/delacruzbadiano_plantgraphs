import cv2
import numpy as np
import os
import glob
from collections import defaultdict

def preprocess_true_black_tif():
    # --- 1. Define Paths and Parameters ---
    
    # Assumes script is run from a 'scripts' folder, base_dir is the parent folder
    base_dir = os.path.dirname(os.getcwd())  
    input_dir = os.path.join(base_dir, 'color_pics')
    # Output folder name as requested
    output_dir = os.path.join(base_dir, 'outputs', 'processed_images') 
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Adaptive Thresholding Parameters (V5 settings)
    BLOCK_SIZE = 21  
    C_CONSTANT = 3
    KERNEL_SIZE = 5
    # Target background color (BGR format)
    TRUE_BLACK = np.array([0, 0, 0], dtype="uint8") 

    print(f"Starting Simplified Preprocessing: RGB Plant on True Black Background.")
    print(f"Output folder: {output_dir}")
    print("-" * 40)

    # --- 2. Batch Processing Loop ---
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            # Use the base name (e.g., 'p007') and append '.tif'
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.tif"

            try:
                # Load original color image (BGR)
                img_bgr = cv2.imread(input_path)
                if img_bgr is None: continue

                # a. Generate the high-fidelity mask (V5 logic)
                img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                L_channel = img_lab[:, :, 0] 

                threshold_mask_inverted = cv2.adaptiveThreshold(
                    L_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, C_CONSTANT)

                # b. Morphological Closing
                dilated_mask = cv2.dilate(threshold_mask_inverted, kernel, iterations=1)
                final_plant_mask = cv2.erode(dilated_mask, kernel, iterations=1)
                
                mask_8bit = final_plant_mask.astype(np.uint8) 
                
                # --- Create the Final RGB Image with True Black Background ---
                
                # Invert the mask to get the background mask (Background = 255)
                background_mask = cv2.bitwise_not(mask_8bit)
                background_mask_3d = np.stack([background_mask] * 3, axis=2)
                
                # Set background area to the TRUE BLACK color
                img_output = np.where(
                    background_mask_3d.astype(bool), 
                    TRUE_BLACK, 
                    img_bgr
                ).astype(np.uint8)
                
                # Save as a 3-channel (RGB/BGR) TIFF
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, img_output)
                print(f"✅ Saved True Black RGB Base: {os.path.basename(output_path)}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    print("-" * 40)
    print("Preprocessing to True Black TIFF complete. Use this file for manual annotation.")

if __name__ == "__main__":
    preprocess_true_black_tif()