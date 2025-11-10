import cv2
import numpy as np
import os
import glob
from skimage.feature import peak_local_max 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.morphology import disk, skeletonize, remove_small_objects
import skimage.measure 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.cm as cm 

# --- Color Definitions & Constants ---
SENTINEL_YELLOW = np.array([0, 255, 255], dtype=np.uint8) 
BACKGROUND_GRAY = np.array([180, 180, 180], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLUE = np.array([255, 0, 0], dtype=np.uint8)     
RED = np.array([0, 0, 255], dtype=np.uint8) # For MAT tips visualization
INF_DISTANCE = 1e20 

# --- Tunable Parameters ---
MAT_PRUNING_SIZE = 10 # Prune small branches from the skeleton
BASE_EXCLUSION_RADIUS = 15 # Exclude any detected tip within this radius of the base centroid
MORP_KERNEL_ERODE = disk(1)

# --- Setup Paths ---
def get_paths():
    base_dir = os.path.dirname(os.getcwd()) 
    input_dir = os.path.join(base_dir, 'outputs', 'final_plant_crops')
    output_dir = os.path.join(base_dir, 'outputs', 'geodesic_features')
    os.makedirs(output_dir, exist_ok=True)
    mat_output_dir = os.path.join(output_dir, 'mat_debug')
    os.makedirs(mat_output_dir, exist_ok=True)
    return input_dir, output_dir, mat_output_dir

# --- CORE CALCULATION: Dijkstra's Shortest Path (Same as V14/V15) ---
def calculate_geodesic_distance_dijkstra(plant_mask, origin_indices):
    """Calculates Geodesic Distance using Dijkstra's Shortest Path Algorithm from one or more origins."""
    H, W = plant_mask.shape
    valid_indices_flat = np.where(plant_mask.flatten() == 255)[0]
    num_nodes = len(valid_indices_flat)
    flat_to_graph_map = np.full(H * W, -1, dtype=np.int32)
    flat_to_graph_map[valid_indices_flat] = np.arange(num_nodes)
    
    origin_node_indices = []
    
    if isinstance(origin_indices, tuple) and len(origin_indices) == 2 and isinstance(origin_indices[0], int):
        r, c = origin_indices
        origin_flat_index = np.ravel_multi_index((r, c), (H, W))
        origin_node_indices.append(flat_to_graph_map[origin_flat_index])
    else:
        for r, c in origin_indices:
            origin_flat_index = np.ravel_multi_index((r, c), (H, W))
            origin_node_indices.append(flat_to_graph_map[origin_flat_index])
    
    origin_node_indices = [idx for idx in origin_node_indices if idx != -1]
    if not origin_node_indices: return np.full((H, W), INF_DISTANCE, dtype=np.float64)

    row_indices = []
    col_indices = []
    data = [] 
    
    for i in range(num_nodes):
        r, c = np.unravel_index(valid_indices_flat[i], (H, W))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue 
                r_neigh, c_neigh = r + dr, c + dc
                
                if 0 <= r_neigh < H and 0 <= c_neigh < W and plant_mask[r_neigh, c_neigh] == 255:
                    neigh_flat_index = np.ravel_multi_index((r_neigh, c_neigh), (H, W))
                    neigh_node_index = flat_to_graph_map[neigh_flat_index]
                    weight = np.sqrt(dr**2 + dc**2)
                    row_indices.append(i)
                    col_indices.append(neigh_node_index)
                    data.append(weight)

    graph_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    distances_matrix = shortest_path(graph_matrix, indices=origin_node_indices) 
    distances_to_nodes = distances_matrix.flatten() 
    
    distance_map = np.full((H, W), INF_DISTANCE, dtype=np.float64)
    for i in range(num_nodes):
        r, c = np.unravel_index(valid_indices_flat[i], (H, W))
        distance_map[r, c] = distances_to_nodes[i] 

    return distance_map


# --- MAT and Terminal Point Extraction (Refined) ---
def extract_mat_terminal_points(plant_mask, origin_coords, min_length=MAT_PRUNING_SIZE, exclusion_radius=BASE_EXCLUSION_RADIUS):
    """
    Calculates the skeleton, prunes small spurs, finds terminal points, 
    and filters out points near the base origin.
    """
    # 1. Skeletonize and Prune
    skeleton = skeletonize(plant_mask > 0)
    skeleton_pruned = remove_small_objects(skeleton, min_size=min_length, connectivity=2)
    
    # 2. Find Terminal Points (Neighbor count = 1)
    # 3x3 kernel counts the point itself + neighbors. Terminal point count = 2.
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_counts = cv2.filter2D(skeleton_pruned.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    # Terminal points are those on the skeleton with a convolution count of exactly 2.
    terminal_mask = (skeleton_pruned > 0) & (neighbor_counts == 2)
    tips_coords = np.argwhere(terminal_mask)
    
    # 3. Filter tips near the base (Exclusion)
    r_base, c_base = origin_coords
    filtered_tips = []
    
    for r, c in tips_coords:
        distance = np.sqrt((r - r_base)**2 + (c - c_base)**2)
        if distance > exclusion_radius:
            filtered_tips.append((r, c))
            
    return skeleton_pruned, np.array(filtered_tips)


# --- Visualization Helper (Same as V15) ---
def visualize_feature_map(data_map, plant_mask, output_path, 
                          tips_coords=None, origin_coords_viz=None, title="Feature Map"):
    
    valid_mask_plant = plant_mask == 255
    final_valid_data = data_map[valid_mask_plant]
    H, W = plant_mask.shape
    
    finite_data = final_valid_data[final_valid_data < INF_DISTANCE]
    if len(finite_data) == 0: return

    min_val = finite_data.min()
    max_val = finite_data.max()
    normalized_data = np.zeros((H, W), dtype=float)
    if max_val != min_val:
        norm_values = (final_valid_data - min_val) / (max_val - min_val)
        normalized_data[valid_mask_plant] = np.clip(norm_values, 0, 1)

    normalized_data_uint8 = (normalized_data * 255).astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(normalized_data_uint8, cv2.COLORMAP_MAGMA)
    final_color_output = np.full((H, W, 3), BACKGROUND_GRAY, dtype=np.uint8)
    plant_mask_3d = np.stack([plant_mask] * 3, axis=2)
    final_color_output = np.where(plant_mask_3d > 0, color_mapped_image, final_color_output).astype(np.uint8)

    if tips_coords is not None and len(tips_coords) > 0:
        for r, c in tips_coords:
            cv2.circle(final_color_output, (c, r), 5, WHITE.tolist(), -1) 
    
    if origin_coords_viz is not None:
        cv2.drawMarker(final_color_output, (origin_coords_viz[1], origin_coords_viz[0]), BLUE.tolist(), 
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    cv2.imwrite(output_path, final_color_output)
    print(f"    ✅ Saved visual map for {title} to: {os.path.basename(output_path)}")


# --- Main Pipeline ---
def geodesic_pipeline():
    input_dir, output_dir, mat_output_dir = get_paths()
    
    print(f"Starting Final Geodesic Pipeline (v16): MAT Terminal Point Refinement.")
    print("-" * 100)

    for plant_mask_path in glob.glob(os.path.join(input_dir, "*_crop_plant_mask.tif")):
        
        output_basename = os.path.basename(plant_mask_path).replace('_crop_plant_mask.tif', '')
        sentinel_mask_path = os.path.join(input_dir, f"{output_basename}_crop_sentinel_mask.tif")
        rgb_image_path = os.path.join(input_dir, f"{output_basename}_crop_rgb.tif")
        
        plant_mask_raw = cv2.imread(plant_mask_path, cv2.IMREAD_GRAYSCALE)
        sentinel_mask_color = cv2.imread(sentinel_mask_path)
        rgb_image = cv2.imread(rgb_image_path)
        
        if plant_mask_raw is None or sentinel_mask_color is None or rgb_image is None:
             print(f"    [!] SKIPPING: Missing input files for {output_basename}.")
             continue
        
        H, W = plant_mask_raw.shape
        print(f"\n--- Processing: {output_basename} (Size: {W}x{H})")

        # --- STEP 1: Define Origin Point ---
        kernel_erode = MORP_KERNEL_ERODE.astype(np.uint8)
        yellow_origin_mask_raw = cv2.inRange(sentinel_mask_color, SENTINEL_YELLOW, SENTINEL_YELLOW)
        yellow_origin_mask_eroded = cv2.erode(yellow_origin_mask_raw, kernel_erode, iterations=1)
        base_region_mask = yellow_origin_mask_eroded if np.any(yellow_origin_mask_eroded) else yellow_origin_mask_raw
        
        if not np.any(base_region_mask): continue
             
        props = skimage.measure.regionprops(base_region_mask)[0]
        r_c, c_c = int(props.centroid[0]), int(props.centroid[1])
        origin_coords_viz = (r_c, c_c) 
        origin_index_tuple = (r_c, c_c)
        if not plant_mask_raw[r_c, c_c] == 255: continue
            
        print("    [1] Defined single-point base origin.")

        # --- STEP 2: T_Base Calculation (Dijkstra) ---
        T_Base = calculate_geodesic_distance_dijkstra(plant_mask_raw, origin_index_tuple)
        print("    [2] Calculated T_Base using Dijkstra's algorithm.")
        
        # --- STEP 3: Tip Detection via MAT/Skeletonization (Refined) ---
        skeleton_pruned, tips_coords = extract_mat_terminal_points(plant_mask_raw, origin_index_tuple)

        # Output MAT visualization for inspection
        mat_vis_output = np.full((H, W, 3), BACKGROUND_GRAY, dtype=np.uint8)
        mat_vis_output[plant_mask_raw == 255] = [100, 100, 100] # Dim gray plant
        mat_vis_output[skeleton_pruned] = WHITE.tolist() # White skeleton
        for r, c in tips_coords:
            cv2.circle(mat_vis_output, (c, r), 5, RED.tolist(), -1)
        cv2.drawMarker(mat_vis_output, (c_c, r_c), BLUE.tolist(), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        cv2.imwrite(os.path.join(mat_output_dir, f"{output_basename}_MAT_Tips.tif"), mat_vis_output)
        print(f"    [3] Detected {len(tips_coords)} MAT terminal points.")
        print(f"        ✅ Saved MAT/Tips visualization to mat_debug folder.")

        # --- STEP 4: T_Tips Calculation (Dijkstra) ---
        T_Tips = calculate_geodesic_distance_dijkstra(plant_mask_raw, tips_coords)
        print("    [4] Calculated T_Tips using Dijkstra's algorithm.")
        
        # --- STEP 5: Feature Generation (Geodesic and Color) ---
        
        # Geodesic Features
        valid_mask = T_Base < INF_DISTANCE
        T_Diff = np.full((H, W), INF_DISTANCE)
        T_Diff[valid_mask] = np.abs(T_Base[valid_mask] - T_Tips[valid_mask])
        T_Sum = np.full((H, W), INF_DISTANCE)
        T_Sum[valid_mask] = T_Base[valid_mask] + T_Tips[valid_mask]
        print("    [5] Generated T_Diff/T_Sum Maps.")

        # Color PCA Features (Same logic as V14)
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
        lab_features = lab_image[plant_mask_raw == 255, :].astype(np.float64)
        scaler = StandardScaler()
        lab_scaled = scaler.fit_transform(lab_features)
        
        pca = PCA(n_components=2) 
        color_pc_2d = pca.fit_transform(lab_scaled)
        pc1_1d = color_pc_2d[:, 0]
        pc2_1d = color_pc_2d[:, 1]

        PC1_map = np.full((H, W), np.inf)
        PC1_map[plant_mask_raw == 255] = pc1_1d
        PC2_map = np.full((H, W), np.inf)
        PC2_map[plant_mask_raw == 255] = pc2_1d
        PC_Combined_1d = np.sqrt(pc1_1d**2 + pc2_1d**2)
        PC_Combined_map = np.full((H, W), np.inf)
        PC_Combined_map[plant_mask_raw == 255] = PC_Combined_1d
        print(f"    [6] Generated PC1/PC2/Combined Maps.")
        
        # --- STEP 6: Save and Visualize ---
        
        # Save Features
        np.save(os.path.join(output_dir, f"{output_basename}_T_Base.npy"), T_Base)
        np.save(os.path.join(output_dir, f"{output_basename}_T_Tips.npy"), T_Tips)
        np.save(os.path.join(output_dir, f"{output_basename}_T_Diff.npy"), T_Diff)
        np.save(os.path.join(output_dir, f"{output_basename}_T_Sum.npy"), T_Sum)
        np.save(os.path.join(output_dir, f"{output_basename}_PC1_Color.npy"), PC1_map)
        np.save(os.path.join(output_dir, f"{output_basename}_PC2_Color.npy"), PC2_map)
        np.save(os.path.join(output_dir, f"{output_basename}_PC_Combined.npy"), PC_Combined_map)
        print(f"    ✅ Saved 7 Feature Maps to {os.path.basename(output_dir)}.")
        
        # Visualization
        visualize_feature_map(T_Base, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_T_Base.tif"), 
                              tips_coords=tips_coords, origin_coords_viz=origin_coords_viz, title="T_Base")
        visualize_feature_map(T_Tips, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_T_Tips.tif"), 
                              tips_coords=tips_coords, origin_coords_viz=origin_coords_viz, title="T_Tips")
        visualize_feature_map(T_Diff, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_T_Diff.tif"), 
                              title="T_Diff")
        visualize_feature_map(T_Sum, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_T_Sum.tif"), 
                              title="T_Sum")
        visualize_feature_map(PC1_map, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_PC1_Color.tif"), 
                              title="PC1 Color")
        visualize_feature_map(PC2_map, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_PC2_Color.tif"), 
                              title="PC2 Color")
        visualize_feature_map(PC_Combined_map, plant_mask_raw, os.path.join(output_dir, f"{output_basename}_vis_PC_Combined.tif"), 
                              title="PC Combined")

    print("-" * 100)
    print("Final Geodesic Pipeline Complete (using MAT/Dijkstra).")

if __name__ == "__main__":
    geodesic_pipeline()