import cv2
import numpy as np
import os
import glob
from skimage.morphology import disk, skeletonize, remove_small_objects
import skimage.measure 
from scipy.spatial.distance import cdist 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool, cpu_count 
import random 

# --- NEW IMPORTS for Graphing and Visualization ---
import networkx as nx
import matplotlib.pyplot as plt

# --- Color Definitions & Constants ---
SENTINEL_YELLOW = np.array([0, 255, 255], dtype=np.uint8) 
DIM_GRAY = np.array([100, 100, 100], dtype=np.uint8) 
BLACK = np.array([0, 0, 0], dtype=np.uint8) 
MAGENTA = np.array([255, 0, 255], dtype=np.uint8) 
ORANGE = np.array([0, 165, 255], dtype=np.uint8) 
INF_DISTANCE = 1e20 

# --- Graph Visualization Constants ---
RANDOM_STATE = 42 

# MAT Parameters
MAT_PRUNING_SIZE = 10 
BASE_EXCLUSION_RADIUS = 15 
MARKER_SIZE_PIXEL = 1 
MARKER_TYPE_PIXEL = cv2.MARKER_SQUARE 

# --- Set1 Color Cycle ---
SET1_COLOR_CYCLE = [
    [0, 0, 255],     # Red
    [0, 255, 0],     # Green
    [255, 0, 0],     # Blue
    [0, 165, 255],   # Orange (or Dark Yellow)
    [255, 255, 0],   # Cyan
    [255, 0, 255],   # Magenta
    [0, 255, 255],   # Yellow
    [128, 0, 0]      # Dark Blue
]

# --- Setup Paths (Unchanged) ---
def get_paths():
    base_dir = os.path.dirname(os.getcwd()) 
    input_dir = os.path.join(base_dir, 'outputs', 'final_plant_crops')
    output_dir = os.path.join(base_dir, 'outputs', 'graph_building') 
    os.makedirs(output_dir, exist_ok=True)
    mat_output_dir = os.path.join(output_dir, 'mat_segmentation') 
    os.makedirs(mat_output_dir, exist_ok=True)
    return input_dir, output_dir, mat_output_dir

# --- Geodesic Distance Calculation (Dijkstra's) (Unchanged) ---
def calculate_geodesic_distance_dijkstra(plant_mask, origin_indices):
    H, W = plant_mask.shape
    valid_indices_flat = np.where(plant_mask.flatten() == 255)[0]
    num_nodes = len(valid_indices_flat)
    flat_to_graph_map = np.full(H * W, -1, dtype=np.int32)
    flat_to_graph_map[valid_indices_flat] = np.arange(num_nodes)
    origin_node_indices = []
    if not isinstance(origin_indices, (list, np.ndarray)): origin_indices = [origin_indices]
    for r, c in origin_indices:
        if 0 <= r < H and 0 <= c < W and plant_mask[r, c] == 255:
            origin_flat_index = np.ravel_multi_index((r, c), (H, W))
            origin_node_indices.append(flat_to_graph_map[origin_flat_index])
    origin_node_indices = [idx for idx in origin_node_indices if idx != -1]
    if not origin_node_indices: return np.full(num_nodes, INF_DISTANCE, dtype=np.float64)

    row_indices, col_indices, data = [], [], [] 
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
    distances_to_nodes = distances_matrix if distances_matrix.ndim == 1 else distances_matrix.min(axis=0) 
    return distances_to_nodes 

# --- Helper to calculate distance between two points using existing Dijkstra's ---
def get_geodesic_distance(plant_mask, start_coord, end_coord):
    """Calculates the geodesic shortest path distance from start_coord to end_coord."""
    
    # 1. Start Dijkstra's from the start_coord
    distances_1d = calculate_geodesic_distance_dijkstra(plant_mask, [start_coord])
    
    H, W = plant_mask.shape
    valid_indices_flat = np.where(plant_mask.flatten() == 255)[0]
    
    # 2. Map 1D distances back to 2D for quick lookup
    distance_map = np.full((H, W), INF_DISTANCE, dtype=np.float64)
    for i, flat_idx in enumerate(valid_indices_flat):
        r, c = np.unravel_index(flat_idx, (H, W))
        if i < len(distances_1d):
            distance_map[r, c] = distances_1d[i]

    # 3. Lookup distance at the end_coord
    r_end, c_end = end_coord
    if 0 <= r_end < H and 0 <= c_end < W:
        return distance_map[r_end, c_end]
    return INF_DISTANCE # Should only happen if point is outside plant mask (r, c) or (H, W) is 0

# --- Helper for Parallel Execution (Unchanged) ---
def _calculate_single_skeleton_distance(args):
    plant_mask, r_seed, c_seed = args
    distances_1d_plant = calculate_geodesic_distance_dijkstra(plant_mask, [(r_seed, c_seed)])
    H, W = plant_mask.shape
    distance_map = np.full((H, W), INF_DISTANCE, dtype=np.float64)
    valid_indices_flat = np.where(plant_mask.flatten() == 255)[0]
    
    for i, flat_idx in enumerate(valid_indices_flat):
        r, c = np.unravel_index(flat_idx, (H, W))
        if i < len(distances_1d_plant):
            distance_map[r, c] = distances_1d_plant[i]
        
    return distance_map

# --- Pixel-to-Segment Assignment (Tie-breaking included) (Unchanged) ---
def perform_pixel_assignment(plant_mask, skeleton_coords_list, plant_coords):
    print("\n        [3] Starting **PARALLEL** Dijkstra's Geodesic Pixel-to-Skeleton Assignment (with tie-breaking)...")
    
    H, W = plant_mask.shape
    num_skeleton_points = len(skeleton_coords_list)
    
    if num_skeleton_points == 0 or len(plant_coords) == 0:
        print("        [3] Error: No skeleton or plant pixels found for assignment.")
        return None, None
    
    parallel_args = [(plant_mask, r, c) for r, c in skeleton_coords_list]
    num_cores = cpu_count()
    print(f"        [3] Distributing {num_skeleton_points} calculations across {num_cores} cores...")
    
    with Pool(num_cores) as pool:
        distance_maps = pool.map(_calculate_single_skeleton_distance, parallel_args)
    
    distance_maps_stacked = np.stack(distance_maps, axis=2)
    print("        [3] Completed Parallel Distance Calculation. Finding closest skeleton pixel...")
    
    segment_map = np.full((H, W), -1, dtype=np.int32)
    R_plant, C_plant = plant_coords[:, 0], plant_coords[:, 1]
    plant_distances = distance_maps_stacked[R_plant, C_plant, :]

    # TIE-BREAKING LOGIC
    closest_skeleton_indices = np.empty(len(plant_coords), dtype=np.int32)
    
    for i in range(len(plant_coords)):
        min_distance = np.min(plant_distances[i, :])
        tying_indices = np.where(plant_distances[i, :] == min_distance)[0]
        
        if len(tying_indices) == 1:
            closest_skeleton_indices[i] = tying_indices[0]
        else:
            # Randomly select one of the tying indices
            closest_skeleton_indices[i] = random.choice(tying_indices)
            
    # Populate the segment_map
    for i, (r, c) in enumerate(plant_coords):
        segment_map[r, c] = closest_skeleton_indices[i]
        
    print("        [3] Completed Parallel Dijkstra's Geodesic Pixel-to-Skeleton Assignment.")
    return segment_map, plant_distances 

# --- MAT Node Identification (Unchanged) ---
def identify_mat_nodes(plant_mask, origin_coords, min_length=MAT_PRUNING_SIZE, exclusion_radius=BASE_EXCLUSION_RADIUS):
    H, W = plant_mask.shape
    r_base, c_base = origin_coords
    skeleton = skeletonize(plant_mask > 0)
    skeleton_pruned = remove_small_objects(skeleton, min_size=min_length, connectivity=2)
    skeleton_pruned_uint8 = skeleton_pruned.astype(np.uint8)
    
    terminal_coords_raw, junction_coords = [], []
    skeleton_coords = np.argwhere(skeleton_pruned)
    
    for r, c in skeleton_coords:
        if 0 < r < H - 1 and 0 < c < W - 1:
            neighbor_sum = np.sum(skeleton_pruned_uint8[r-1:r+2, c-1:c+2]) - 1 
            if neighbor_sum == 1: terminal_coords_raw.append((r, c))
            elif neighbor_sum >= 3: junction_coords.append((r, c))

    terminal_coords_filtered = []
    for r, c in terminal_coords_raw:
        distance = np.sqrt((r - r_base)**2 + (c - c_base)**2)
        if distance > exclusion_radius: terminal_coords_filtered.append((r, c))
            
    anchor_coords = np.concatenate((terminal_coords_filtered, junction_coords))
    return skeleton_pruned, np.array(terminal_coords_filtered), np.array(junction_coords), anchor_coords

# --- Skeleton Path Tracing: Segment Decomposition (Unchanged) ---
def trace_skeleton_segments(skeleton_pruned, terminal_coords, junction_coords):
    H, W = skeleton_pruned.shape
    anchor_coords = np.concatenate((terminal_coords, junction_coords))
    anchor_set = set(map(tuple, anchor_coords))
    temp_skeleton = skeleton_pruned.copy()
    for r, c in anchor_coords: temp_skeleton[r, c] = False 
    labeled_skeleton = skimage.measure.label(temp_skeleton, connectivity=2)
    regions = skimage.measure.regionprops(labeled_skeleton)
    segments = []
    for i, region in enumerate(regions):
        segment_body = region.coords
        start_anchor, end_anchor = None, None
        for r_body, c_body in segment_body:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    r_neigh, c_neigh = r_body + dr, c_body + dc
                    if (r_neigh, c_neigh) in anchor_set:
                        anchor_point = (r_neigh, c_neigh)
                        if start_anchor is None: start_anchor = anchor_point
                        elif anchor_point != start_anchor: end_anchor = anchor_point; break
                if end_anchor: break
            if end_anchor: break
        if start_anchor and end_anchor and len(segment_body) > 0:
            segment_pixels = [tuple(p) for p in segment_body]
            segments.append({
                'path': [start_anchor] + segment_pixels + [end_anchor],
                'start': start_anchor,
                'end': end_anchor,
                'skeleton_body': segment_pixels 
            })
    print(f"        [4] Detected {len(segments)} discrete Line Segments.")
    return segments

# --- Feature Aggregation by Segment (Unchanged from previous update) ---
def aggregate_segment_features(segments, segment_map, skeleton_coords_list, plant_coords, original_rgb_image, anchor_coords):
    print("\n        [6] Starting Feature Aggregation (Average Color and Pixel Count)...")
    
    H, W = segment_map.shape
    
    skeleton_idx_to_info = {}
    for seg_idx, segment in enumerate(segments):
        for r, c in segment['skeleton_body']:
            match_indices = np.where((skeleton_coords_list == [r, c]).all(axis=1))[0]
            if len(match_indices) > 0: skeleton_idx = match_indices[0]; skeleton_idx_to_info[skeleton_idx] = {'type': 'segment', 'index': seg_idx}
        for anchor_coord in [segment['start'], segment['end']]:
            match_indices = np.where((skeleton_coords_list == list(anchor_coord)).all(axis=1))[0]
            if len(match_indices) > 0:
                skeleton_idx = match_indices[0]
                if skeleton_idx not in skeleton_idx_to_info: skeleton_idx_to_info[skeleton_idx] = {'type': 'anchor', 'segments': [seg_idx]}
                else: skeleton_idx_to_info[skeleton_idx]['segments'].append(seg_idx)

    segment_pixel_colors = [[] for _ in segments]
    segment_pixel_counts = [0 for _ in segments]
    segment_pixel_coords = [[] for _ in segments] 
    anchor_pixel_groups = {skeleton_idx: [] for skeleton_idx, info in skeleton_idx_to_info.items() if info['type'] == 'anchor'}

    # Tally colors and counts
    for i, (r, c) in enumerate(plant_coords):
        closest_skeleton_idx = segment_map[r, c]
        if closest_skeleton_idx in skeleton_idx_to_info:
            info = skeleton_idx_to_info[closest_skeleton_idx]
            color = original_rgb_image[r, c]
            if info['type'] == 'segment': 
                segment_pixel_colors[info['index']].append(color)
                segment_pixel_counts[info['index']] += 1
                segment_pixel_coords[info['index']].append((r, c))
            elif info['type'] == 'anchor': anchor_pixel_groups[closest_skeleton_idx].append(color)

    # Calculate initial features, including centroid
    for i, color_list in enumerate(segment_pixel_colors):
        if color_list: 
            segments[i]['avg_color'] = np.mean(color_list, axis=0).astype(np.uint8).tolist()
            coords_np = np.array(segment_pixel_coords[i])
            segments[i]['avg_r'] = np.mean(coords_np[:, 0]) # Y-axis
            segments[i]['avg_c'] = np.mean(coords_np[:, 1]) # X-axis
        else: 
            segments[i]['avg_color'] = BLACK.tolist() 
            segments[i]['avg_r'] = segments[i]['start'][0]
            segments[i]['avg_c'] = segments[i]['start'][1]

        segments[i]['pixel_count'] = segment_pixel_counts[i]

    final_skeleton_idx_to_segment_idx = {}
    
    # 1. Process Segment Body and Anchor Distributions
    for skeleton_idx, info in skeleton_idx_to_info.items():
        if info['type'] == 'segment':
            final_skeleton_idx_to_segment_idx[skeleton_idx] = info['index']
        elif info['type'] == 'anchor':
            anchor_colors = anchor_pixel_groups.get(skeleton_idx, [])
            connected_seg_indices = info['segments']
            
            if anchor_colors and connected_seg_indices:
                anchor_avg_color_np = np.mean(anchor_colors, axis=0)
                best_seg_idx, min_distance = -1, np.inf
                for seg_idx in connected_seg_indices:
                    seg_avg_color_np = np.array(segments[seg_idx]['avg_color'])
                    distance = np.linalg.norm(anchor_avg_color_np - seg_avg_color_np)
                    if distance < min_distance: min_distance, best_seg_idx = distance, seg_idx
                
                if best_seg_idx != -1:
                    current_count = segments[best_seg_idx]['pixel_count']
                    anchor_count = len(anchor_colors)
                    old_avg = np.array(segments[best_seg_idx]['avg_color'])
                    new_avg = (old_avg * current_count + anchor_avg_color_np * anchor_count) / (current_count + anchor_count)
                    
                    segments[best_seg_idx]['avg_color'] = new_avg.astype(np.uint8).tolist()
                    segments[best_seg_idx]['pixel_count'] += anchor_count
                    
                    final_skeleton_idx_to_segment_idx[skeleton_idx] = best_seg_idx
            
            if skeleton_idx not in final_skeleton_idx_to_segment_idx and connected_seg_indices:
                final_skeleton_idx_to_segment_idx[skeleton_idx] = random.choice(connected_seg_indices)
            elif skeleton_idx not in final_skeleton_idx_to_segment_idx and len(segments) > 0:
                final_skeleton_idx_to_segment_idx[skeleton_idx] = 0

    # 2. ENFORCEMENT: Ensure ALL Skeleton Indices have a Final Segment Mapping
    num_skeleton_points = len(skeleton_coords_list)
    if num_skeleton_points > 0 and len(segments) > 0:
        default_seg_index = 0
        
        for skeleton_idx in range(num_skeleton_points):
            if skeleton_idx not in final_skeleton_idx_to_segment_idx:
                r, c = skeleton_coords_list[skeleton_idx]
                neighbor_seg_indices = []
                
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        r_neigh, c_neigh = r + dr, c + dc
                        
                        if 0 <= r_neigh < H and 0 <= c_neigh < W:
                            neigh_match_indices = np.where((skeleton_coords_list == [r_neigh, c_neigh]).all(axis=1))[0]
                            
                            if len(neigh_match_indices) > 0:
                                neigh_skeleton_idx = neigh_match_indices[0]
                                if neigh_skeleton_idx in final_skeleton_idx_to_segment_idx:
                                    neighbor_seg_indices.append(final_skeleton_idx_to_segment_idx[neigh_skeleton_idx])
                        
                if neighbor_seg_indices:
                    final_skeleton_idx_to_segment_idx[skeleton_idx] = random.choice(neighbor_seg_indices)
                else:
                    final_skeleton_idx_to_segment_idx[skeleton_idx] = default_seg_index
                
    print("        [6] Completed Feature Aggregation and Color-Based Anchor Distribution.")
    return segments, final_skeleton_idx_to_segment_idx


# --- Centroid-Based Layout Generation (Unchanged) ---
def create_centroid_layout(G):
    pos = {}
    
    all_r = np.array([data['avg_r'] for node, data in G.nodes(data=True)])
    all_c = np.array([data['avg_c'] for node, data in G.nodes(data=True)])
    
    if len(all_r) == 0:
        return pos
        
    min_r, max_r = all_r.min(), all_r.max()
    min_c, max_c = all_c.min(), all_c.max()

    r_range = max_r - min_r + 1e-6
    c_range = max_c - min_c + 1e-6

    for node, data in G.nodes(data=True):
        normalized_x = (data['avg_c'] - min_c) / c_range
        normalized_y = 1.0 - ((data['avg_r'] - min_r) / r_range)
        
        pos[node] = (normalized_x, normalized_y)
        
    return pos


# --- Plant Recoloring Helper (Unchanged) ---
def create_recolored_plant_mask(plant_mask, segment_map, skeleton_coords_list, segments, skeleton_idx_to_segment_idx, output_path, color_type='avg_color'):
    H, W = plant_mask.shape
    bg_color = BLACK 
    recolored_plant_vis = np.full((H, W, 3), bg_color, dtype=np.uint8) 
    plant_coords = np.argwhere(plant_mask > 0)
    num_colors = len(SET1_COLOR_CYCLE)
    
    for r, c in plant_coords:
        closest_skeleton_idx = segment_map[r, c]
        
        if closest_skeleton_idx in skeleton_idx_to_segment_idx:
            seg_idx = skeleton_idx_to_segment_idx[closest_skeleton_idx]
            
            if color_type == 'avg_color': color = segments[seg_idx]['avg_color']
            elif color_type == 'cycle': color = SET1_COLOR_CYCLE[seg_idx % num_colors]

            recolored_plant_vis[r, c] = color
        
    cv2.imwrite(output_path, recolored_plant_vis)
    print(f"        ✅ Saved Plant Recoloring Visualization ({color_type.upper()}) to: {os.path.basename(output_path)}")

# --- Skeleton Plotting Function (Unchanged) ---
def create_segmented_skeleton_plot(H, W, plant_mask, segments, output_path):
    skeleton_vis = np.full((H, W, 3), BLACK, dtype=np.uint8) 
    plant_coords = np.argwhere(plant_mask > 0)
    for r, c in plant_coords: skeleton_vis[r, c] = DIM_GRAY 
    num_colors = len(SET1_COLOR_CYCLE)
    
    for i, segment in enumerate(segments):
        color = SET1_COLOR_CYCLE[i % num_colors].copy()
        for r, c in segment['path']:
            if plant_mask[r, c] > 0: skeleton_vis[r, c] = color 
            
    cv2.imwrite(output_path, skeleton_vis)
    print(f"        ✅ Saved Segmented Skeleton Plot to: {os.path.basename(output_path)}")

# --- Diagnostic Visualization Function (Unchanged) ---
def create_diagnostic_visualization(H, W, plant_mask, segment_map, skeleton_idx_to_segment_idx, output_path):
    diag_vis = np.full((H, W, 3), BLACK, dtype=np.uint8) 
    plant_coords = np.argwhere(plant_mask > 0)
    num_colors = len(SET1_COLOR_CYCLE)

    unassigned_count = 0
    
    for r, c in plant_coords:
        closest_skeleton_idx = segment_map[r, c]
        
        if closest_skeleton_idx == -1:
            diag_vis[r, c] = MAGENTA
            unassigned_count += 1
            continue

        if closest_skeleton_idx in skeleton_idx_to_segment_idx:
            seg_idx = skeleton_idx_to_segment_idx[closest_skeleton_idx]
            diag_vis[r, c] = SET1_COLOR_CYCLE[seg_idx % num_colors]
        else:
            diag_vis[r, c] = MAGENTA 
            unassigned_count += 1
            
    cv2.imwrite(output_path, diag_vis)
    print(f"        ⚠️ UNASSIGNED PIXEL COUNT: {unassigned_count}")
    print(f"        ✅ Saved Diagnostic Visualization to: {os.path.basename(output_path)}")
    return unassigned_count

# --- MAT Graph Building (Updated for Geodesic Weighting) ---
def build_mat_graph(segments, plant_mask, segment_map, skeleton_coords_list, skeleton_idx_to_segment_idx):
    """
    Builds the NetworkX graph using anchor connectivity and pixel proximity checks, 
    with edge weights set to the geodesic distance between segment centroids.
    """
    G = nx.Graph()
    num_segments = len(segments)
    H, W = plant_mask.shape
    
    if num_segments == 0:
        return G
        
    # 1. Add Nodes with Features (Unchanged)
    max_count = max(s['pixel_count'] for s in segments) if segments else 1

    for i, segment in enumerate(segments):
        b, g, r = segment['avg_color']
        mpl_color = (r / 255.0, g / 255.0, b / 255.0)
        node_size = 100 + (segment['pixel_count'] / max_count) * 3900 if max_count > 0 else 100

        G.add_node(i, 
                   name=f'MAT_Seg_{i}', 
                   avg_color_bgr=segment['avg_color'], 
                   mpl_color=mpl_color,
                   size=node_size,
                   pixel_count=segment['pixel_count'],
                   start_anchor=segment['start'],
                   end_anchor=segment['end'],
                   avg_r=segment['avg_r'],
                   avg_c=segment['avg_c']
                  )
    
    # 2. Add Edges based on shared anchors
    anchor_to_segments = {}
    for i, segment in enumerate(segments):
        for anchor in [segment['start'], segment['end']]:
            if anchor in anchor_to_segments: anchor_to_segments[anchor].append(i)
            else: anchor_to_segments[anchor] = [i]

    for anchor, seg_indices in anchor_to_segments.items():
        if len(seg_indices) > 1:
            for i in range(len(seg_indices)):
                for j in range(i + 1, len(seg_indices)):
                    u, v = seg_indices[i], seg_indices[j]
                    
                    # --- NEW: Calculate Geodesic Weight ---
                    u_center = (int(segments[u]['avg_r']), int(segments[u]['avg_c']))
                    v_center = (int(segments[v]['avg_r']), int(segments[v]['avg_c']))
                    geodesic_dist = get_geodesic_distance(plant_mask, u_center, v_center)
                    # ------------------------------------

                    if not G.has_edge(u, v):
                         G.add_edge(u, v, type='anchor', weight=geodesic_dist, shared_anchor=anchor)

    # 3. Add Edges based on Pixel Proximity
    print("        [7] Performing pixel proximity check for boundary adjacency...")
    
    # 3a. Create a final pixel-to-segment ID map
    final_seg_id_map = np.full((H, W), -1, dtype=np.int32)
    plant_coords = np.argwhere(plant_mask > 0)
    
    for r, c in plant_coords:
        skel_idx = segment_map[r, c]
        if skel_idx in skeleton_idx_to_segment_idx:
             final_seg_id_map[r, c] = skeleton_idx_to_segment_idx[skel_idx]

    # 3b. Iterate over plant pixels and check neighbors
    proximity_edges = set()
    
    for r, c in plant_coords:
        seg_id_u = final_seg_id_map[r, c]
        if seg_id_u == -1: continue

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r_neigh, c_neigh = r + dr, c + dc
                
                if 0 <= r_neigh < H and 0 <= c_neigh < W:
                    seg_id_v = final_seg_id_map[r_neigh, c_neigh]
                    
                    if seg_id_v != -1 and seg_id_u != seg_id_v:
                        u, v = min(seg_id_u, seg_id_v), max(seg_id_u, seg_id_v)
                        proximity_edges.add((u, v))

    # 3c. Add proximity edges to the graph
    newly_added_edges = 0
    for u, v in proximity_edges:
        
        if not G.has_edge(u, v):
            # --- NEW: Calculate Geodesic Weight (Only if edge doesn't exist) ---
            u_center = (int(segments[u]['avg_r']), int(segments[u]['avg_c']))
            v_center = (int(segments[v]['avg_r']), int(segments[v]['avg_c']))
            geodesic_dist = get_geodesic_distance(plant_mask, u_center, v_center)
            # ------------------------------------
            
            G.add_edge(u, v, type='proximity', weight=geodesic_dist, shared_anchor=None)
            newly_added_edges += 1

    print(f"        [7] Found and added {newly_added_edges} new proximity-based edges.")
    print(f"        ✅ NetworkX MAT Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- MAT Graph Plotting Function (Standalone - Updated Dynamic Width) ---
def create_mat_graph_plot(G, output_path, output_basename):
    if G.number_of_nodes() == 0:
        print(f"        [!] Cannot plot: Graph has no nodes.")
        return

    # 1. Get features for plotting
    node_colors = [data['mpl_color'] for node, data in G.nodes(data=True)]
    node_sizes = [data['size'] for node, data in G.nodes(data=True)]
    
    # --- Edge Thickness based on inverse distance (Weighting) ---
    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    
    # Filter out infinite distances and find max weight
    finite_weights = [w for w in edge_weights if w < INF_DISTANCE]
    max_weight = max(finite_weights) if finite_weights else 1.0 # Avoid div by zero/inf
    
    # Scale: Thickness = 0.5 (min) + 4.5 * (1 - normalized distance)
    # Shorter distances (small weight) get thicker lines (closer to 5.0)
    edge_widths = [0.5 + 4.5 * (1 - (w / max_weight)) if w < INF_DISTANCE else 0.5 
                   for w in edge_weights]
    # ----------------------------------------------------

    # 2. Set up figure and layout
    plt.figure(figsize=(10, 10))
    pos = create_centroid_layout(G) 

    # 3. Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_colors, 
                           node_size=node_sizes, 
                           edgecolors='black', 
                           linewidths=1.5)

    # 4. Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths) 
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"        ✅ Saved Standalone NetworkX Graph to: {os.path.basename(output_path)}")

# --- Integrated Subplot Visualization (Updated Dynamic Width) ---
def create_integrated_subplot(G, original_rgb_image_path, skeleton_plot_path, recolored_plot_path, output_path, output_basename):
    
    # 1. Load image visualizations (must exist from previous steps)
    try:
        rgb_img = cv2.cvtColor(cv2.imread(original_rgb_image_path), cv2.COLOR_BGR2RGB)
        skeleton_img = cv2.cvtColor(cv2.imread(skeleton_plot_path), cv2.COLOR_BGR2RGB)
        recolored_img = cv2.cvtColor(cv2.imread(recolored_plot_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"        [!] Error loading visualization files for subplot: {e}")
        return

    # 2. Setup Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # --- Top Left: Original RGB ---
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].axis('off')
    
    # --- Top Right: Segmented Skeleton ---
    axes[0, 1].imshow(skeleton_img)
    axes[0, 1].axis('off')
    
    # --- Bottom Left: Recolored Plant (AvgColor) ---
    axes[1, 0].imshow(recolored_img)
    axes[1, 0].axis('off')

    # --- Bottom Right: NetworkX Graph ---
    ax_graph = axes[1, 1]
    
    if G.number_of_nodes() > 0:
        node_colors = [data['mpl_color'] for node, data in G.nodes(data=True)]
        node_sizes = [data['size'] for node, data in G.nodes(data=True)]

        # --- Edge Thickness based on inverse distance (Weighting) ---
        edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
        finite_weights = [w for w in edge_weights if w < INF_DISTANCE]
        max_weight = max(finite_weights) if finite_weights else 1.0
        edge_widths = [0.5 + 4.5 * (1 - (w / max_weight)) if w < INF_DISTANCE else 0.5 
                       for w in edge_weights]
        # ----------------------------------------------------

        pos = create_centroid_layout(G) 
        
        nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                               node_color=node_colors, 
                               node_size=node_sizes, 
                               edgecolors='black', 
                               linewidths=1.5)
        
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='gray', width=edge_widths) 
        
        ax_graph.axis('off')
    else:
        ax_graph.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 1]) 
    plt.savefig(output_path)
    plt.close()
    print(f"        ✅ Saved Integrated Subplot to: {os.path.basename(output_path)}")


# --- Main Pipeline (Graph Integration) ---
def mat_segmentation_focus_pipeline():
    input_dir, output_dir, mat_output_dir = get_paths()
    
    print(f"Starting MAT Core Setup and Graph Visualization (v64-Geodesic-Weighting).")
    print("-" * 100)

    for plant_mask_path in glob.glob(os.path.join(input_dir, "*_crop_plant_mask.tif")):
        
        output_basename = os.path.basename(plant_mask_path).replace('_crop_plant_mask.tif', '')
        sentinel_mask_path = os.path.join(input_dir, f"{output_basename}_crop_sentinel_mask.tif")
        rgb_image_path = os.path.join(input_dir, f"{output_basename}_crop_rgb.tif")
        
        plant_mask_raw = cv2.imread(plant_mask_path, cv2.IMREAD_GRAYSCALE)
        sentinel_mask_color = cv2.imread(sentinel_mask_path)
        original_rgb_image = cv2.imread(rgb_image_path) 
        
        if plant_mask_raw is None or sentinel_mask_color is None or original_rgb_image is None:
             print(f"    [!] SKIPPING: Missing input files for {output_basename}.")
             continue
        
        H, W = plant_mask_raw.shape
        print(f"\n--- Processing: {output_basename} (Size: {W}x{H})")
        
        r_c, c_c = plant_mask_raw.shape[0]//2, plant_mask_raw.shape[1]//2
        origin_index_tuple = (r_c, c_c) 
        
        # 1. MAT Segmentation
        skeleton_pruned, terminal_coords_raw, junction_coords_raw, anchor_coords_raw = identify_mat_nodes(plant_mask_raw, origin_index_tuple)
        segments = trace_skeleton_segments(skeleton_pruned, terminal_coords_raw, junction_coords_raw)
        
        skeleton_coords_list = np.argwhere(skeleton_pruned)
        plant_coords = np.argwhere(plant_mask_raw > 0)
        
        segment_map, plant_distances = perform_pixel_assignment(
            plant_mask_raw, skeleton_coords_list, plant_coords
        )
        
        if segment_map is None: continue

        segments_with_features, skeleton_idx_to_segment_idx = aggregate_segment_features(
            segments, segment_map, skeleton_coords_list, plant_coords, original_rgb_image, anchor_coords_raw
        )
        
        # --- DIAGNOSTICS & VISUALIZATION (Image Outputs) ---
        
        recolored_avg_path = os.path.join(mat_output_dir, f"{output_basename}_Plant_Recolored_AvgColor.tif")
        segmented_skeleton_path = os.path.join(mat_output_dir, f"{output_basename}_Segmented_Skeleton.tif")

        create_diagnostic_visualization(
            H, W, plant_mask_raw, segment_map, skeleton_idx_to_segment_idx,
            os.path.join(mat_output_dir, f"{output_basename}_Unassigned_Pixels_Diag.tif")
        )

        create_recolored_plant_mask(
            plant_mask_raw, segment_map, skeleton_coords_list, segments_with_features,
            skeleton_idx_to_segment_idx, recolored_avg_path, color_type='avg_color'
        )
        
        create_recolored_plant_mask(
            plant_mask_raw, segment_map, skeleton_coords_list, segments_with_features,
            skeleton_idx_to_segment_idx, 
            os.path.join(mat_output_dir, f"{output_basename}_Plant_Recolored_Cycle.tif"),
            color_type='cycle'
        )
        
        create_segmented_skeleton_plot(
            H, W, plant_mask_raw, segments_with_features, segmented_skeleton_path
        )
        
        # --- 2. GRAPH GENERATION (Passing extra maps for proximity check) ---
        G = build_mat_graph(segments_with_features, plant_mask_raw, segment_map, skeleton_coords_list, skeleton_idx_to_segment_idx)
        
        # --- 3. GRAPH VISUALIZATION ---
        
        # Standalone Graph
        create_mat_graph_plot(
            G, 
            os.path.join(mat_output_dir, f"{output_basename}_NetworkX_Graph_Standalone.png"), 
            output_basename
        )

        # Integrated Subplot
        create_integrated_subplot(
            G, 
            rgb_image_path, 
            segmented_skeleton_path, 
            recolored_avg_path,
            os.path.join(mat_output_dir, f"{output_basename}_Integrated_Subplot.png"),
            output_basename
        )

    print("-" * 100)
    print("MAT Setup and Graph Visualization Complete (v64-Geodesic-Weighting).")


if __name__ == "__main__":
    mat_segmentation_focus_pipeline()