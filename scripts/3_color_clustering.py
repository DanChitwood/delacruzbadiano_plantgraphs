import cv2
import numpy as np
import os
import glob
import skimage.measure 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path 
import networkx as nx
import matplotlib.pyplot as plt

# --- Required Imports for Clustering ---
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# --- Constants & Parameters ---
SENTINEL_YELLOW = np.array([0, 255, 255], dtype=np.uint8) 
BACKGROUND_GRAY = np.array([180, 180, 180], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8) 

# Clustering Parameters
K_RANGE = range(2, 8) # Test k from 2 to 7
RANDOM_STATE = 42
NN_ITERATIONS = 2 # Number of times to run the smoothing operation
INF_DISTANCE = 1e20 # For graph algorithms

# --- Setup Paths ---
def get_paths():
    base_dir = os.path.dirname(os.getcwd()) 
    input_dir = os.path.join(base_dir, 'outputs', 'final_plant_crops')
    output_dir = os.path.join(base_dir, 'outputs', 'graph_building') 
    os.makedirs(output_dir, exist_ok=True)
    color_output_dir = os.path.join(output_dir, 'color_clustering')
    os.makedirs(color_output_dir, exist_ok=True)
    return input_dir, output_dir, color_output_dir

# --- Geodesic Distance Calculation (Dijkstra's) (Required for Edge Weights) ---
def calculate_geodesic_distance_dijkstra(plant_mask, origin_indices):
    H, W = plant_mask.shape
    valid_indices_flat = np.where(plant_mask.flatten() > 0)[0]
    num_nodes = len(valid_indices_flat)
    flat_to_graph_map = np.full(H * W, -1, dtype=np.int32)
    flat_to_graph_map[valid_indices_flat] = np.arange(num_nodes)
    origin_node_indices = []
    if not isinstance(origin_indices, (list, np.ndarray)): origin_indices = [origin_indices]
    for r, c in origin_indices:
        if 0 <= r < H and 0 <= c < W and plant_mask[r, c] > 0:
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
                if 0 <= r_neigh < H and 0 <= c_neigh < W and plant_mask[r_neigh, c_neigh] > 0:
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

def get_geodesic_distance(plant_mask, start_coord, end_coord):
    """Calculates the geodesic shortest path distance from start_coord to end_coord."""
    distances_1d = calculate_geodesic_distance_dijkstra(plant_mask, [start_coord])
    H, W = plant_mask.shape
    valid_indices_flat = np.where(plant_mask.flatten() > 0)[0]
    distance_map = np.full((H, W), INF_DISTANCE, dtype=np.float64)
    for i, flat_idx in enumerate(valid_indices_flat):
        r, c = np.unravel_index(flat_idx, (H, W))
        if i < len(distances_1d): distance_map[r, c] = distances_1d[i]
    r_end, c_end = end_coord
    if 0 <= r_end < H and 0 <= c_end < W: return distance_map[r_end, c_end]
    return INF_DISTANCE

# --- Core Color Clustering Logic (Unchanged) ---
def determine_optimal_k_and_cluster(rgb_pixels):
    """
    Finds the optimal number of clusters (k) for color segmentation.
    """
    if len(rgb_pixels) < 100:
        optimal_k = 3
    else:
        optimal_k = K_RANGE.start
    
    lab_pixels = cv2.cvtColor(rgb_pixels[None, :, :], cv2.COLOR_BGR2LAB)[0]
    
    # ... (K-means optimization logic unchanged) ...
    if len(rgb_pixels) >= 100:
        best_score = -1
        print("    [C1] Testing K values for optimal clustering...")
        for k in K_RANGE:
            try:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto', max_iter=200)
                labels = kmeans.fit_predict(lab_pixels)
                if len(set(labels)) > 1:
                    score = silhouette_score(lab_pixels, labels)
                    if score > best_score:
                        best_score = score
                        optimal_k = k
            except ValueError:
                continue
    
    print(f"    [C2] Optimal k detected: {optimal_k} (Score: {best_score:.4f} if calculated).")

    # Final clustering
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init='auto', max_iter=200)
    labels = kmeans.fit_predict(lab_pixels)
    canonical_lab_colors = kmeans.cluster_centers_
    canonical_bgr_colors = cv2.cvtColor(canonical_lab_colors[None].astype(np.uint8), cv2.COLOR_LAB2BGR)[0]
    
    return labels, optimal_k, canonical_bgr_colors

# --- Nearest Neighbor Cleanup Logic (Unchanged) ---
def apply_nearest_neighbor_cleanup(plant_mask, initial_labels_1d, optimal_k, plant_coords):
    """
    Applies nearest neighbor majority voting to smooth color cluster labels.
    """
    H, W = plant_mask.shape
    labeled_map = np.full((H, W), -1, dtype=np.int32)
    labeled_map[plant_coords[:, 0], plant_coords[:, 1]] = initial_labels_1d
    current_map = labeled_map.copy()
    
    print(f"    [C3] Applying {NN_ITERATIONS} iterations of Nearest Neighbor cleanup...")
    
    for iteration in range(NN_ITERATIONS):
        next_map = current_map.copy()
        for r, c in plant_coords:
            r_start, r_end = max(0, r - 1), min(H, r + 2)
            c_start, c_end = max(0, c - 1), min(W, c + 2)
            neighbor_labels = current_map[r_start:r_end, c_start:c_end].flatten()
            valid_neighbors = neighbor_labels[neighbor_labels != -1]
            if len(valid_neighbors) > 0:
                counts = np.bincount(valid_neighbors)
                majority_label = np.argmax(counts)
                if current_map[r, c] != majority_label:
                    next_map[r, c] = majority_label
        current_map = next_map
        print(f"        -> Iteration {iteration + 1} complete.")

    cleaned_labels_1d = current_map[plant_coords[:, 0], plant_coords[:, 1]]
    print(f"    [C3] Nearest Neighbor cleanup complete.")
    return cleaned_labels_1d, current_map 

# --- Visualization Function (Unchanged) ---
def recolor_plant_by_cluster(plant_mask, original_rgb, cluster_labels, canonical_bgr_colors, output_basename, output_dir, k, suffix=""):
    """
    Recolors the plant using the canonical colors of its assigned cluster.
    """
    recolored_img = np.full_like(original_rgb, BACKGROUND_GRAY, dtype=np.uint8)
    plant_pixel_indices = np.argwhere(plant_mask > 0)
    for i, (r, c) in enumerate(plant_pixel_indices):
        label = cluster_labels[i]
        recolored_img[r, c] = canonical_bgr_colors[label]
    output_path = os.path.join(output_dir, f"{output_basename}_Recolored_k_{k}{suffix}.tif")
    cv2.imwrite(output_path, recolored_img)
    print(f"    [C4] Recolored visualization saved ({suffix.strip('_')}) to: {os.path.basename(output_path)}")

# --- Connected Component Averaging Visualization Function (Unchanged) ---
def recolor_plant_by_connected_component_average(plant_mask, original_rgb, cleaned_labels_2d, output_basename, output_dir, k):
    """
    Recolors each connected component with the *average original RGB color* of the pixels it contains.
    """
    H, W, C = original_rgb.shape
    recolored_img = np.full_like(original_rgb, BACKGROUND_GRAY, dtype=np.uint8)
    plant_area_mask = cleaned_labels_2d != -1
    unique_labels = np.unique(cleaned_labels_2d[plant_area_mask])
    
    for color_class in unique_labels:
        if color_class == -1: continue
            
        class_mask = cleaned_labels_2d == color_class
        # Find connected components within this color class
        component_labels, num_components = skimage.measure.label(class_mask, connectivity=2, background=0, return_num=True)
        
        for i in range(1, num_components + 1):
            component_mask = component_labels == i
            original_pixels = original_rgb[component_mask]
            
            if original_pixels.size > 0:
                avg_color = np.mean(original_pixels, axis=0).astype(np.uint8)
                recolored_img[component_mask] = avg_color
    
    output_path = os.path.join(output_dir, f"{output_basename}_Recolored_k_{k}_CCA.tif")
    cv2.imwrite(output_path, recolored_img)
    
    print(f"    [C5] CCA visualization saved to: {os.path.basename(output_path)}")
    
# --- Centroid-Based Layout Generation (Unchanged) ---
def create_centroid_layout(G):
    pos = {}
    
    all_r = np.array([data['avg_r'] for node, data in G.nodes(data=True)])
    all_c = np.array([data['avg_c'] for node, data in G.nodes(data=True)])
    
    if len(all_r) == 0: return pos
        
    min_r, max_r = all_r.min(), all_r.max()
    min_c, max_c = all_c.min(), all_c.max()

    r_range = max_r - min_r + 1e-6
    c_range = max_c - min_c + 1e-6

    for node, data in G.nodes(data=True):
        normalized_x = (data['avg_c'] - min_c) / c_range
        normalized_y = 1.0 - ((data['avg_r'] - min_r) / r_range)
        
        pos[node] = (normalized_x, normalized_y)
        
    return pos

# --- CORRECTED: Color Cluster Graph Builder (Nodes = Connected Components) ---
def build_color_cluster_graph(plant_mask, original_rgb, cleaned_labels_2d):
    """
    Builds a NetworkX graph where NODES are the Connected Components (CCs) 
    of the color clusters (Regions), matching the MAT graph structure.
    """
    G = nx.Graph()
    H, W = plant_mask.shape
    
    print("\n    [C6] Identifying Connected Components (CCs) for Node Definition...")
    
    # 1. Identify and extract features for all connected components (Nodes)
    unique_labels = np.unique(cleaned_labels_2d[cleaned_labels_2d != -1])
    # Map the entire image space pixel to its corresponding graph node index
    global_component_map = np.full((H, W), -1, dtype=np.int32)
    next_node_index = 0

    for color_class in unique_labels:
        class_mask = cleaned_labels_2d == color_class
        # Find CCs within this color class
        component_labels, num_components = skimage.measure.label(class_mask, connectivity=2, background=0, return_num=True)
        
        for i in range(1, num_components + 1):
            component_mask = component_labels == i
            component_coords = np.argwhere(component_mask)
            
            if component_coords.size > 0:
                original_pixels = original_rgb[component_mask]
                
                # Node Features: Average color and centroid of the region
                avg_color_bgr = np.mean(original_pixels, axis=0).astype(np.uint8).tolist()
                avg_r, avg_c = np.mean(component_coords, axis=0) # Centroid
                pixel_count = component_coords.shape[0]
                
                # BGR to Matplotlib RGB, normalized
                mpl_color = np.array(avg_color_bgr)[::-1] / 255.0
                
                # Add node
                G.add_node(next_node_index,
                           cluster_id=int(color_class), # The original K-means cluster ID
                           name=f'CC_{next_node_index}_K{int(color_class)}',
                           avg_color_bgr=avg_color_bgr,
                           mpl_color=mpl_color,
                           avg_r=avg_r,
                           avg_c=avg_c,
                           pixel_count=pixel_count)
                
                # Map the component area to the global node index
                global_component_map[component_mask] = next_node_index
                next_node_index += 1

    print(f"    [C6] Identified {G.number_of_nodes()} connected components (Nodes).")
    
    # 2. Add Edges based on 3x3 Pixel Proximity
    print("    [C7] Checking 3x3 adjacency for edges between CCs...")
    edges = set()
    plant_coords = np.argwhere(plant_mask > 0)
    
    for r, c in plant_coords:
        node_u = global_component_map[r, c]
        if node_u == -1: continue

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r_neigh, c_neigh = r + dr, c + dc
                
                if 0 <= r_neigh < H and 0 <= c_neigh < W:
                    node_v = global_component_map[r_neigh, c_neigh]
                    
                    # If two adjacent pixels belong to two different CCs (nodes)
                    if node_v != -1 and node_u != node_v:
                        u, v = min(node_u, node_v), max(node_u, node_v)
                        edges.add((u, v))

    # Add edges with Geodesic Weighting
    for u, v in edges:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        u_center = (int(u_data['avg_r']), int(u_data['avg_c']))
        v_center = (int(v_data['avg_r']), int(v_data['avg_c']))
        
        geodesic_dist = get_geodesic_distance(plant_mask, u_center, v_center)
        
        G.add_edge(u, v, weight=geodesic_dist, type='proximity')

    print(f"    [C7] Added {G.number_of_edges()} adjacency edges.")
    return G

# --- Graph Visualization Function (Fixed Pathing) ---
def create_color_graph_plots(G, original_rgb_path, raw_recolored_path, cca_recolored_path, output_basename, color_output_dir):
    
    # Load images
    rgb_img = cv2.cvtColor(cv2.imread(original_rgb_path), cv2.COLOR_BGR2RGB)
    raw_recolored_bgr = cv2.imread(raw_recolored_path)
    cca_recolored_bgr = cv2.imread(cca_recolored_path)
    
    # Create black background versions
    H, W, C = rgb_img.shape
    raw_black_bg = np.full((H, W, 3), BLACK, dtype=np.uint8)
    cca_black_bg = np.full((H, W, 3), BLACK, dtype=np.uint8)
    
    raw_mask = np.any(raw_recolored_bgr != BACKGROUND_GRAY, axis=2)
    raw_black_bg[raw_mask] = raw_recolored_bgr[raw_mask]
    
    cca_mask = np.any(cca_recolored_bgr != BACKGROUND_GRAY, axis=2)
    cca_black_bg[cca_mask] = cca_recolored_bgr[cca_mask]

    # Convert to RGB for Matplotlib
    raw_plot_img = cv2.cvtColor(raw_black_bg, cv2.COLOR_BGR2RGB)
    cca_plot_img = cv2.cvtColor(cca_black_bg, cv2.COLOR_BGR2RGB)

    # 1. Setup Graph Plot features
    max_count = max(data.get('pixel_count', 1) for node, data in G.nodes(data=True)) if G.number_of_nodes() > 0 else 1
    
    # Node features are stored directly in the G.nodes data structure
    node_colors = [data['mpl_color'] for node, data in G.nodes(data=True)]
    node_sizes = [100 + (data.get('pixel_count', 1) / max_count) * 3900 for node, data in G.nodes(data=True)]

    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    finite_weights = [w for w in edge_weights if w < INF_DISTANCE]
    max_weight = max(finite_weights) if finite_weights else 1.0
    edge_widths = [0.5 + 4.5 * (1 - (w / max_weight)) if w < INF_DISTANCE else 0.5 
                   for w in edge_weights]
    
    pos = create_centroid_layout(G)
    
    # 2. Create Subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Top Left: Original RGB
    axes[0, 0].imshow(rgb_img)
    #axes[0, 0].set_title("A. Original RGB Image")
    axes[0, 0].axis('off')
    
    # Top Right: Raw K-Means (Black BG)
    axes[0, 1].imshow(raw_plot_img)
    #axes[0, 1].set_title(f"B. Raw K-Means (k={G.nodes[0]['cluster_id'] if G.number_of_nodes()>0 else '?'})")
    axes[0, 1].axis('off')
    
    # Bottom Left: CCA Recoloring (Black BG)
    axes[1, 0].imshow(cca_plot_img)
    #axes[1, 0].set_title("C. Connected Component Averaging (CCA)")
    axes[1, 0].axis('off')

    # Bottom Right: NetworkX Graph
    ax_graph = axes[1, 1]
    
    if G.number_of_nodes() > 0:
        nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                               node_color=node_colors, 
                               node_size=node_sizes, 
                               edgecolors='black', 
                               linewidths=1.5)
        
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='gray', width=edge_widths) 
        
        #ax_graph.set_title(f"D. Color Region Graph (Nodes={G.number_of_nodes()} Regions)")
        ax_graph.axis('off')
    else:
        ax_graph.set_title("D. Graph: No Components Found")
        ax_graph.axis('off')
        
    plt.tight_layout() 
    
    # Save Subplot
    subplot_path = os.path.join(color_output_dir, f"{output_basename}_ColorGraph_Integrated_Subplot.png")
    plt.savefig(subplot_path)
    plt.close(fig)
    print(f"    ✅ Saved Integrated Subplot to: {os.path.basename(subplot_path)}")

    # Save Standalone Graph
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths)
    plt.axis('off')
    standalone_path = os.path.join(color_output_dir, f"{output_basename}_ColorGraph_Standalone.png")
    plt.savefig(standalone_path)
    plt.close()
    print(f"    ✅ Saved Standalone Graph to: {os.path.basename(standalone_path)}")


# --- Main Pipeline (Finalized) ---
def color_clustering_pipeline():
    input_dir, output_dir, color_output_dir = get_paths()
    
    print(f"Starting Global Color Clustering Graphing Pipeline (v65-Region-Node-Final).")
    print("-" * 100)

    for plant_mask_path in glob.glob(os.path.join(input_dir, "*_crop_plant_mask.tif")):
        
        output_basename = os.path.basename(plant_mask_path).replace('_crop_plant_mask.tif', '')
        rgb_image_path = os.path.join(input_dir, f"{output_basename}_crop_rgb.tif")
        
        plant_mask = cv2.imread(plant_mask_path, cv2.IMREAD_GRAYSCALE)
        original_rgb = cv2.imread(rgb_image_path)
        
        if plant_mask is None or original_rgb is None:
             print(f"\n    [!] SKIPPING: Missing input files for {output_basename}.")
             continue
        
        print(f"\n--- Processing: {output_basename}")
        
        # 1. Extract masked pixels
        plant_pixel_mask = plant_mask > 0
        original_rgb_pixels = original_rgb[plant_pixel_mask]
        plant_coords = np.argwhere(plant_mask > 0)
        
        if len(original_rgb_pixels) == 0:
            print("    [!] SKIPPING: Plant mask is empty.")
            continue

        # 2. K-Means Clustering
        raw_labels_1d, optimal_k, canonical_bgr_colors = determine_optimal_k_and_cluster(original_rgb_pixels)
        
        # 3. Apply Nearest Neighbor Cleanup
        cleaned_labels_1d, cleaned_labels_2d = apply_nearest_neighbor_cleanup(
            plant_mask, raw_labels_1d, optimal_k, plant_coords
        )

        # 4. Generate Visualization Files (Used for the subplot)
        raw_recolored_path = os.path.join(color_output_dir, f"{output_basename}_Recolored_k_{optimal_k}_Raw.tif")
        cca_recolored_path = os.path.join(color_output_dir, f"{output_basename}_Recolored_k_{optimal_k}_CCA.tif")
        
        recolor_plant_by_cluster(plant_mask, original_rgb, raw_labels_1d, canonical_bgr_colors, output_basename, color_output_dir, optimal_k, suffix="_Raw")
        recolor_plant_by_connected_component_average(plant_mask, original_rgb, cleaned_labels_2d, output_basename, color_output_dir, optimal_k)

        # 5. BUILD GRAPH (Nodes = Connected Components/Regions)
        G = build_color_cluster_graph(plant_mask, original_rgb, cleaned_labels_2d)

        # 6. VISUALIZE GRAPH (Standalone and Integrated Subplot)
        create_color_graph_plots(
            G, 
            rgb_image_path, 
            raw_recolored_path, 
            cca_recolored_path, 
            output_basename, 
            color_output_dir # Corrected path
        )

    print("-" * 100)
    print(f"Color Clustering Graphing Complete. Please review the output images.")


if __name__ == "__main__":
    color_clustering_pipeline()