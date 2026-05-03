import os
import numpy as np

BASE_EXP_RESULT = "exp/so100/semseg-pt-v3m1-0-base-only-grid/result/"
BASE_DATA_VALIDATION = "data/so100/validation"

COORD_FNAME = "Cosmos_SO100_Beyond_Success_1031_Cleaned_ep0001_fr000042.npy"
SEG_FNAME = "Cosmos_SO100_Beyond_Success_1031_Cleaned_ep0001_fr000042_pred.npy"
FIXED_LABEL_COLORS = {
    0: [1, 0, 0],      # red
    1: [0, 1, 0],      # green
    2: [0, 0, 1],      # blue
    3: [1, 1, 0],      # yellow
    4: [1, 0, 1],      # magenta
    5: [0, 1, 1],      # cyan
    6: [0.5, 0.5, 0],  # olive
}


def visualize_pointcloud_with_labels(coord, segment=None, label_colors=None, voxel_size=None, colorize: bool = True):
    """根据类别 ID 给点云上色并显示。

    Args:
        coord: [N,3]
        segment: [N] label id per point. If None and colorize=True, will raise.
        label_colors: dict[int, (r,g,b)]
        voxel_size: optional voxel downsample size.
        colorize: if False, do not set pcd.colors (no label coloring).
    """
    import open3d as o3d

    coord = np.asarray(coord)
    if segment is not None:
        segment = np.asarray(segment)

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if colorize:
        if segment is None:
            raise ValueError("colorize=True requires segment != None")

        # Generate colors for each label if not provided
        if label_colors is None:
            unique_labels = np.unique(segment)
            label_colors = {}
            colors_palette = [
                [1, 0, 0],      # red
                [0, 1, 0],      # green
                [0, 0, 1],      # blue
                [1, 1, 0],      # yellow
                [1, 0, 1],      # magenta
                [0, 1, 1],      # cyan
                [0.5, 0.5, 0],  # olive
                [0.5, 0, 0.5],  # purple
                [0, 0.5, 0.5],  # teal
            ]
            for i, label_id in enumerate(unique_labels):
                label_colors[int(label_id)] = colors_palette[i % len(colors_palette)]

        # Assign colors based on labels
        colors = np.array(
            [label_colors.get(int(label_id), [0.7, 0.7, 0.7]) for label_id in segment], dtype=np.float64
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Explicit uniform color to avoid inheriting any previous colors/visualizer state.
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

    o3d.visualization.draw_geometries([pcd])


def _load_any_npy(path):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.dtype == object:
        try:
            d = d.item()
        except Exception:
            pass
    return d


def load_coord_and_segments(coord_path, seg_path):
    coord_data = _load_any_npy(coord_path)
    seg_data = _load_any_npy(seg_path)

    if isinstance(coord_data, dict):
        coord = coord_data.get("coord")
        segment_gt = coord_data.get("segment")
    else:
        coord = coord_data
        segment_gt = None

    if isinstance(seg_data, dict):
        segment = seg_data.get("segment")
    else:
        segment = seg_data

    coord = np.asarray(coord)
    segment = np.asarray(segment)
    if segment_gt is not None:
        segment_gt = np.asarray(segment_gt)

    return coord, segment, segment_gt


if __name__ == "__main__":
    seg_path = os.path.join(BASE_EXP_RESULT, SEG_FNAME)
    coord_path = os.path.join(BASE_DATA_VALIDATION, COORD_FNAME)
    coord, segment, segment_gt = load_coord_and_segments(coord_path, seg_path)

    print("Visualizing raw point cloud (no coloring) ...")
    visualize_pointcloud_with_labels(coord, segment=None, voxel_size=0.005, colorize=False)

    print("Visualizing predicted labels ...")
    visualize_pointcloud_with_labels(
        coord, segment, label_colors=FIXED_LABEL_COLORS, voxel_size=0.005, colorize=True
    )

    if segment_gt is not None:
        print("Visualizing GT labels ...")
        visualize_pointcloud_with_labels(
            coord, segment_gt, label_colors=FIXED_LABEL_COLORS, voxel_size=0.005, colorize=True
        )
