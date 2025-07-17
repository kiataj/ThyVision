import napari
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def extract_3d_patches(volume, patch_size, stride):
    """
    Extract overlapping 3D patches from a volume.

    Args:
        volume (ndarray): 3D array of shape (D, H, W)
        patch_size (tuple): Size of each patch (d, h, w)
        stride (tuple): Stride between patches (sd, sh, sw)

    Returns:
        patches (list): List of 3D patch arrays
        centers (list): List of (cz, cy, cx) coordinates corresponding to patch centers
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    centers = []

    for z in range(0, D - pd + 1, sd):
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patch = volume[z:z+pd, y:y+ph, x:x+pw]
                patches.append(patch)
                centers.append((z + pd // 2, y + ph // 2, x + pw // 2))

    return patches, centers

def infer_patchwise(encoder, classifier, volume, patch_size, stride, task_list, device="cuda"):
    encoder.eval()
    classifier.eval()
    encoder.to(device)
    classifier.to(device)

    D, H, W = volume.shape
    prob_maps = {task: np.zeros((D, H, W), dtype=np.float32) for task in task_list}
    count_map = np.zeros((D, H, W), dtype=np.uint8)

    patches, centers = extract_3d_patches(volume, patch_size, stride)

    for patch, (cz, cy, cx) in tqdm(zip(patches, centers), total=len(patches), desc="Inference"):
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)

        with torch.no_grad():
            features = encoder(patch_tensor)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            logits = classifier(features)

        # Define patch bounds
        sz, sy, sx = patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2
        z0, z1 = max(cz - sz, 0), min(cz + sz, D)
        y0, y1 = max(cy - sy, 0), min(cy + sy, H)
        x0, x1 = max(cx - sx, 0), min(cx + sx, W)

        for task in task_list:
            prob = torch.sigmoid(logits[task]).squeeze().item()
            prob_maps[task][z0:z1, y0:y1, x0:x1] += prob
        count_map[z0:z1, y0:y1, x0:x1] += 1

    for task in task_list:
        prob_maps[task] = np.divide(prob_maps[task], count_map, where=count_map > 0)

    return prob_maps

def show_prediction_napari(volume: np.ndarray, prob_maps: dict):
    """Launch Napari to view the volume and overlaid probability maps.

    Args:
        volume (np.ndarray): Input volume, shape (D, H, W).
        prob_maps (dict): Dictionary of {task_name: prob_map}, each (D, H, W).
    """
    viewer = napari.Viewer()
    viewer.add_image(
        volume,
        name="Input Volume",
        colormap="gray",
        contrast_limits=(0, 1),
        blending="additive"
    )
    for task, prob_map in prob_maps.items():
        viewer.add_image(
            prob_map,
            name=f"{task}_prob",
            colormap="magenta",
            opacity=0.5,
            contrast_limits=(0, 1),
            blending="additive"
        )
    napari.run()

