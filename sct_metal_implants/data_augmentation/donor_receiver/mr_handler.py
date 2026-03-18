from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, generate_binary_structure, binary_erosion, \
    binary_dilation, binary_opening
from scipy.ndimage import distance_transform_edt
import numpy as np
from skimage.measure import label, regionprops

class MRHandler:
    """Handle MR void extraction and synthesis."""

    def __init__(self):
        self.relative_void_threshold = 0.1

    def extract_mr_void(self, mr_volume, implant_mask, spacing, body_mask,
                        threshold=0.1, radius_mm=20, min_void_volume_voxels=500):
        """Extract MR void using intensity, distance, and morphology.
        Parameters:
            mr_volume (ndarray) -- MR volume
            implant_mask (ndarray) -- binary implant mask
            spacing (tuple) -- voxel spacing (Z, Y, X)
            body_mask (ndarray) -- binary body mask
            threshold (float) -- intensity threshold
            radius_mm (float) -- search radius in mm
            min_void_volume_voxels (int) -- minimum region size
        Returns:
            ndarray -- cleaned void mask
        """
        # Normalize MR only within body
        body_voxels = mr_volume[body_mask > 0]
        min_val, max_val = body_voxels.min(), body_voxels.max()
        norm_mr = np.zeros_like(mr_volume, dtype=np.float32)
        norm_mr[body_mask > 0] = (mr_volume[body_mask > 0] - min_val) / (max_val - min_val + 1e-6)

        # Distance transform from implant
        dist = distance_transform_edt(~implant_mask.astype(bool), sampling=spacing)
        within_radius = dist <= radius_mm

        # Void = low intensity within radius
        void_mask = (norm_mr < threshold) & within_radius

        # Morphological smoothing
        structure = generate_binary_structure(3, 1)
        void_mask = binary_opening(void_mask, structure=structure)
        void_mask = binary_closing(void_mask, structure=structure)
        void_mask = gaussian_filter(void_mask.astype(np.float32), sigma=1) > 0.5

        # Label and filter regions
        labeled = label(void_mask)
        cleaned_mask = np.zeros_like(void_mask, dtype=np.uint8)

        for region in regionprops(labeled):
            region_mask = (labeled == region.label)
            if region.area >= min_void_volume_voxels and np.any(implant_mask[region_mask]):
                cleaned_mask[region_mask] = 1

        # Final smoothing
        cleaned_mask = binary_opening(cleaned_mask, structure=structure)
        cleaned_mask = binary_closing(cleaned_mask, structure=structure)
        cleaned_mask = gaussian_filter(cleaned_mask.astype(np.float32), sigma=1) > 0.5

        return cleaned_mask.astype(np.uint8)

    def _normalize(self, img):
        """Normalize image to [0,1]."""
        return (img - img.min()) / (img.ptp() + 1e-6)

    def calculate_mr_void(self, volume, void_mask, implant_mask, method="standard"):
        """Synthesize MR void values.
        Parameters:
            volume (ndarray) -- MR volume
            void_mask (ndarray) -- binary void mask
            implant_mask (ndarray) -- binary implant mask
            method (str) -- synthesis method: standard | overwrite_min_intensity |
                            overwrite_gradient_based | sample_from_distribution
        Returns:
            ndarray -- modified or synthetic void volume
        """
        # Overwrite with minimum intensity uniformly
        if method == "overwrite_min_intensity":
            min_intensity = volume[implant_mask > 0].min()
            void_only = np.zeros_like(volume)
            void_only[void_mask > 0] = max(min_intensity, 10)
            return void_only

        # Overwrite with a gradient (10, 50)
        elif method == "overwrite_gradient_based":

            min_val = 10
            max_val = 50

            distance = distance_transform_edt(~implant_mask.astype(bool), sampling=(2.0, 1.5625, 1.5625))
            distance[~void_mask.astype(bool)] = 0  # only keep values inside the void region

            max_distance = distance.max()
            if max_distance > 0:
                normalized = distance / max_distance
            else:
                normalized = distance

            # Invert distance → darkest at implant, brightest at edge
            gradient = min_val + normalized * (max_val - min_val)

            # Outside void region: set to 0 or np.nan
            gradient[~void_mask.astype(bool)] = 0

            return gradient

        # Overwrite by sampling from MR intensities within donor void
        elif method == "sample_from_distribution":

            distance = distance_transform_edt(~implant_mask.astype(bool), sampling=(2.0, 1.5625, 1.5625))
            distributions = {}

            # Implant itself
            distributions['implant'] = volume[implant_mask > 0]

            # Shells
            range_distance = 5
            number_of_shells = 5

            for i in range(number_of_shells):
                min_mm = i * range_distance
                max_mm = (i + 1) * range_distance
                mask = (distance >= min_mm) & (distance < max_mm)
                values = volume[mask]
                distributions[f"{min_mm}-{max_mm}mm"] = values

            synthetic_void = np.zeros_like(volume, dtype=np.float32)

            # Fill implant region
            implant_coords = np.argwhere(implant_mask > 0)
            if len(implant_coords) > 0:
                sampled_vals = np.random.choice(distributions['implant'], size=len(implant_coords), replace=True)
                for idx, val in zip(implant_coords, sampled_vals):
                    synthetic_void[tuple(idx)] = val

            # Fill shells
            for i in range(number_of_shells):
                min_mm = i * range_distance
                max_mm = (i + 1) * range_distance
                shell_mask = (distance >= min_mm) & (distance < max_mm) & (void_mask > 0)

                coords = np.argwhere(shell_mask)
                key = f"{min_mm}-{max_mm}mm"

                sampled_vals = np.random.choice(distributions[key], size=len(coords), replace=True)
                for idx, val in zip(coords, sampled_vals):
                    synthetic_void[tuple(idx)] = val

            return synthetic_void

        # Default approach, using identical values as in donor
        return volume * void_mask



