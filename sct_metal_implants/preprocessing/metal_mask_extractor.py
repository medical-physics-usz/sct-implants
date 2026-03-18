import plotly.graph_objects as go
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, binary_closing, distance_transform_edt, \
    generate_binary_structure, binary_opening, gaussian_filter
from skimage import measure


class MetalMaskExtractor:
    """Extracts metal (candidate) masks from CT/MR volumes."""

    def __init__(self):
        self.MIN_VOXEL_SIZE = 5000
        self.MR_IN_INTENSITY_THRESHOLD = 0.025
        self.CT_INTENSITY_THRESHOLD = 2000

    def extract_void(self, volume, body_mask):
        """Extract MR void regions (metal candidate mask).
        Parameters:
            volume (ndarray): MR volume
            body_mask (ndarray): binary body mask
        Returns:
            ndarray: binary void mask
        """
        void_mask = self.mask_low_intensity_for_volume(volume, body_mask, "MR")

        # Morphological smoothing
        structure = generate_binary_structure(3, 1)
        void_mask = binary_opening(void_mask, structure=structure)
        void_mask = binary_closing(void_mask, structure=structure)
        void_mask = gaussian_filter(void_mask.astype(np.float32), sigma=1) > 0.5

        # Label and filter regions
        labeled = measure.label(void_mask)
        cleaned_mask = np.zeros_like(void_mask, dtype=np.uint8)

        for region in measure.regionprops(labeled):
            region_mask = (labeled == region.label)
            if region.area >= self.MIN_VOXEL_SIZE and self.region_has_valid_flatness_ratio(region):

                # Widen voids
                smoothed_region_mask = binary_dilation(region_mask, structure=structure, iterations=2)  # expand
                smoothed_region_mask = binary_closing(smoothed_region_mask, structure=structure)  # fill gaps

                if self.region_is_not_hollow(smoothed_region_mask):
                    cleaned_mask[region_mask] = 1

        # Smooth edges
        cleaned_mask = gaussian_filter(cleaned_mask.astype(np.float32), sigma=1) > 0.5
        return (cleaned_mask > 0).astype(np.float32)

    def get_metal_mask_CT(self, volume, body_mask, min_region=100):
        """Extract CT metal mask by thresholding.
        Parameters:
            volume (ndarray): CT volume
            body_mask (ndarray): binary body mask
            min_region (int): minimum region size
        Returns:
            ndarray: binary CT metal mask
        """
        metal_mask = np.zeros_like(volume, dtype=bool)

        binary_mask = (volume > self.CT_INTENSITY_THRESHOLD) & body_mask
        labeled_volume = measure.label(binary_mask, connectivity=1)
        regions = measure.regionprops(labeled_volume)

        for region in regions:
            if region.area > min_region:
                region_mask = (labeled_volume == region.label)
                mask = region_mask  # & structure_mask
                mask = self._fill_mask(mask)
                metal_mask = metal_mask | mask

        return metal_mask.astype(np.float32)

    def mask_low_intensity_for_volume(self, volume, body_mask, modality):
        """Mask low intensity regions in CT/MR.
        Parameters:
            volume (ndarray): input volume
            body_mask (ndarray): binary body mask
            modality (str): modality ("CT" or "MR")
        Returns:
            ndarray: binary mask of low-intensity regions
        """
        mask = np.zeros_like(volume, dtype=bool)
        num_slices = volume.shape[2]

        for z in range(num_slices):
            if modality.startswith("MR"):
                normalized_slice = self.normalize(volume[:, :, z])
                mask[:, :, z] = (normalized_slice < self.MR_IN_INTENSITY_THRESHOLD) & body_mask[:, :, z]
            else:
                slice = volume[:, :, z]
                mask[:, :, z] = (slice > self.CT_INTENSITY_THRESHOLD) & body_mask[:, :, z]

        return mask

    def _fill_mask(self, mask, iterations=2):
        """Fill and close holes in mask."""
        filled = binary_fill_holes(mask)
        closed = binary_closing(filled, iterations=iterations)
        return closed.astype(np.uint8)

    def extract_connected_components(self, mask):
        """Label connected components in mask."""
        labeled_mask = measure.label(mask, connectivity=1)
        props = measure.regionprops(labeled_mask)
        return labeled_mask, props

    def normalize(self, img):
        """Normalize image to [0,1]."""
        return (img - np.min(img)) / (np.ptp(img) + 1e-6)
        # return (img - img.min()) / (img.ptp() + 1e-6)

    def extract_potential_metal_regions_from_connected_components(self, labeled_mask, props):
        """Filter connected components into potential metal regions.
        Parameters:
            labeled_mask (ndarray): labeled mask
            props (list): regionprops list
        Returns:
            ndarray: binary mask of selected regions
        """
        final_mask = np.zeros_like(labeled_mask, dtype=bool)

        for prop in props:
            if prop.area >= self.MIN_VOXEL_SIZE and self.region_has_valid_flatness_ratio(prop):
                # Create binary mask for component
                region_mask = (labeled_mask == prop.label)
                region_mask = self._fill_mask(region_mask)
                final_mask = final_mask | region_mask

        return (final_mask > 0).astype(np.float32) # when np.uint8 -> value-range 0-255

    def region_is_not_hollow(self, mask, holowness_threshold = 0.5):
        """Check whether region is not hollow.
        Parameters:
            mask (ndarray): binary region mask
            holowness_threshold (float): max allowed hollow ratio
        Returns:
            bool: True if region is not hollow
        """
        mask = (mask > 0).astype(np.bool_)
        hollow_slice_count = 0
        total_slices = 0

        for i in range(mask.shape[2]):
            slice_2d = mask[:, :, i]
            if not slice_2d.any():
                continue  # skip completely empty slices

            total_slices += 1

            filled = binary_fill_holes(slice_2d)
            hollow = filled & ~slice_2d

            if np.any(hollow):
                hollow_slice_count += 1

        ratio = hollow_slice_count / total_slices if total_slices > 0 else 0.0
        print(f"Holowness Ratio: {ratio}")
        return ratio < holowness_threshold

    def region_has_valid_flatness_ratio(self, prop, flatness_ratio_threshold=0.1):
        """Check if region is flat enough."""
        flatness_ratio = self.calculate_flatness_ratio(prop)
        print(f"Flatnesss Ratio: {flatness_ratio}")
        if flatness_ratio < flatness_ratio_threshold:
            return False
        return True

    def calculate_flatness_ratio(self, prop):
        """Calculate flatness ratio of region."""
        # Check flatness of region
        minr, minc, mind = prop.bbox[0:3]
        maxr, maxc, maxd = prop.bbox[3:6]
        dims = [maxr - minr, maxc - minc, maxd - mind]
        flatness_ratio = min(dims) / max(dims)
        return flatness_ratio

    def plot_regions(self, mask, body_mask, save_path):
        """Plot body and metal candidate regions in 3D."""
        # Prepare plot
        fig = go.Figure()

        # --- Extract body surface using marching cubes ---
        verts_b, faces_b, normals_b, _ = measure.marching_cubes(body_mask.astype(np.float32), level=0.5)
        x_b, y_b, z_b = verts_b.T
        i_b, j_b, k_b = faces_b.T

        # --- Add body mesh to the plot (transparent & light) ---
        fig.add_trace(go.Mesh3d(
            x=x_b, y=y_b, z=z_b,
            i=i_b, j=j_b, k=k_b,
            color='lightgray',
            opacity=0.15,
            name='Body Outline',
            lighting=dict(ambient=0.5, diffuse=0.3),
            lightposition=dict(x=0, y=0, z=300),
            showscale=False,
            showlegend=True
        ))

        # Get coordinates of non-zero voxels
        x, y, z = np.nonzero(mask)

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color="purple"), #marker=dict(size=2, color="red"),
            name="Metal Candidate"
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title='3D Binary Mask Visualization',
            margin=dict(l=0, r=0, b=0, t=30)
        )

        fig.write_html(save_path)

    def extract_body_mask(self, volume):
        """Extract body mask slice by slice."""
        body_mask = np.zeros_like(volume, dtype=bool)
        num_slices = volume.shape[2]

        for z in range(num_slices):
            slice_img = volume[:, :, z]
            adaptive_threshold = slice_img.max() * 0.1

            # Get body mask (per slice)
            body_mask_slice = self.get_body_mask_threshold(slice_img[:, :, np.newaxis], adaptive_threshold).squeeze()
            body_mask_slice = self.erode_body_mask(body_mask_slice, border_margin=5)
            body_mask[:, :, z] = body_mask_slice

        return body_mask

    def get_body_mask_threshold(self, volume, threshold):
        """Threshold volume and keep largest contour."""
        mask = np.zeros_like(volume)
        mask[volume > threshold] = 1
        mask[volume <= threshold] = 0
        mask = self.get_mask_biggest_contour(mask)
        mask = binary_dilation(mask, iterations=5).astype(np.int16)
        return mask

    def get_mask_biggest_contour(self, mask):
        """Keep only largest contour per slice."""
        for i in range(mask.shape[2]):
            inmask = np.expand_dims(mask[:, :, i].astype(np.uint8), axis=2)
            _, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) != 0:
                segmented = max(cnts, key=cv2.contourArea)
                bin_img[bin_img > 0] = 0
                a = cv2.drawContours(np.expand_dims(bin_img, axis=2), [segmented], 0, (255, 255, 255), -1)
                a[a > 0] = 1
                mask[:, :, i] = a.squeeze()
        return mask

    def erode_body_mask(self, body_mask, border_margin=5):
        """Erode mask to remove boundary artifacts."""
        return binary_erosion(body_mask, iterations=border_margin).astype(np.uint8)

    def extract_metal_regions(self, volume, body_mask, modality):
        """Extract potential metal regions."""
        in_phase_low_intensity_mask = self.mask_low_intensity_for_volume(volume, body_mask, modality)
        in_phase_labeled_mask, in_phase_props = self.extract_connected_components(in_phase_low_intensity_mask)

        metal_region_mask = self.extract_potential_metal_regions_from_connected_components(in_phase_labeled_mask, in_phase_props)
        return metal_region_mask