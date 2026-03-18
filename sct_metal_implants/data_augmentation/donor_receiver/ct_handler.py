from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, binary_closing
import numpy as np
from skimage.measure import label, regionprops

class CTHandler:
    """Utility class for extracting body and implant masks from CT volumes."""

    def __init__(self):
        self.body_mask_threshold = -400
        self.implant_mask_threshold = 2000

    def get_body_mask(self, ct_volume):
        """Get body mask from CT volume.
        Parameters:
            ct_volume (ndarray) -- input CT volume
        Returns:
            ndarray -- binary mask of the body
        """
        binary_mask = ct_volume > self.body_mask_threshold
        for i in range(binary_mask.shape[0]):
            binary_mask[i] = binary_fill_holes(binary_mask[i])

        labeled = label(binary_mask, connectivity=1)
        props = regionprops(labeled)
        if not props:
            return np.zeros_like(ct_volume, dtype=np.uint8)

        largest = max(props, key=lambda r: r.area)
        body_mask = (labeled == largest.label)
        body_mask = binary_erosion(body_mask, iterations=1)
        body_mask = binary_dilation(body_mask, iterations=3)

        return body_mask.astype(np.uint8)

    def get_implant_mask(self, volume, body_mask, structure_mask):
        """Get implant mask for a single structure.
        Parameters:
            volume (ndarray)         -- CT volume
            body_mask (ndarray)      -- binary body mask
            structure_mask (ndarray) -- binary mask of the structure
        Returns:
            ndarray -- binary implant mask
        """
        implant_mask = np.zeros_like(volume, dtype=bool)

        binary_mask = (volume > self.implant_mask_threshold) & body_mask
        labeled_volume = label(binary_mask, connectivity=1)
        regions = regionprops(labeled_volume)

        for region in regions:
            if region.area > 100:
                region_mask = (labeled_volume == region.label)
                if np.sum(structure_mask & region_mask) > 0:
                    mask = region_mask  # & structure_mask
                    mask = self._fill_mask(mask)
                    implant_mask = implant_mask | mask

        return implant_mask

    def _fill_mask(self, mask, iterations=2):
        """Fill and close holes in a mask.
        Parameters:
            mask (ndarray)   -- binary mask
            iterations (int) -- closing iterations
        Returns:
            ndarray -- processed mask
        """
        filled = binary_fill_holes(mask)
        closed = binary_closing(filled, iterations=iterations)
        return closed.astype(np.uint8)

    def get_femur_head_mask(self, structure_mask):
        """
        Get femoral head, which is cut off in THR surgery.
        """

        print(structure_mask.shape)

    def cut_femur_by_fraction(self, femur_mask, axis, fraction = 0.6, spacing = (2.0, 1.5625, 1.5625)):

        femur_mask = femur_mask.astype(bool)
        coords = np.argwhere(femur_mask)  # (N,3) in (z,y,x)
        if coords.size == 0:
            z = np.zeros_like(femur_mask, dtype=bool)
            return z, z, None, None

        n = np.asarray(axis, dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

        if spacing is None:
            coords_p = coords.astype(float)
        else:
            sp = np.asarray(spacing, dtype=float)  # (sz,sy,sx)
            coords_p = coords * sp

        proj = coords_p @ n
        pmin, pmax = float(proj.min()), float(proj.max())
        L = max(pmax - pmin, 1e-6)

        # Decide which end is proximal (head end usually has fewer voxels in a thin slab)
        slab = 0.05 * L
        count_min = np.sum(proj <= pmin + slab)
        count_max = np.sum(proj >= pmax - slab)
        proximal_is_max = count_max < count_min

        # Cut position: remove proximal 'fraction' along axis
        if proximal_is_max:
            cutoff = pmax - fraction * L
            keep = proj < cutoff  # keep distal
        else:
            cutoff = pmin + fraction * L
            keep = proj > cutoff  # keep distal

        kept_mask = np.zeros_like(femur_mask, dtype=bool)
        removed_mask = np.zeros_like(femur_mask, dtype=bool)

        kept_coords = coords[keep]
        removed_coords = coords[~keep]

        kept_mask[kept_coords[:, 0], kept_coords[:, 1], kept_coords[:, 2]] = True
        removed_mask[removed_coords[:, 0], removed_coords[:, 1], removed_coords[:, 2]] = True

        return femur_mask * kept_mask, femur_mask * removed_mask
