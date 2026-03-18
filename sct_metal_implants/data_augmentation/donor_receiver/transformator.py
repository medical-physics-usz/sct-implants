import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform


class Transformator:
    """Align and transform structures between donor and recipient."""

    def __init__(self, donor_structure, recipient_structure):
        self.rotation = None
        self.translation = None
        self.rotation_center = None
        self.align_structures(donor_structure, recipient_structure)

    def align_structures(self, donor_structure, recipient_structure):
        """Align donor to recipient by PCA and center matching.
        Parameters:
            donor_structure (ndarray)       -- donor binary mask of structure (eg. FemurHead)
            recipient_structure (ndarray)   -- recipient binary mask of structure (eg. FemurHead)
        """

        donor_axis = self.compute_pca_alignment(donor_structure)
        recipient_axis = self.compute_pca_alignment(recipient_structure)

        donor_center = self.calculate_center(donor_structure)
        recipient_center = self.calculate_center(recipient_structure)

        if np.dot(donor_axis, recipient_axis) < 0:
            donor_axis *= -1  # Flip direction

        R = self.rotation_matrix_from_vectors(donor_axis, recipient_axis)
        T = recipient_center - donor_center
        self.set_rotation_and_translation(R, T)
        self.set_rotation_center(donor_center)

    def compute_pca_alignment(self, mask):
        """Compute main axis with PCA.
        Parameters:
            mask (ndarray) -- binary mask
        Returns:
            ndarray -- axis vector
        """
        coords = np.argwhere(mask > 0)
        pca = PCA(n_components=3)
        pca.fit(coords)
        axis = pca.components_[0]  # femoral shaft direction
        return axis

    def calculate_center(self, mask):
        """Calculate center of a mask.
        Parameters:
            mask (ndarray) -- binary mask
        Returns:
            ndarray -- center coordinate
        """
        coords = np.argwhere(mask > 0)
        center = coords.mean(axis=0)
        return center

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """Compute rotation matrix from vec1 to vec2.
        Parameters:
            vec1 (ndarray) -- source vector (donor)
            vec2 (ndarray) -- target vector (receiver)
        Returns:
            ndarray -- rotation matrix (3x3)
        """
        v = np.cross(vec1, vec2)
        c = np.dot(vec1, vec2)
        if np.allclose(v, 0) or np.isclose(c, 1):
            return np.eye(3)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
        return R

    def transform(self, volume_to_transform, volume_to_get_center=None):
        """Apply stored transform to volume.
        Parameters:
            volume_to_transform (ndarray)       -- volume to transform
            volume_to_get_center (ndarray|None) -- optional reference for center
        Returns:
            ndarray -- transformed volume
        """
        if not volume_to_get_center:
            center = self.rotation_center
        else:
            center = self.calculate_center(volume_to_get_center)
        transformed_volume = self.apply_affine_to_volume(volume_to_transform, center)
        return transformed_volume

    def apply_affine_to_volume(self, volume, center):
        """Apply affine transform to volume.
        Parameters:
            volume (ndarray) -- volume to transform
            center (ndarray) -- rotation center
        Returns:
            ndarray -- transformed volume
        """
        # Build composite transform: translate to origin -> rotate-> translate to target
        affine_mat = np.eye(4)
        affine_mat[:3, :3] = self.rotation
        affine_mat[:3, 3] = self.translation + center - self.rotation @ center  # shift to keep center fixed

        inv_mat = np.linalg.inv(affine_mat)
        return affine_transform(volume, inv_mat[:3, :3], offset=inv_mat[:3, 3], output_shape=volume.shape, order=1)

    def set_rotation_and_translation(self, R, T):
        self.rotation = R
        self.translation = T

    def set_rotation_center(self, center):
        self.rotation_center = center
