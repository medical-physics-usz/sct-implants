import numpy as np
from scipy.ndimage import map_coordinates, zoom

class GRE_Dixon3D_ForwardModel_IntraVoxelSinc:
    """
    GRE / Dixon forward model (B0 already in Hz = Δf).
    Effects:
      1) In-plane geometric distortion along readout axis (warp)
      2) Optional Jacobian pile-up/spread (usually small/zero for GRE)
      3) TE-dependent intravoxel dephasing using sinc-product model (local linear field)

    Key feature:
      - Gradients are computed on the *native B0 grid* (high-res), then resampled to Dixon grid.
    """

    def __init__(
        self,
        TE_ms=2.39,
        BW_per_pixel=1116.0,
        gradient_polarity="+",
        readout_axis=0,                          # 0=rows(y), 1=cols(x)

        # Dixon/anatomy voxel spacing in mm, numpy order (y,x,z)
        voxel_spacing_mm=(1.5625, 1.5625, 2.0),

        # B0 voxel spacing in mm, numpy order (y,x,z)
        b0_spacing_mm=(0.5, 0.5, 0.5),

        # dephasing controls all should be 1 to keep original sinc physics
        void_floor=0.0,
        dephasing_power=1.0,

        # optional pile-up
        jacobian_strength_inplane=0.0,

        # stabilizers
        max_pixel_shift=200.0,
    ):
        self.TE_s = float(TE_ms) / 1000.0
        self.BW_per_pixel = float(BW_per_pixel)
        self.gradient_polarity = gradient_polarity
        self.readout_axis = int(readout_axis)

        self.dy, self.dx, self.dz = [float(v) for v in voxel_spacing_mm]
        self.b0_dy, self.b0_dx, self.b0_dz = [float(v) for v in b0_spacing_mm]

        self.void_floor = float(void_floor)
        self.dephasing_power = float(dephasing_power)

        self.jacobian_strength_inplane = float(jacobian_strength_inplane)
        self.max_pixel_shift = float(max_pixel_shift)

        if self.gradient_polarity not in ("+", "-"):
            raise ValueError("gradient_polarity must be '+' or '-'")
        if self.readout_axis not in (0, 1):
            raise ValueError("readout_axis must be 0 or 1")
        if self.TE_s <= 0:
            raise ValueError("TE_ms must be > 0")
        if self.BW_per_pixel <= 0:
            raise ValueError("BW_per_pixel must be > 0")

        print("=" * 70)
        print("GRE / Dixon 3D Forward (B0 in Hz) — intravoxel sinc dephasing (hi-res gradients)")
        print("=" * 70)
        print(f"TE: {TE_ms:.3f} ms | BW/pixel: {self.BW_per_pixel:.1f} Hz/px | polarity: {gradient_polarity}")
        print(f"readout_axis: {self.readout_axis}  (0=rows/y, 1=cols/x)")
        print(f"Dixon voxel spacing (y,x,z) mm: {self.dy:.4f}, {self.dx:.4f}, {self.dz:.4f}")
        print(f"B0 voxel spacing    (y,x,z) mm: {self.b0_dy:.4f}, {self.b0_dx:.4f}, {self.b0_dz:.4f}")
        print(f"void_floor: {self.void_floor} | dephasing_power: {self.dephasing_power}")
        print(f"jacobian_strength_inplane: {self.jacobian_strength_inplane}")
        print("=" * 70 + "\n")

    @staticmethod
    def resize_to(vol, target_shape, order=1):
        if vol.shape == target_shape:
            return vol
        scale = [t / s for t, s in zip(target_shape, vol.shape)]
        return zoom(vol, scale, order=order, mode="nearest")

    def _signed_df(self, df_hz):
        return -df_hz if self.gradient_polarity == "-" else df_hz

    def geometric_shift_pixels(self, df_hz_signed):
        shift = df_hz_signed / (self.BW_per_pixel + 1e-20)
        return np.clip(shift, -self.max_pixel_shift, self.max_pixel_shift)

    def apply_geometric_warp_2d(self, img2d, pixel_shift):
        h, w = img2d.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        if self.readout_axis == 1:      # shift x/cols
            xx_shifted = np.clip(xx + pixel_shift, 0, w - 1)
            coords = [yy, xx_shifted]
        else:                           # shift y/rows
            yy_shifted = np.clip(yy + pixel_shift, 0, h - 1)
            coords = [yy_shifted, xx]

        return map_coordinates(img2d, coords, order=1, mode="nearest")

    def jacobian_inplane(self, pixel_shift):
        if self.jacobian_strength_inplane == 0.0:
            return None

        if self.readout_axis == 1:
            grad = np.gradient(pixel_shift, axis=1)
        else:
            grad = np.gradient(pixel_shift, axis=0)

        jac = np.clip(np.abs(1.0 + grad), 0.01, 10.0)
        jac_adj = 1.0 + self.jacobian_strength_inplane * (jac - 1.0)
        return np.clip(jac_adj, 0.01, 10.0)

    def void_mask_intravoxel_sinc_from_highres(self, df_signed_hi, target_shape):
        """
        Local-linear intravoxel dephasing:
          A = |sinc(gx*Dx*TE)| * |sinc(gy*Dy*TE)| * |sinc(gz*Dz*TE)|

        - gradients gx,gy,gz computed on HIGH-RES grid (Hz/mm)
        - then resampled to target_shape (Dixon grid)
        - Dx,Dy,Dz are the Dixon voxel sizes (mm)
        """
        # gradients on high-res grid (Hz/mm)
        dfd_y_hi, dfd_x_hi, dfd_z_hi = np.gradient(df_signed_hi, self.b0_dy, self.b0_dx, self.b0_dz)

        # resample gradients to Dixon grid
        dfd_y = self.resize_to(dfd_y_hi, target_shape, order=1)
        dfd_x = self.resize_to(dfd_x_hi, target_shape, order=1)
        dfd_z = self.resize_to(dfd_z_hi, target_shape, order=1)

        # NOTE: np.sinc(u) = sin(pi*u)/(pi*u)
        # Here u = (gradient [Hz/mm]) * (voxel_size [mm]) * TE [s]
        ay = np.abs(np.sinc(dfd_y * self.dy * self.TE_s))
        ax = np.abs(np.sinc(dfd_x * self.dx * self.TE_s))
        az = np.abs(np.sinc(dfd_z * self.dz * self.TE_s))

        mask = ax * ay * az

        if self.dephasing_power != 1.0:
            mask = np.clip(mask, 0.0, 1.0) ** self.dephasing_power

        if self.void_floor > 0:
            mask = self.void_floor + (1.0 - self.void_floor) * mask

        mask = np.clip(mask, 0.0, 1.0)

        gmag = np.sqrt(dfd_y**2 + dfd_x**2 + dfd_z**2)  # Hz/mm on Dixon grid (nice to visualize)
        return mask, gmag

    def forward(self, undist_volume, B0_volume_Hz, verbose=True):
        und = np.asarray(undist_volume, dtype=np.float64)
        b0  = np.asarray(B0_volume_Hz, dtype=np.float64)
        b0  = np.nan_to_num(b0, nan=0.0, posinf=0.0, neginf=0.0)

        if und.ndim != 3 or b0.ndim != 3:
            raise ValueError(f"Expected 3D volumes; got und={und.shape}, b0={b0.shape}")

        # signed B0 at native resolution (hi-res)
        df_signed_hi = self._signed_df(b0)

        # dephasing mask from hi-res gradients -> mapped to und grid
        void_mask_3d, gmag_3d = self.void_mask_intravoxel_sinc_from_highres(df_signed_hi, target_shape=und.shape)

        # geometry needs df on und grid
        df_signed_lo = df_signed_hi if df_signed_hi.shape == und.shape else self.resize_to(df_signed_hi, und.shape, order=1)

        H, W, Z = und.shape
        out = np.zeros_like(und, dtype=np.float64)

        inter = {
            "b0_signed_lo": df_signed_lo,
            "void_mask_3d": void_mask_3d,
            "grad_mag_hz_per_mm": gmag_3d,
            "pixel_shift": [],
        }

        if verbose:
            print("\n" + "=" * 70)
            print("GRE/Dixon forward — intravoxel sinc dephasing (hi-res gradients)")
            print("=" * 70)
            print(f"Undistorted: {und.shape}")
            print(f"B0 native:   {b0.shape} (Hz)")
            print(f"B0(lo) range: min={df_signed_lo.min():.1f}, max={df_signed_lo.max():.1f}, max|B0|={np.max(np.abs(df_signed_lo)):.1f}")
            print(f"Void mask: min={void_mask_3d.min():.3f}, max={void_mask_3d.max():.3f}, mean={void_mask_3d.mean():.3f}")
            print("=" * 70)

        for z in range(Z):
            if verbose and (z % 20 == 0 or z == Z - 1):
                print(f"  slice {z+1}/{Z}")

            I  = und[:, :, z]
            df = df_signed_lo[:, :, z]

            # geometry
            shift_px = self.geometric_shift_pixels(df)
            warped = self.apply_geometric_warp_2d(I, shift_px)

            # optional Jacobian
            jac = self.jacobian_inplane(shift_px)
            if jac is not None:
                warped = warped / (jac + 1e-20)

            out[:, :, z] = warped * void_mask_3d[:, :, z]
            inter["pixel_shift"].append(shift_px)

        if verbose:
            print("  ✓ Done.\n")

        return out, inter
