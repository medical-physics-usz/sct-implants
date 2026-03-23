**Literature**

Intravoxel dephasing : https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28182
 https://pubmed.ncbi.nlm.nih.gov/18575911/

 Off frequency : https://onlinelibrary.wiley.com/doi/10.1002/mrm.24343 
 matlab script for field shift https://ch.mathworks.com/matlabcentral/fileexchange/37278-forward-field-shift-calculation-for-mri , the rest of the code for the calculation of the off frequency was dev. by Jonas Wahlen 

Magnetic Susceptibility values : https://pmc.ncbi.nlm.nih.gov/articles/PMC5529184/ 

## Physical Basis

### Intravoxel Dephasing Theory

In gradient-echo (GRE) imaging, static magnetic field (B0) inhomogeneities cause intravoxel dephasing: different locations within a single voxel experience different off-resonance frequencies, leading to destructive interference and signal loss.

For a **piecewise linear B0 field** (constant gradients within each voxel), the signal attenuation due to intravoxel dephasing follows the **sinc-product model**:

$$
A(\mathbf{r}) = \left|\text{sinc}\left(g_x \cdot \Delta x \cdot \text{TE}\right)\right| \times \left|\text{sinc}\left(g_y \cdot \Delta y \cdot \text{TE}\right)\right| \times \left|\text{sinc}\left(g_z \cdot \Delta z \cdot \text{TE}\right)\right|
$$

Where:
- $g_x, g_y, g_z$ = B0 field gradients in Hz/mm (spatial derivatives of the field)
- $\Delta x, \Delta y, \Delta z$ = **Dixon/anatomy voxel dimensions** in mm
- $\text{TE}$ = echo time in seconds
- $\text{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$ (NumPy's convention)



The sinc argument $g \cdot \Delta \cdot \text{TE}$ represents the **phase dispersion across a voxel**:
- The gradient $g$ tells us how fast the field changes spatially
- The voxel size $\Delta$ tells us **over what distance** we integrate that phase
- Together, they determine how much dephasing occurs within the acquired voxel

**Important**: The dephasing happens in the **acquired image voxel** (Dixon grid), not in the B0 map voxel. We care about phase variation across the Dixon voxel dimensions.



## Multi-Resolution Grid Handling

We have **two different spatial grids**:

1. **B0 field map**: High-resolution (e.g., 0.5 × 0.5 × 0.5 mm³)
   - Provides detailed B0(x,y,z) in Hz
   
2. **Dixon/anatomy image**: Coarser resolution (e.g., 1.5625 × 1.5625 × 2.0 mm³)
   - The actual acquired image we want to simulate

### The Solution: Gradient Computation Strategy

To accurately model intravoxel dephasing, we follow this pipeline:

```
Step 1: Compute gradients on HIGH-RES B0 grid
   ↓
Step 2: Resample gradients to DIXON grid  
   ↓
Step 3: Apply sinc formula using DIXON voxel sizes
```





### Step 1: High-Resolution Gradient Computation

Given the B0 field map $\Delta f(\mathbf{r})$ on its native grid with spacing $(dy_{\text{B0}}, dx_{\text{B0}}, dz_{\text{B0}})$:

$$
g_y^{\text{hi}} = \frac{\partial \Delta f}{\partial y}\bigg|_{\text{B0 grid}} \quad [\text{Hz/mm}]
$$

$$
g_x^{\text{hi}} = \frac{\partial \Delta f}{\partial x}\bigg|_{\text{B0 grid}} \quad [\text{Hz/mm}]
$$

$$
g_z^{\text{hi}} = \frac{\partial \Delta f}{\partial z}\bigg|_{\text{B0 grid}} \quad [\text{Hz/mm}]
$$

**Implementation**:
```python
# NumPy gradient with explicit spacing
dfd_y_hi, dfd_x_hi, dfd_z_hi = np.gradient(
    df_signed_hi,           # B0 field in Hz
    self.b0_dy,             # B0 spacing y (mm)
    self.b0_dx,             # B0 spacing x (mm)
    self.b0_dz              # B0 spacing z (mm)
)
```

**Why high-resolution?** Computing gradients on the finer grid captures sharp B0 transitions more accurately than computing on the coarser Dixon grid.



### Step 2: Gradient Resampling

Resample gradients from B0 grid → Dixon grid using trilinear interpolation:

$$
g_x(\mathbf{r}_{\text{Dixon}}) = \text{Resample}\left[g_x^{\text{hi}}(\mathbf{r}_{\text{B0}})\right]
$$

$$
g_y(\mathbf{r}_{\text{Dixon}}) = \text{Resample}\left[g_y^{\text{hi}}(\mathbf{r}_{\text{B0}})\right]
$$

$$
g_z(\mathbf{r}_{\text{Dixon}}) = \text{Resample}\left[g_z^{\text{hi}}(\mathbf{r}_{\text{B0}})\right]
$$

**Implementation**:
```python
dfd_y = self.resize_to(dfd_y_hi, target_shape=dixon_shape, order=1)
dfd_x = self.resize_to(dfd_x_hi, target_shape=dixon_shape, order=1)
dfd_z = self.resize_to(dfd_z_hi, target_shape=dixon_shape, order=1)
```

Now we have gradients in Hz/mm on the Dixon grid.


### Step 3: Sinc-Product Dephasing

Apply the sinc formula using **Dixon voxel dimensions**:

$$
A_y = \left|\text{sinc}\left(g_y \cdot \Delta y_{\text{Dixon}} \cdot \text{TE}\right)\right|
$$

$$
A_x = \left|\text{sinc}\left(g_x \cdot \Delta x_{\text{Dixon}} \cdot \text{TE}\right)\right|
$$

$$
A_z = \left|\text{sinc}\left(g_z \cdot \Delta z_{\text{Dixon}} \cdot \text{TE}\right)\right|
$$

$$
A_{\text{total}} = A_x \cdot A_y \cdot A_z
$$

**Implementation**:
```python
ay = np.abs(np.sinc(dfd_y * self.dy * self.TE_s))  # self.dy = Dixon voxel size y
ax = np.abs(np.sinc(dfd_x * self.dx * self.TE_s))  # self.dx = Dixon voxel size x
az = np.abs(np.sinc(dfd_z * self.dz * self.TE_s))  # self.dz = Dixon voxel size z

mask = ax * ay * az  # Combined attenuation
```



### Physical Interpretation

$$
\phi_{\text{dispersion}} = 2\pi \cdot \underbrace{g \cdot \Delta}_{\text{frequency variation across voxel (Hz)}} \cdot \underbrace{\text{TE}}_{\text{accumulation time (s)}}
$$

This represents the **range of phase accumulation** across the Dixon voxel during the echo time. The sinc function emerges from integrating $e^{i\phi}$ over this range.



## Key Takeaways

 **Gradients are computed on the B0 grid** for accuracy in capturing field variations
 **Gradients are resampled to the Dixon grid** to align with the anatomy
 **Dixon voxel sizes are used in the sinc formula** because dephasing occurs over the acquired voxel dimensions
 Larger voxels → more dephasing → stronger signal attenuation
