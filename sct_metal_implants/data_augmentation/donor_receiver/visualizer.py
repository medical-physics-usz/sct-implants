import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt
from skimage import measure
from sklearn.decomposition import PCA

pio.renderers.default = 'browser'

class Visualizer:
    """3D visualization of patient structures, implants, and voids."""

    def __init__(self):
        self.fig = None

    def plot_patient(self, patient, html_path=None):
        """Plot patient CT/MR structures, implants, and voids.
        Parameters:
            patient (PatientHandler) -- patient object
            html_path (str|None) -- output html path or None to display
        """
        patient_nr = patient.patient_nr
        body_mask = patient.ct_body_mask
        mask_dict = patient.structure_masks
        implant_mask_dict = patient.implant_masks
        implant_mask_dict = {f"{key}_implant": value for key, value in implant_mask_dict.items()}
        void_mask_dict = patient.mr_voids

        self.plot(patient_nr, body_mask, mask_dict, implant_mask_dict, void_mask_dict, html_path)

    def plot(self, patient_nr, body_mask, mask_dict, implant_mask_dict, void_mask_dict, html_path=None):
        """Generic plot function.
        Parameters:
            patient_nr (str) -- patient ID
            body_mask (ndarray) -- CT body mask
            mask_dict (dict) -- structure name -> mask
            implant_mask_dict (dict) -- implant name -> mask
            void_mask_dict (dict) -- void name -> mask
            html_path (str|None) -- output html path or None to display
        """
        self.fig = go.Figure()

        # Combine Mask & Dictionary
        combined_dict = {**mask_dict, **implant_mask_dict}

        # Add Body Outline
        self.add_outline_from_mask(body_mask, 'lightgray', 'Body Outline')

        # Add Masks
        cmap = plt.get_cmap('tab20')
        color_count = 0

        for i, (name, mask) in enumerate(combined_dict.items()):
            rgb = cmap(color_count % cmap.N)[:3]
            color_hex = f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})'
            if name == "FemurHead_R":
                color_hex = "rgb(31, 119, 180)"
            elif name == "FemurHead_L":
                color_hex = "rgb(174, 199, 232)"
            color_count += 1
            self.add_trace_from_mask(mask, color_hex, name)

        # Add void
        for name, mask in void_mask_dict.items():
            self.add_outline_from_mask(mask, 'lightblue', f'Void Region {name.split("_")[-1]}')

        # Update Layout
        self.fig.update_layout(
            scene=dict(aspectmode='data'),
            title=f'3D Plot of Patient {patient_nr}',
            margin=dict(l=0, r=0, b=0, t=30)
        )

        if not html_path:
            self.fig.show()
        else:
            self.fig.write_html(html_path)
            print(f"Saved Plot for Patient {patient_nr}")

    def add_outline_from_mask(self, mask, color, title, opacity=0.15):
        """Add outline surface from mask.
        Parameters:
            mask (ndarray) -- binary mask
            color (str) -- color string
            title (str) -- legend name
            opacity (float) -- surface opacity
        """
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        self.fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 2], j=faces[:, 1], k=faces[:, 0],
            opacity=opacity,
            color=color,
            name=title,
            showlegend=True
        ))

    def add_trace_from_mask(self, mask, color, title):
        """Add scatter trace from mask voxels.
        Parameters:
            mask (ndarray) -- binary mask
            color (str) -- color string
            title (str) -- legend name
        """
        z, y, x = np.nonzero(mask)
        self.fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=color),
            name=title
        ))

    def plot_implant(self, patient_nr, body_mask, implant_mask_dict, html_path=None, cut=None):
        """Plot body and implant masks.
        Parameters:
            patient_nr (str) -- patient ID
            body_mask (ndarray) -- CT body mask
            implant_mask_dict (dict) -- implant name -> mask
            html_path (str|None) -- output html path or None to display
            cut (int|None) -- optional slice index for cut visualization
        """
        self.fig = go.Figure()

        # Add Body Outline
        self.add_outline_from_mask(body_mask, 'lightgray', 'Body Outline')

        # Add void
        for name, mask in implant_mask_dict.items():
            if np.sum(mask) == 0:
                continue
            self.add_outline_from_mask(mask, 'red', name, opacity=0.5)

        # Add Cut
        if cut:
            cut_mask = np.zeros_like(body_mask)
            cut_mask[cut] = 1
            self.add_outline_from_mask(cut_mask, 'green', f"Cut at z={cut}")


        # Update Layout
        self.fig.update_layout(
            scene=dict(aspectmode='data'),
            title=f'3D Plot of Patient {patient_nr}',
            margin=dict(l=0, r=0, b=0, t=30)
        )

        if not html_path:
            self.fig.show()
        else:
            self.fig.write_html(html_path + f"Pat{patient_nr}_metal_area.html")
            print(f"Saved Plot for Patient {patient_nr}")