import argparse
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pydicom
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2
import numpy as np

class DICOMLineViewer:
    def __init__(self, root, images, titles, patient_name):
        """Interactive GUI for visualizing multimodal DICOM slices.

        Displays CT and MR images side by side with slice navigation,
        overlayed metal masks, and interactive line profile plotting
        of intensity values across modalities.
        """
        self.root = root
        self.original_images = images  # 3D images: [modality][slices, height, width]
        self.titles = titles
        self.patient_name = patient_name
        self.image_dimension = int(256 * 1.25)
        self.selected_mr_sequences = ["In-phase", "Opposed-phase", "Water-only"]

        self.mr_colors = {
            "In-phase": "orange",
            "Opposed-phase": "green",
            "Water-only": "blue",
            "Fat-only": "red"
        }

        self.current_slice = tk.IntVar(value=0)
        self.max_slices = images[0].shape[0]

        self.start = None
        self.end = None
        self.show_mask_3000 = tk.BooleanVar(value=False)
        self.show_mask_2000 = tk.BooleanVar(value=False)
        self.enable_artifact_detection = tk.BooleanVar(value=False)
        self.enable_artifact_contour = tk.BooleanVar(value=False)
        self.enable_artifact_candidate_area = tk.BooleanVar(value=False)

        self.low_signal_threshold = tk.DoubleVar(value=0.02)

        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=0, columnspan=5)

        self.image_canvases = []
        for i in range(len(images)):
            frame = tk.Frame(self.image_frame)
            frame.grid(row=0, column=i, padx=5, pady=5)

            canvas = tk.Canvas(frame, width=self.image_dimension, height=self.image_dimension)
            canvas.pack()
            canvas.bind("<Button-1>", self.on_click)
            canvas.bind("<ButtonRelease-1>", self.on_release)
            self.image_canvases.append(canvas)

            label = tk.Label(frame, text=titles[i])
            label.pack()

        self.display_all_images()

        self.plot_frame = tk.Frame(root)
        self.plot_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))

        # === Checkbox Controls ===
        self.checkbox_frame = tk.Frame(root)
        self.checkbox_frame.grid(row=2, column=0, columnspan=5, pady=(10, 10), sticky="w")

        # --- Row for MR/CT sequence checkboxes ---
        self.sequence_checkbox_frame = tk.Frame(self.checkbox_frame)
        self.sequence_checkbox_frame.pack(anchor="w", pady=(0, 5))

        self.profile_vars = {}
        for title in titles:
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(self.sequence_checkbox_frame, text=title, variable=var, command=self.plot_profiles)
            chk.pack(side="left", padx=5)
            self.profile_vars[title] = var

        # --- Row for CT mask checkboxes ---
        self.mask_checkbox_row = tk.Frame(self.checkbox_frame)
        self.mask_checkbox_row.pack(anchor="w", pady=(0, 5))

        self.mask_checkbox_2000 = tk.Checkbutton(
            self.mask_checkbox_row, text="Show CT mask > 2000",
            variable=self.show_mask_2000, command=self.display_all_images
        )
        self.mask_checkbox_2000.pack(side="left", padx=(0, 10))

        self.mask_checkbox_3000 = tk.Checkbutton(
            self.mask_checkbox_row, text="Show CT mask > 3000",
            variable=self.show_mask_3000, command=self.display_all_images
        )
        self.mask_checkbox_3000.pack(side="left")

        self.slice_slider = tk.Scale(root, from_=0, to=self.max_slices - 1, orient=tk.HORIZONTAL,
                                     variable=self.current_slice, label="Slice", command=self.on_slice_change)
        self.slice_slider.grid(row=3, column=0, columnspan=5, sticky="we")

        self.load_button = tk.Button(root, text="Load Patient", command=self.load_new_patient)
        self.load_button.grid(row=4, column=0, columnspan=5, pady=5)


        self.clear_plot_button = tk.Button(self.sequence_checkbox_frame, text="Clear Plot", command=self.clear_plot)
        self.clear_plot_button.pack(side="left", padx=5)

        self.patient_label = tk.Label(root, text=f"Patient: {self.patient_name}", font=("Helvetica", 14, "bold"))
        self.patient_label.grid(row=5, column=0, columnspan=5, pady=(5, 10))



    def load_new_patient(self):
        """Load new patient to display in GUI."""

        try:
            paths_dict, titles, patient_name = select_dicom_directories(None)
            images = [load_dicom_series(paths_dict[title]) for title in titles]
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not load patient:\n{e}")
            return

        self.original_images = images
        self.titles = titles
        self.patient_name = patient_name
        self.patient_label.config(text=f"Patient: {self.patient_name}")
        self.max_slices = images[0].shape[0]
        self.current_slice.set(0)
        self.slice_slider.config(to=self.max_slices - 1)
        self.display_all_images()

    def clear_plot(self):
        """Clear CT/MR intensity plot."""
        self.plot_frame.destroy()
        self.plot_frame = tk.Frame(root)
        self.plot_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))

        self.start = None
        self.end = None
        self.display_all_images()

    def resize_with_padding(self, img, target_size):
        """Resize image to target size for GUI."""
        h, w = img.shape
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = np.zeros(target_size, dtype=img.dtype)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return result

    def display_all_images(self):
        """Display all image modalities on GUI."""

        slice_idx = self.current_slice.get()
        ct_index = self.titles.index("CT")
        ct_resized = self.resize_with_padding(self.original_images[ct_index][slice_idx], (self.image_dimension, self.image_dimension))
        mask_3000 = ct_resized > 3000 if self.show_mask_3000.get() else None
        mask_2000 = ct_resized > 2000 if self.show_mask_2000.get() else None


        # Loop through all canvases (displayed modalities)
        for i, canvas in enumerate(self.image_canvases):
            img = self.original_images[i][slice_idx]
            resized = self.resize_with_padding(img, (self.image_dimension, self.image_dimension))
            disp_img = ((resized - resized.min()) / (resized.ptp() + 1e-6) * 255).astype(np.uint8)
            rgb_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)


            # Overlay metal mask (HU > 2000) with yellow color on all images
            if mask_2000 is not None:
                rgb_img[mask_2000] = [255, 255, 0]

            # Overlay metal mask (HU > 3000) with green color on all images
            if mask_3000 is not None:
                rgb_img[mask_3000] = [0, 255, 0]


            # Show image on GUI
            pil_img = Image.fromarray(rgb_img)
            tk_img = ImageTk.PhotoImage(pil_img)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            canvas.image = tk_img

            # Draw corss-section in red on all images
            if self.start and self.end:
                canvas.create_line(
                    self.start[0], self.start[1],
                    self.end[0], self.end[1],
                    fill='red', width=2, tag='line'
                )

    def on_click(self, event):
        """Start drawing cross-section."""
        self.start = (event.x, event.y)

    def on_release(self, event):
        """Stop drawing cross-section, update GUI."""
        self.end = (event.x, event.y)
        self.display_all_images()
        self.plot_profiles()

    def on_slice_change(self, val):
        """Render GUI, when new slice selcted."""
        self.display_all_images()
        self.plot_profiles()

    def extract_line_profile(self, img):
        """Extract intensities of cross-section."""
        x0, y0 = self.start
        x1, y1 = self.end
        length = int(np.hypot(x1 - x0, y1 - y0))
        x_vals = np.linspace(x0, x1, length).astype(int)
        y_vals = np.linspace(y0, y1, length).astype(int)
        x_vals = np.clip(x_vals, 0, 511)
        y_vals = np.clip(y_vals, 0, 511)
        return img[y_vals, x_vals]

    def plot_profiles(self):
        """Plot profiles on line plot."""

        if not (self.start and self.end):
            return

        slice_idx = self.current_slice.get()
        mri_profiles = []
        ct_profile = None
        ct_profile_raw = None

        for i, volume in enumerate(self.original_images):
            img = self.resize_with_padding(volume[slice_idx], (self.image_dimension, self.image_dimension))
            profile = self.extract_line_profile(img)
            title = self.titles[i]

            if title.lower() == "ct":
                ct_profile_raw = (title, profile)
                if self.profile_vars[title].get():
                    ct_profile = (title, profile)
            elif title in self.mr_colors:
                mri_profiles.append((title, profile))

        fig, ax1 = plt.subplots(figsize=(12, 5), dpi=100)
        plt.subplots_adjust(right=0.7)
        ax2 = ax1.twinx()

        # Plot MR intensity profiles
        for title, prof in mri_profiles:
            if self.profile_vars[title].get():
                ax1.plot(prof, label=title, color=self.mr_colors.get(title))

        # Plot CT intensity profile
        if ct_profile:
            ct_label, ct_values = ct_profile
            ax2.plot(ct_values, label=ct_label, color='black', linestyle='--')

        # Plot CT mask  (HU < 3000)
        if ct_profile_raw and self.show_mask_3000.get():
            _, ct_vals = ct_profile_raw
            high_indices = np.where(ct_vals > 3000)[0]
            ranges = np.split(high_indices, np.where(np.diff(high_indices) != 1)[0] + 1)
            for r in ranges:
                if len(r) > 0:
                    ax1.axvspan(r[0], r[-1], color='lightgreen', alpha=0.3)
        # Plot CT mask  (HU < 2000)
        if ct_profile_raw and self.show_mask_2000.get():
            _, ct_vals = ct_profile_raw
            high_indices = np.where(ct_vals > 2000)[0]
            ranges = np.split(high_indices, np.where(np.diff(high_indices) != 1)[0] + 1)
            for r in ranges:
                if len(r) > 0:
                    ax1.axvspan(r[0], r[-1], color='yellow', alpha=0.3)

        ax1.set_xlabel("Distance")
        ax1.set_ylabel("MRI Intensity")
        ax2.set_ylabel("CT Intensity")
        ax1.set_title("Intensity Profiles")
        ax1.grid(True)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="center left", bbox_to_anchor=(1.15, 0.5), borderaxespad=0.)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack()

        plt.close(fig)


def load_dicom_series(dir_path, skip_first=10, skip_last=10):
    """Load a DICOM series from a directory of .dcm files."""

    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in files]

    # Sort by InstanceNumber or Z-position
    try:
        slices.sort(key=lambda ds: int(ds.InstanceNumber))
    except AttributeError:
        slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    # Detect Z-axis direction and flip if needed (for CT)
    try:
        z_positions = [float(ds.ImagePositionPatient[2]) for ds in slices]
        if z_positions[0] < z_positions[-1]:  # Bottom to top → needs reversing
            slices.reverse()
    except Exception:
        pass  # If no ImagePositionPatient, just trust InstanceNumber

    # Apply skip at load time
    if skip_last > 0:
        slices = slices[skip_first:-skip_last]
    else:
        slices = slices[skip_first:]

    # Apply slope & intercept to get HU units
    pixel_arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * slope + intercept
        pixel_arrays.append(arr)

    return np.stack(pixel_arrays)


def select_dicom_directories(path):
    """Select directory containing DICOMs of different modalities."""

    parent_dir = filedialog.askdirectory(
        initialdir= path,
        title="Select Patient Folder (contains CT, MR_in, MR_opp, MR_W, MR_F)")
    if not parent_dir:
        raise ValueError("No parent directory selected.")

    patient_name = os.path.basename(parent_dir)  # <- extract folder name

    subdirs = {
        "In-phase": "MR_in",
        "Opposed-phase": "MR_opp",
        "Water-only": "MR_W",
        "Fat-only": "MR_F",
        "CT": "CT"
    }

    paths = {}
    for title, folder in subdirs.items():
        full_path = os.path.join(parent_dir, folder)
        if not os.path.isdir(full_path):
            raise ValueError(f"Missing subdirectory: {folder}")
        paths[title] = full_path

    return paths, list(subdirs.keys()), patient_name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DICOM Intensity Viewer")
    parser.add_argument("--data_path", type=str, required=True,
        help="Path to directory containing patient folders (with CT, MR_in, MR_opp, MR_W, MR_F subdirs).")
    args = parser.parse_args()

    root = tk.Tk()
    root.withdraw()

    paths_dict, titles, patient_name = select_dicom_directories(args.data_path)
    images = [load_dicom_series(paths_dict[title]) for title in titles]

    root.deiconify()
    root.title("DICOM Intensity Viewer with Slice Navigation")
    viewer = DICOMLineViewer(root, images, titles, patient_name)
    root.mainloop()