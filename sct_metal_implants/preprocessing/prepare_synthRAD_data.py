#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import uuid
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Sequence, Dict

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from scipy.ndimage import binary_erosion
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pydicom

TARGET_SPACING = (1.5625, 1.5625, 2.0)  # (sx, sy, sz) mm
TARGET_SIZE_XY = (320, 240)  # (size_x, size_y) voxels

def _uid() -> str:
    return f"2.25.{uuid.uuid4().int}"


def _dicom_date_time() -> tuple[str, str]:
    now = dt.datetime.now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S")


def _write_dicom_series(
    image: sitk.Image,
    out_dir: Path,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    frame_of_ref_uid: str,
    modality: str,
    series_description: str,
    series_number: int,
) -> None:
    if modality not in {"CT", "MR"}:
        raise ValueError(f"Unsupported modality: {modality}")

    sop_class_uid = {
        "CT": "1.2.840.10008.5.1.4.1.1.2",
        "MR": "1.2.840.10008.5.1.4.1.1.4",
    }[modality]

    out_dir.mkdir(parents=True, exist_ok=True)
    date_str, time_str = _dicom_date_time()
    direction = image.GetDirection()

    image_i16 = sitk.Cast(sitk.Round(image), sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for z in range(image_i16.GetDepth()):
        slice_img = image_i16[:, :, z]
        sop_instance_uid = _uid()

        slice_img.SetMetaData("0008|0016", sop_class_uid)
        slice_img.SetMetaData("0008|0018", sop_instance_uid)
        slice_img.SetMetaData("0008|0060", modality)
        slice_img.SetMetaData("0008|0020", date_str)
        slice_img.SetMetaData("0008|0030", time_str)
        slice_img.SetMetaData("0008|1030", "NIfTI2DICOM")
        slice_img.SetMetaData("0008|103e", series_description)
        slice_img.SetMetaData("0010|0010", patient_id)
        slice_img.SetMetaData("0010|0020", patient_id)
        slice_img.SetMetaData("0020|000d", study_uid)
        slice_img.SetMetaData("0020|000e", series_uid)
        slice_img.SetMetaData("0020|0052", frame_of_ref_uid)
        slice_img.SetMetaData("0020|0011", str(series_number))
        slice_img.SetMetaData("0020|0013", str(z + 1))
        slice_img.SetMetaData(
            "0020|0032",
            "\\".join(map(str, image_i16.TransformIndexToPhysicalPoint((0, 0, z)))),
        )
        slice_img.SetMetaData(
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        )
        slice_img.SetMetaData("0018|0050", str(image_i16.GetSpacing()[2]))
        slice_img.SetMetaData("0018|0088", str(image_i16.GetSpacing()[2]))
        slice_img.SetMetaData("0028|1052", "0")
        slice_img.SetMetaData("0028|1053", "1")

        writer.SetFileName(str(out_dir / f"{z + 1:04d}.dcm"))
        writer.Execute(slice_img)


def _discover_mask_paths(
    patient_dir: Path,
    masks_subdir: str,
    ct_filename: str,
    mr_filename: str,
) -> list[Path]:
    ts_dir = patient_dir / masks_subdir
    if ts_dir.is_dir():
        masks = sorted(ts_dir.glob("*.nii.gz"))
        if masks:
            return masks

    excluded = {ct_filename.lower(), mr_filename.lower()}
    return sorted(
        p for p in patient_dir.glob("*.nii.gz") if p.name.lower() not in excluded
    )


def _build_rtstruct_from_masks(
    patient_dir: Path,
    ct_dicom_dir: Path,
    out_file: Path,
    masks_subdir: str,
    ct_filename: str,
    mr_filename: str,
) -> None:
    roi_name_map = {
        "femur_left": "FemurHead_L",
        "femur_right": "FemurHead_R",
    }

    mask_paths = _discover_mask_paths(
        patient_dir=patient_dir,
        masks_subdir=masks_subdir,
        ct_filename=ct_filename,
        mr_filename=mr_filename,
    )
    if not mask_paths:
        raise FileNotFoundError(
            f"No mask NIfTI files found in {patient_dir} or {patient_dir / masks_subdir}"
        )

    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dicom_dir))

    for mask_path in mask_paths:
        name = roi_name_map.get(mask_path.name.replace(".nii.gz", ""), mask_path.name.replace(".nii.gz", ""))
        arr = nib.load(str(mask_path)).get_fdata()
        binary = arr > 0.5

        if binary.ndim != 3:
            raise ValueError(f"Mask is not 3D: {mask_path}")

        mask_rcs = np.transpose(binary, (1, 0, 2)).astype(bool)
        if not mask_rcs.any():
            continue

        rtstruct.add_roi(
            mask=mask_rcs,
            name=name,
            description=f"Auto-converted from {mask_path.name}",
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(out_file))


def _resolve_named_mask_path(
    patient_dir: Path,
    masks_subdir: str,
    ct_filename: str,
    mr_filename: str,
    mask_name: str,
) -> Optional[Path]:
    mask_name_lower = mask_name.lower()
    for mask_path in _discover_mask_paths(
        patient_dir=patient_dir,
        masks_subdir=masks_subdir,
        ct_filename=ct_filename,
        mr_filename=mr_filename,
    ):
        if mask_path.name.replace(".nii.gz", "").lower() == mask_name_lower:
            return mask_path
    return None


def _load_ct_as_sitk(ct_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in CT folder: {ct_dir}")
    files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def _resample_label_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def _binary_expand_mm(mask_img: sitk.Image, margin_mm: float) -> sitk.Image:
    mask = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
    dist = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.Cast(dist <= margin_mm, sitk.sitkUInt8)


def _binary_subtract(a: sitk.Image, b: sitk.Image) -> sitk.Image:
    return sitk.Cast((a > 0) & ~(b > 0), sitk.sitkUInt8)


def _keep_biggest_body_contours_yxz(mask_yxz: "np.ndarray") -> "np.ndarray":
    """Contour cleanup for masks stored as (Y, X, Z)."""
    out_mask = mask_yxz.copy().astype(np.uint8)

    for i in range(out_mask.shape[2]):
        slice_mask = out_mask[:, :, i].astype(np.uint8)
        if not np.any(slice_mask):
            continue

        _, bin_img = cv2.threshold(slice_mask, 0.5, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            bin_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        area1 = cv2.contourArea(contours_sorted[0])
        keep = [contours_sorted[0]]

        if len(contours_sorted) > 1:
            area2 = cv2.contourArea(contours_sorted[1])
            if area1 > 0 and area2 >= 0.5 * area1:
                keep.append(contours_sorted[1])

        contour_mask = np.zeros_like(bin_img, dtype=np.uint8)
        cv2.drawContours(contour_mask, keep, contourIdx=-1, color=1, thickness=-1)
        out_mask[:, :, i] = contour_mask

    return out_mask


def _build_body_mask_array_yxz(image: sitk.Image, threshold: float) -> "np.ndarray":
    """Build BODY mask in plain array space with shape (Y, X, Z)."""
    image_zyx = sitk.GetArrayFromImage(image)
    image_yxz = np.transpose(image_zyx, (1, 2, 0))

    mask_yxz = np.zeros(image_yxz.shape, dtype=np.uint8)
    mask_yxz[image_yxz > threshold] = 1
    mask_yxz = binary_erosion(mask_yxz, iterations=2).astype(np.uint8)
    mask_yxz = _keep_biggest_body_contours_yxz(mask_yxz).astype(np.uint8)
    return mask_yxz


def _mask_yxz_to_sitk(mask_yxz: "np.ndarray", reference_img: sitk.Image) -> sitk.Image:
    mask_zyx = np.transpose(mask_yxz.astype(np.uint8), (2, 0, 1))
    mask_img = sitk.GetImageFromArray(mask_zyx)
    mask_img.CopyInformation(reference_img)
    return sitk.Cast(mask_img, sitk.sitkUInt8)


def _rtutils_mask_to_sitk(mask_xyz: "np.ndarray", ct_img: sitk.Image) -> sitk.Image:
    ct_size = ct_img.GetSize()  # (X, Y, Z)
    mask_shape = tuple(mask_xyz.shape)

    if mask_shape == ct_size:
        # Legacy/ambiguous case: (X, Y, Z)
        arr_zyx = np.transpose(mask_xyz.astype(np.uint8), (2, 1, 0))
    elif mask_shape == (ct_size[1], ct_size[0], ct_size[2]):
        # rt-utils convention used elsewhere in this repo: (Y, X, Z)
        arr_zyx = np.transpose(mask_xyz.astype(np.uint8), (2, 0, 1))
    else:
        raise ValueError(
            f"Unexpected RTSTRUCT mask shape {mask_shape} for CT size {ct_size}"
        )

    img = sitk.GetImageFromArray(arr_zyx)
    img.CopyInformation(ct_img)
    return img


def _sitk_to_rtutils_mask_like(
    img: sitk.Image,
    reference_shape: tuple[int, int, int],
    ct_img: sitk.Image,
) -> "np.ndarray":
    arr_zyx = sitk.GetArrayFromImage(img) > 0
    ct_size = ct_img.GetSize()  # (X, Y, Z)

    if reference_shape == ct_size:
        # Match source mask layout: (X, Y, Z)
        return np.transpose(arr_zyx, (2, 1, 0))
    if reference_shape == (ct_size[1], ct_size[0], ct_size[2]):
        # Match source mask layout: (Y, X, Z)
        return np.transpose(arr_zyx, (1, 2, 0))

    raise ValueError(
        f"Unexpected reference mask shape {reference_shape} for CT size {ct_size}"
    )


def _load_nifti_mask_on_ct_grid(mask_path: Path, ct_nifti_path: Path) -> sitk.Image:
    mask_arr = nib.load(str(mask_path)).get_fdata()
    mask_binary = (mask_arr > 0.5).astype(np.uint8)
    if mask_binary.ndim != 3:
        raise ValueError(f"Mask is not 3D: {mask_path}")

    ct_nifti_img = sitk.ReadImage(str(ct_nifti_path))
    mask_yxz = np.transpose(mask_binary, (1, 0, 2))
    return _mask_yxz_to_sitk(mask_yxz, ct_nifti_img)


def _augment_rtstruct_with_derived_contours(
    ct_dicom_series_dir: Path,
    mr_dicom_series_dir: Path,
    rtstruct_in_path: Path,
    out_rtstruct_path: Path,
    prostate_roi_name: str,
    ptv_margin_mm: float,
    outer_mm: float,
    body_nifti_path: Optional[Path],
    body_threshold_ct: Optional[float],
    body_threshold_mr: Optional[float],
    body_roi_name: str,
    prostate_nifti_path: Optional[Path] = None,
    ct_nifti_path: Optional[Path] = None,
) -> None:
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_dicom_series_dir),
        rt_struct_path=str(rtstruct_in_path),
    )
    ct_img = _load_ct_as_sitk(ct_dicom_series_dir)
    mr_img = _load_ct_as_sitk(mr_dicom_series_dir)

    if prostate_nifti_path is not None and ct_nifti_path is not None:
        prostate_mask_shape = (ct_img.GetSize()[1], ct_img.GetSize()[0], ct_img.GetSize()[2])
        prostate_src_img = _load_nifti_mask_on_ct_grid(prostate_nifti_path, ct_nifti_path)
        prostate_img = _resample_label_to_reference(prostate_src_img, ct_img)
    else:
        prostate_mask = rtstruct.get_roi_mask_by_name(prostate_roi_name)
        prostate_mask_shape = tuple(prostate_mask.shape)
        prostate_img = _rtutils_mask_to_sitk(prostate_mask, ct_img)

    ptv5_img = _binary_expand_mm(prostate_img, ptv_margin_mm)
    ptv20_img = _binary_expand_mm(ptv5_img, outer_mm)
    ring_img = _binary_subtract(ptv20_img, ptv5_img)

    existing = set(rtstruct.get_roi_names())
    new_rois = {
        "PTV": _sitk_to_rtutils_mask_like(ptv5_img, prostate_mask_shape, ct_img),
        "PTV_plus20mm": _sitk_to_rtutils_mask_like(ptv20_img, prostate_mask_shape, ct_img),
        "Ring_PTV": _sitk_to_rtutils_mask_like(ring_img, prostate_mask_shape, ct_img),
    }

    if body_nifti_path is not None:
        body_arr = nib.load(str(body_nifti_path)).get_fdata()
        body_binary = (body_arr > 0).astype(np.uint8)
        body_img = sitk.GetImageFromArray(np.transpose(body_binary, (2, 1, 0)))
        body_on_ct = _resample_label_to_reference(body_img, ct_img)
        body_minus = _binary_subtract(body_on_ct, ptv20_img)
        new_rois[body_roi_name] = _sitk_to_rtutils_mask_like(
            body_minus,
            prostate_mask_shape,
            ct_img,
        )
    elif body_threshold_ct is not None and body_threshold_mr is not None:
        body_mask_ct_yxz = _build_body_mask_array_yxz(ct_img, threshold=body_threshold_ct)
        body_mask_mr_yxz = _build_body_mask_array_yxz(mr_img, threshold=body_threshold_mr)
        body_mask_ct_img = _mask_yxz_to_sitk(body_mask_ct_yxz, ct_img)
        body_mask_mr_img = _mask_yxz_to_sitk(body_mask_mr_yxz, mr_img)
        body_mask_mr_on_ct = _resample_label_to_reference(body_mask_mr_img, ct_img)
        body_on_ct = sitk.Cast(
            (body_mask_ct_img > 0) & (body_mask_mr_on_ct > 0),
            sitk.sitkUInt8,
        )
        body_minus = _binary_subtract(body_on_ct, ptv20_img)
        new_rois[body_roi_name] = _sitk_to_rtutils_mask_like(
            body_minus,
            prostate_mask_shape,
            ct_img,
        )

    duplicate_names = sorted(set(new_rois) & existing)
    if duplicate_names:
        raise RuntimeError(f"ROI name(s) already exist in RTSTRUCT: {duplicate_names}")

    for name, mask in new_rois.items():
        if name == body_roi_name:
            try:
                rtstruct.add_roi(
                    mask=mask,
                    name=name,
                    use_pin_hole=True,
                    approximate_contours=False,
                )
                continue
            except TypeError:
                pass
            try:
                rtstruct.add_roi(mask=mask, name=name, use_pin_hole=True)
                continue
            except TypeError:
                pass
        rtstruct.add_roi(mask=mask, name=name)

    out_rtstruct_path.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(out_rtstruct_path))


def _resolve_body_mask_path(body_mask_template: Optional[str], patient_id: str) -> Optional[Path]:
    if not body_mask_template:
        return None
    path = Path(body_mask_template.format(patient_id=patient_id))
    if not path.exists():
        raise FileNotFoundError(f"Missing BODY mask for {patient_id}: {path}")
    return path

def load_dicom_series(series_dir: Path) -> Tuple[sitk.Image, List[str]]:
    """Load a DICOM series folder as a 3D SimpleITK image and slice file list."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {series_dir}")

    file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute(), list(file_names)


def make_reference_image(
    src_img: sitk.Image,
    target_spacing: Tuple[float, float, float] = TARGET_SPACING,
    target_size_xy: Tuple[int, int] = TARGET_SIZE_XY,
    target_size_z: Optional[int] = None,
) -> sitk.Image:
    """
    Build the target image geometry while preserving the source image center.

    X/Y are forced to the requested size; Z is derived from physical extent unless
    explicitly provided.
    """
    src_size = np.array(src_img.GetSize(), dtype=np.float64)
    src_spacing = np.array(src_img.GetSpacing(), dtype=np.float64)
    src_origin = np.array(src_img.GetOrigin(), dtype=np.float64)
    direction = np.array(src_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    tgt_spacing = np.array(target_spacing, dtype=np.float64)

    if target_size_z is None:
        physical_z = src_size[2] * src_spacing[2]
        target_size_z = max(int(np.round(physical_z / tgt_spacing[2])), 1)

    tgt_size = np.array(
        [target_size_xy[0], target_size_xy[1], target_size_z],
        dtype=np.float64,
    )

    src_center = src_origin + direction @ ((src_size - 1) * src_spacing / 2.0)
    tgt_origin = src_center - direction @ ((tgt_size - 1) * tgt_spacing / 2.0)

    ref = sitk.Image([int(x) for x in tgt_size], src_img.GetPixelID())
    ref.SetSpacing(tuple(tgt_spacing))
    ref.SetDirection(src_img.GetDirection())
    ref.SetOrigin(tuple(tgt_origin))
    return ref


def resample_to_reference(
    src_img: sitk.Image,
    ref_img: sitk.Image,
    is_label: bool = False,
    default_value: float = -1024.0,
) -> sitk.Image:
    """Resample `src_img` onto `ref_img` geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    return resampler.Execute(src_img)


def write_resampled_dicom_series(
    resampled_img: sitk.Image,
    template_first_slice_path: str,
    out_dir: Path,
    series_description: str,
    keep_study_uid: bool = True,
) -> Dict[str, str]:
    """
    Write a resampled image as a coherent DICOM series.

    SimpleITK writes the pixel data; pydicom then patches identifiers and
    geometry so viewers group the slices as one series.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.dcm"):
        old_file.unlink()

    tmpl = pydicom.dcmread(template_first_slice_path, stop_before_pixels=True)

    study_uid = str(getattr(tmpl, "StudyInstanceUID", "")) if keep_study_uid else ""
    if not study_uid:
        study_uid = generate_uid()

    series_uid = generate_uid()
    frame_of_ref_uid = str(getattr(tmpl, "FrameOfReferenceUID", "")) or generate_uid()
    sop_class_uid = str(getattr(tmpl, "SOPClassUID", "")) or "1.2.840.10008.5.1.4.1.1.7"

    patient_name = str(getattr(tmpl, "PatientName", ""))
    patient_id = str(getattr(tmpl, "PatientID", ""))
    birth_date = str(getattr(tmpl, "PatientBirthDate", ""))
    sex = str(getattr(tmpl, "PatientSex", ""))

    modality = str(getattr(tmpl, "Modality", "OT"))
    study_desc = str(getattr(tmpl, "StudyDescription", ""))
    study_date = str(getattr(tmpl, "StudyDate", ""))
    study_time = str(getattr(tmpl, "StudyTime", ""))
    series_date = str(getattr(tmpl, "SeriesDate", "")) or study_date
    series_time = str(getattr(tmpl, "SeriesTime", "")) or study_time
    series_number = int(getattr(tmpl, "SeriesNumber", 900))

    spacing = resampled_img.GetSpacing()
    direction = np.array(resampled_img.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.array(resampled_img.GetOrigin(), dtype=float)
    size = resampled_img.GetSize()

    row = direction[:, 0]
    col = direction[:, 1]
    slice_dir = direction[:, 2]

    image_orientation_patient = [float(v) for v in np.r_[row, col]]
    pixel_spacing = [float(spacing[1]), float(spacing[0])]

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOff()
    writer.SetImageIO("GDCMImageIO")

    written_paths: List[Path] = []
    for k in range(size[2]):
        out_path = out_dir / f"IM_{k + 1:04d}.dcm"
        writer.SetFileName(str(out_path))
        writer.Execute(resampled_img[:, :, k])
        written_paths.append(out_path)

    for k, path in enumerate(written_paths):
        ds = pydicom.dcmread(str(path))

        sop_instance_uid = generate_uid()

        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = sop_instance_uid
        ds.file_meta.MediaStorageSOPClassUID = sop_class_uid
        ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid

        if patient_name:
            ds.PatientName = patient_name
        if patient_id:
            ds.PatientID = patient_id
        if birth_date:
            ds.PatientBirthDate = birth_date
        if sex:
            ds.PatientSex = sex

        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1

        if study_desc:
            ds.StudyDescription = study_desc
        ds.SeriesDescription = series_description
        ds.Modality = modality
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.ImageOrientationPatient = [f"{v:.12g}" for v in image_orientation_patient]
        ds.PixelSpacing = [f"{v:.12g}" for v in pixel_spacing]
        ds.SliceThickness = f"{float(spacing[2]):.12g}"
        ds.SpacingBetweenSlices = f"{float(spacing[2]):.12g}"

        ipp = origin + k * float(spacing[2]) * slice_dir
        ds.ImagePositionPatient = [f"{float(v):.12g}" for v in ipp]

        if series_date:
            ds.SeriesDate = series_date
        if series_time:
            ds.SeriesTime = series_time

        ds.save_as(str(path), write_like_original=False)

    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "FrameOfReferenceUID": frame_of_ref_uid,
    }


def _sorted_dicom_slices(dicom_series_dir: Path):
    files = sorted(dicom_series_dir.glob("*.dcm"))
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_series_dir}")

    dss = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in files]

    iop = np.array([float(x) for x in dss[0].ImageOrientationPatient], dtype=float)
    row = iop[:3]
    col = iop[3:]
    normal = np.cross(row, col)

    positions = []
    for ds in dss:
        ipp = np.array([float(x) for x in ds.ImagePositionPatient], dtype=float)
        positions.append(float(np.dot(ipp, normal)))

    order = np.argsort(np.array(positions, dtype=float))
    dss = [dss[i] for i in order]
    return dss, row, col, normal


def build_series_geometry(dicom_series_dir: Path) -> Dict[str, object]:
    dss, row_dir, col_dir, normal = _sorted_dicom_slices(dicom_series_dir)

    pixel_spacing = np.array([float(x) for x in dss[0].PixelSpacing], dtype=float)
    sop_to_k = {str(ds.SOPInstanceUID): k for k, ds in enumerate(dss)}

    return {
        "dss": dss,
        "rows": int(dss[0].Rows),
        "cols": int(dss[0].Columns),
        "row_dir": np.array(row_dir, dtype=float),
        "col_dir": np.array(col_dir, dtype=float),
        "normal": np.array(normal, dtype=float),
        "ipp0": np.array([float(x) for x in dss[0].ImagePositionPatient], dtype=float),
        "row_spacing": float(pixel_spacing[0]),
        "col_spacing": float(pixel_spacing[1]),
        "sop_to_k": sop_to_k,
    }


def patient_xyz_to_rc(xyz: np.ndarray, geom: Dict[str, object]) -> Tuple[float, float]:
    """
    Convert patient coordinates to DICOM row/column coordinates.

    Row increases along IOP[3:6], column increases along IOP[0:3].
    """
    v = xyz - geom["ipp0"]
    row = float(np.dot(v, geom["col_dir"]) / geom["row_spacing"])
    col = float(np.dot(v, geom["row_dir"]) / geom["col_spacing"])
    return row, col


def extract_rois_from_rtstruct(rtstruct_path: Path) -> List[Tuple[int, str]]:
    ds_rt = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=True)
    if not hasattr(ds_rt, "StructureSetROISequence"):
        return []

    rois = []
    for roi in ds_rt.StructureSetROISequence:
        rois.append((int(roi.ROINumber), str(roi.ROIName)))
    return rois


def rasterize_roi_from_rtstruct(
    rtstruct_path: Path,
    dicom_series_dir: Path,
    roi_number: int,
) -> np.ndarray:
    """
    Rasterize one ROI from RTSTRUCT to a mask of shape (Y, X, Z).
    """
    ds_rt = pydicom.dcmread(str(rtstruct_path))
    geom = build_series_geometry(dicom_series_dir)
    mask_yxz = np.zeros((geom["rows"], geom["cols"], len(geom["dss"])), dtype=np.uint8)

    if not hasattr(ds_rt, "ROIContourSequence"):
        return mask_yxz

    roi_contour = None
    for contour in ds_rt.ROIContourSequence:
        if int(contour.ReferencedROINumber) == int(roi_number):
            roi_contour = contour
            break

    if roi_contour is None or not hasattr(roi_contour, "ContourSequence"):
        return mask_yxz

    for contour_seq in roi_contour.ContourSequence:
        if not hasattr(contour_seq, "ContourData") or len(contour_seq.ContourData) < 6:
            continue
        if (
            not hasattr(contour_seq, "ContourImageSequence")
            or len(contour_seq.ContourImageSequence) == 0
        ):
            continue

        sop_uid = str(contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID)
        if sop_uid not in geom["sop_to_k"]:
            continue
        slice_index = geom["sop_to_k"][sop_uid]

        pts = np.array(contour_seq.ContourData, dtype=float).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 3:
            continue

        rc = np.array([patient_xyz_to_rc(point, geom) for point in pts], dtype=float)
        poly = np.stack([rc[:, 1], rc[:, 0]], axis=1)
        poly = np.round(poly).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, geom["cols"] - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, geom["rows"] - 1)
        if poly.shape[0] < 3:
            continue

        slice_mask = np.zeros((geom["rows"], geom["cols"]), dtype=np.uint8)
        cv2.fillPoly(slice_mask, [poly], 1)
        mask_yxz[:, :, slice_index] |= slice_mask

    return mask_yxz


def rtstruct_to_resampled_rtstruct(
    original_dicom_series_dir: Path,
    original_rtstruct_path: Path,
    resampled_dicom_series_dir: Path,
    ref_resampled_img: sitk.Image,
    out_rtstruct_path: Path,
) -> None:
    """Resample RTSTRUCT contours onto the resampled CT grid and write a new RTSTRUCT."""
    src_img, _ = load_dicom_series(original_dicom_series_dir)
    rt_new = RTStructBuilder.create_new(dicom_series_path=str(resampled_dicom_series_dir))

    rois = extract_rois_from_rtstruct(original_rtstruct_path)
    if not rois:
        raise ValueError("No ROIs found in RTSTRUCT.")

    for roi_number, roi_name in rois:
        try:
            mask_yxz = rasterize_roi_from_rtstruct(
                rtstruct_path=original_rtstruct_path,
                dicom_series_dir=original_dicom_series_dir,
                roi_number=roi_number,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to rasterize ROI {roi_number} '{roi_name}': {exc}")
            continue

        if mask_yxz.sum() == 0:
            print(f"[INFO] ROI {roi_number} '{roi_name}' rasterized empty; skipping")
            continue

        mask_zyx = np.transpose(mask_yxz, (2, 0, 1)).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(mask_zyx)
        mask_sitk.SetSpacing(src_img.GetSpacing())
        mask_sitk.SetDirection(src_img.GetDirection())
        mask_sitk.SetOrigin(src_img.GetOrigin())

        mask_resampled_sitk = resample_to_reference(
            mask_sitk,
            ref_resampled_img,
            is_label=True,
            default_value=0.0,
        )

        mask_resampled_yxz = np.transpose(
            sitk.GetArrayFromImage(mask_resampled_sitk).astype(np.uint8),
            (1, 2, 0),
        ).astype(bool)

        rx, ry, rz = ref_resampled_img.GetSize()
        if mask_resampled_yxz.shape != (ry, rx, rz):
            raise ValueError(
                f"Resampled mask shape mismatch for '{roi_name}': "
                f"got {mask_resampled_yxz.shape}, expected {(ry, rx, rz)}"
            )

        rt_new.add_roi(mask=mask_resampled_yxz, name=roi_name)

    out_rtstruct_path.parent.mkdir(parents=True, exist_ok=True)
    rt_new.save(str(out_rtstruct_path))


def _resolve_series_dir(patient_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        path = patient_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Missing DICOM series under {patient_dir}. Tried: {', '.join(candidates)}"
    )


def _normalize_rtstruct_output_path(path: Path) -> Path:
    if path.suffix.lower() == ".dcm":
        return path
    return path / "rtstruct.dcm"


def resample_ct_mr_and_rtstruct_to_dicom(
    original_series_dir: Path,
    out_series_dir: Path,
    out_rtstruct_path: Optional[Path] = None,
    original_rtstruct_path: Optional[Path] = None,
    target_spacing: Tuple[float, float, float] = TARGET_SPACING,
    target_size_xy: Tuple[int, int] = TARGET_SIZE_XY,
) -> None:
    ct_dir = _resolve_series_dir(original_series_dir, ("CT",))
    mr_dir = _resolve_series_dir(original_series_dir, ("MR", "MR_in"))

    ct_img, ct_files = load_dicom_series(ct_dir)
    ct_ref = make_reference_image(ct_img, target_spacing, target_size_xy)
    ct_resampled = resample_to_reference(
        ct_img,
        ct_ref,
        is_label=False,
        default_value=-1024.0,
    )

    ct_out_dir = out_series_dir / "CT"
    write_resampled_dicom_series(
        resampled_img=ct_resampled,
        template_first_slice_path=ct_files[0],
        out_dir=ct_out_dir,
        series_description=(
            f"Resampled {target_spacing}mm {target_size_xy[0]}x{target_size_xy[1]}"
        ),
        keep_study_uid=True,
    )

    mr_img, mr_files = load_dicom_series(mr_dir)
    mr_ref = make_reference_image(mr_img, target_spacing, target_size_xy)
    mr_resampled = resample_to_reference(
        mr_img,
        mr_ref,
        is_label=False,
        default_value=0.0,
    )

    mr_out_dir = out_series_dir / "MR_in"
    write_resampled_dicom_series(
        resampled_img=mr_resampled,
        template_first_slice_path=mr_files[0],
        out_dir=mr_out_dir,
        series_description=(
            f"Resampled {target_spacing}mm {target_size_xy[0]}x{target_size_xy[1]}"
        ),
        keep_study_uid=True,
    )

    if original_rtstruct_path and out_rtstruct_path:
        rtstruct_to_resampled_rtstruct(
            original_dicom_series_dir=ct_dir,
            original_rtstruct_path=original_rtstruct_path,
            resampled_dicom_series_dir=ct_out_dir,
            ref_resampled_img=ct_resampled,
            out_rtstruct_path=_normalize_rtstruct_output_path(out_rtstruct_path),
        )

def main() -> None:
    dataset_type = "without_implant"
    input_root = Path(f"/media/nico/Extreme SSD/USZ/data_synthrad_2023/{dataset_type}")
    output_root = Path(
        f"/media/nico/Extreme SSD/USZ/data_synthrad_2023/processed_data/{dataset_type}"
    )

    patient_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        rtstruct_path = patient_dir / "RTSTRUCT" / "rtstruct.dcm"

        print(f"[INFO] Processing patient: {patient_id}")

        try:
            resample_ct_mr_and_rtstruct_to_dicom(
                original_series_dir=patient_dir,
                out_series_dir=output_root / patient_id,
                original_rtstruct_path=rtstruct_path if rtstruct_path.exists() else None,
                out_rtstruct_path=output_root / patient_id / "RTSTRUCT" / "rtstruct.dcm",
                target_spacing=TARGET_SPACING,
                target_size_xy=TARGET_SIZE_XY,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed for {patient_id}: {exc}")
            continue

    print(f"[INFO] Finished. Output root: {output_root}")


def process_patient(
    patient_dir: Path,
    intermediate_root: Path,
    output_root: Path,
    ct_filename: str,
    mr_filename: str,
    masks_subdir: str,
    build_rtstruct: bool,
    add_derived_contours: bool,
    prostate_roi_name: str,
    ptv_margin_mm: float,
    outer_mm: float,
    body_mask_template: Optional[str],
    body_threshold_ct: Optional[float],
    body_threshold_mr: Optional[float],
    body_roi_name: str,
    target_spacing: tuple[float, float, float],
    target_size_xy: tuple[int, int],
) -> None:
    patient_id = patient_dir.name
    ct_path = patient_dir / ct_filename
    mr_path = patient_dir / mr_filename

    if not ct_path.exists():
        raise FileNotFoundError(f"Missing CT NIfTI: {ct_path}")
    if not mr_path.exists():
        raise FileNotFoundError(f"Missing MR NIfTI: {mr_path}")

    patient_intermediate = intermediate_root / patient_id
    ct_out = patient_intermediate / "CT"
    mr_out = patient_intermediate / "MR"

    ct_img = sitk.ReadImage(str(ct_path))
    mr_img = sitk.ReadImage(str(mr_path))

    study_uid = _uid()
    frame_of_ref_uid = _uid()

    _write_dicom_series(
        image=ct_img,
        out_dir=ct_out,
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=_uid(),
        frame_of_ref_uid=frame_of_ref_uid,
        modality="CT",
        series_description="CT (from NIfTI)",
        series_number=1,
    )
    _write_dicom_series(
        image=mr_img,
        out_dir=mr_out,
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=_uid(),
        frame_of_ref_uid=frame_of_ref_uid,
        modality="MR",
        series_description="MR (from NIfTI)",
        series_number=2,
    )

    rtstruct_for_resampling: Optional[Path] = None

    if build_rtstruct:
        rtstruct_base = patient_intermediate / "RTSTRUCT" / "rtstruct_base.dcm"
        _build_rtstruct_from_masks(
            patient_dir=patient_dir,
            ct_dicom_dir=ct_out,
            out_file=rtstruct_base,
            masks_subdir=masks_subdir,
            ct_filename=ct_filename,
            mr_filename=mr_filename,
        )
        rtstruct_for_resampling = rtstruct_base

    body_mask_path = _resolve_body_mask_path(body_mask_template, patient_id)
    prostate_mask_path = _resolve_named_mask_path(
        patient_dir=patient_dir,
        masks_subdir=masks_subdir,
        ct_filename=ct_filename,
        mr_filename=mr_filename,
        mask_name=prostate_roi_name,
    )
    resample_ct_mr_and_rtstruct_to_dicom(
        original_series_dir=patient_intermediate,
        out_series_dir=output_root / patient_id,
        original_rtstruct_path=rtstruct_for_resampling,
        out_rtstruct_path=output_root / patient_id / "RTSTRUCT" / "rtstruct.dcm",
        target_spacing=target_spacing,
        target_size_xy=target_size_xy,
    )

    if build_rtstruct and add_derived_contours:
        final_rtstruct = output_root / patient_id / "RTSTRUCT" / "rtstruct.dcm"
        final_rtstruct_augmented = output_root / patient_id / "RTSTRUCT" / "rtstruct_with_derived.dcm"
        _augment_rtstruct_with_derived_contours(
            ct_dicom_series_dir=output_root / patient_id / "CT",
            mr_dicom_series_dir=output_root / patient_id / "MR_in",
            rtstruct_in_path=final_rtstruct,
            out_rtstruct_path=final_rtstruct_augmented,
            prostate_roi_name=prostate_roi_name,
            ptv_margin_mm=ptv_margin_mm,
            outer_mm=outer_mm,
            body_nifti_path=body_mask_path,
            body_threshold_ct=body_threshold_ct,
            body_threshold_mr=body_threshold_mr,
            body_roi_name=body_roi_name,
            prostate_nifti_path=prostate_mask_path,
            ct_nifti_path=ct_path,
        )
        final_rtstruct_augmented.replace(final_rtstruct)


def main(args: argparse.Namespace) -> int:
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    patient_dirs = sorted(p for p in args.input_root.iterdir() if p.is_dir())
    if args.patient:
        requested = set(args.patient)
        patient_dirs = [p for p in patient_dirs if p.name in requested]
        missing = sorted(requested - {p.name for p in patient_dirs})
        if missing:
            raise FileNotFoundError(f"Requested patient(s) not found: {missing}")

    if not patient_dirs:
        print("[WARN] No patient folders found to process.")
        return 0

    build_rtstruct = not args.skip_rtstruct
    if args.add_derived_contours and not build_rtstruct:
        raise ValueError("--add-derived-contours requires RTSTRUCT creation.")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"[INFO] Processing patient: {patient_id}")
        try:
            process_patient(
                patient_dir=patient_dir,
                intermediate_root=args.intermediate_root,
                output_root=args.output_root,
                ct_filename=args.ct_filename,
                mr_filename=args.mr_filename,
                masks_subdir=args.masks_subdir,
                build_rtstruct=build_rtstruct,
                add_derived_contours=args.add_derived_contours,
                prostate_roi_name=args.prostate_roi_name,
                ptv_margin_mm=args.ptv_margin_mm,
                outer_mm=args.outer_mm,
                body_mask_template=args.body_mask_template,
                body_threshold_ct=args.body_threshold_ct,
                body_threshold_mr=args.body_threshold_mr,
                body_roi_name=args.body_roi_name,
                target_spacing=tuple(args.target_spacing),
                target_size_xy=tuple(args.target_size_xy),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed for {patient_id}: {exc}")
            continue

    print(f"[INFO] Finished. Intermediate root: {args.intermediate_root}")
    print(f"[INFO] Finished. Output root: {args.output_root}")
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: NIfTI -> DICOM/RTSTRUCT -> optional derived contours -> resampled DICOM/RTSTRUCT."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--intermediate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patient", action="append", help="Patient ID to process. Repeatable.")
    parser.add_argument("--ct-filename", default="ct.nii.gz")
    parser.add_argument("--mr-filename", default="mr.nii.gz")
    parser.add_argument("--masks-subdir", default="TS_CT")
    parser.add_argument("--skip-rtstruct", action="store_true")
    parser.add_argument("--add-derived-contours", action="store_true")
    parser.add_argument("--prostate-roi-name", default="prostate")
    parser.add_argument("--ptv-margin-mm", type=float, default=5.0)
    parser.add_argument("--outer-mm", type=float, default=20.0)
    parser.add_argument(
        "--body-mask-template",
        help="Optional path template for BODY masks, e.g. '/data/body/mask_{patient_id}_3D_body.nii'.",
    )
    parser.add_argument(
        "--body-threshold-ct",
        type=float,
        default=-400.0,
        help="Optional CT threshold for generating BODY from CT and MR masks instead of an external NIfTI mask.",
    )
    parser.add_argument(
        "--body-threshold-mr",
        type=float,
        default=20.0,
        help="Optional MR threshold for generating BODY from CT and MR masks instead of an external NIfTI mask.",
    )
    parser.add_argument("--body-roi-name", default="BODY")
    parser.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        default=list(TARGET_SPACING),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument(
        "--target-size-xy",
        nargs=2,
        type=int,
        default=list(TARGET_SIZE_XY),
        metavar=("NX", "NY"),
    )

    raise SystemExit(main(parser.parse_args()))
