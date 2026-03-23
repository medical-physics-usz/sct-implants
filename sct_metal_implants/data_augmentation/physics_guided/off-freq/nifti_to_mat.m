%%  Create MAT from NIFTI

donorPatient = '1PA155';
receiverPatient = '1PA101';
laterality = 'L';

inputPath = fullfile( ...
    "/media/nico/Extreme SSD/USZ/data_augmentation_balgrist/metal_masks_usz/synthrad", ...
    "mask_" + receiverPatient + "_aug_by_" + donorPatient + "_" + laterality + "_3D_metal.nii");

outputPath = fullfile( ...
    "/media/nico/Extreme SSD/USZ/data_augmentation_balgrist/metal_masks_MAT/synthrad", ...
    "mask_" + receiverPatient + "_aug_by_" + donorPatient + "_" + laterality + "_3D_metal.mat");

% Create output path if not exists
outDir = fileparts(outputPath);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Read the NIfTI file
V = niftiread(inputPath);

% Swap X and Y
V = permute(V,[2 1 3]);

% Flip Z
V = flip(V,3);

% Convert to single precision to save space
V = single(V);

% Save to .mat file
save(outputPath, 'V');

