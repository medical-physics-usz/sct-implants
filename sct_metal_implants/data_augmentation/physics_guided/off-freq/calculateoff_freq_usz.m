%%  import MAT file of the bianry mask 

donorPatient = '1PA155';
receiverPatient = '1PA101';
laterality = 'L';

%% Input path to metal mask stored as .mat file created with nifti_to_mat.m script
inputPath = fullfile("/media/nico/Extreme SSD/USZ/data_augmentation_balgrist/metal_masks_MAT/synthrad", ...
    "mask_" + receiverPatient + "_aug_by_" + donorPatient + "_" + laterality + "_3D_metal.mat");
hip = load(inputPath);

%% Output path to store B0 map
outputPath = fullfile("/media/nico/Extreme SSD/USZ/data_augmentation_balgrist/B0_maps/synthrad", ...
    "B0_" + receiverPatient + "_aug_by_" + donorPatient + "_" + laterality + "_3D");

%% MR parameters
B0 = 1.5;
gamma = 267.51e6;

%% create voxel map
volume = implanttool.structures.Volume(hip.V,0.5,0.5);
phantom = implanttool.structures.VoxelMap(volume,0.5);

%% segment stem from acetabular ball 
volumeList2 = implanttool.segmentVolume(phantom.V);
%% assign material suscpetibility 
chi = implanttool.volumeToSusceptibility(volumeList2,-9.05e-06,[154e-6,154e-6]); % both Titanium alloy
%%
chi2 = implanttool.structures.VoxelMap(chi.V,[0.5 0.5 0.5]);
%%
df = implanttool.calculateOffFrequencyMap(chi2,B0,gamma);
implanttool.exportB0Inhomogeneity(df,outputPath + ".dat",gamma); % this is needed to get the map in Hz
%% nifti save
volumeViewer(df.V)
niftiwrite(df.V , outputPath);
%% other possible 
chi = implanttool.volumeToSusceptibility(volumeList2,-9.05e-06,[900e-6,154e-6]); % cobal chrome + titanium alloy stem
chi = implanttool.volumeToSusceptibility(volumeList2,-9.05e-06,[154e-6,154e-6]); % both Titanium alloy


%% 
% always check that the orientation is axial coronal sagittal in clock wise
%  order when opening the df.V on volumeviewer
%% the field shift is from https://pubmed.ncbi.nlm.nih.gov/22711589/ 
%% for the suscpetibilities : https://pmc.ncbi.nlm.nih.gov/articles/PMC5529184/ 
%% this code is adapted from Jonas Wahlen work please cite this ISMRM abstract from him https://archive.ismrm.org/2024/0772_LUiM26Rak.html 