% Reads the dicom files located at the specified directory. The measured
% values are thresholded to get the volume of the implant.
function v = dicomToVolume(filename)
    import implanttool.structures.Volume

    [V,spatial] = dicomreadVolume(filename);
    
%     perm = [1,3,2];
    perm = [1,2,3];

    V = squeeze(V(:,:,1,:));
    V = permute(V,perm);
    voxelSize = [spatial.PixelSpacings(1,:), spatial.PatientPositions(2,3)-spatial.PatientPositions(1,3)];
    voxelSize = voxelSize(perm);
    threshold = 12000;

    v = Volume(V,threshold,voxelSize);
end
