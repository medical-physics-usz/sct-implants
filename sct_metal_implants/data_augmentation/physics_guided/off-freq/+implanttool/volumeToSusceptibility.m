% Given an array of volumes, a background suscepibility and susceptibility
% values for each volume, a susceptibility map is returned.
% The returned susceptibility map is relative to the background.
function chi = volumeToSusceptibility(volumes, chi_background, chi_volumes)
    import implanttool.structures.VoxelMap

%     values = ones(volumes(1).N) * chi_background;
    values = zeros(volumes(1).N);
    for i = 1:length(volumes)
%         values(volumes(i).V) = chi_volumes(i);
        values(volumes(i).V) = chi_volumes(i) - chi_background;
    end
    chi = VoxelMap(values,volumes(1).voxelSize);
    chi.copyCoordinates(volumes(1))
end
