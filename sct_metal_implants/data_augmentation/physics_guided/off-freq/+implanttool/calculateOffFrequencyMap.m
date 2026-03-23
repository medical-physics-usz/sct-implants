% Calculates an off-resonance map, given a susceptibility distribution
function df = calculateOffFrequencyMap(chi, B0, gamma)
    import implanttool.structures.VoxelMap

    dField_3D = calculateFieldShift(chi.V, chi.voxelSize);
    dB = B0*dField_3D; % [T]
    values = gamma*dB / (2*pi); % [Hz]

    df = VoxelMap(values,chi.voxelSize);
    df.copyCoordinates(chi)
end
