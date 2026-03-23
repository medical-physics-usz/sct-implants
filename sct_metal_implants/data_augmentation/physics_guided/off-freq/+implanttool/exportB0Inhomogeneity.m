% The speficied off-resonance map is exported into a file compatible with
% the simulation environment.
function exportB0Inhomogeneity(df, filename, gamma)
    arguments
        df implanttool.structures.VoxelMap;
        filename;
        gamma = 267.51e6; % [s^-1/T]
    end
    B0 = implanttool.structures.VoxelMap(df.V./gamma * 2*pi*10000, df.voxelSize);
    B0.X = df.X;
    B0.Y = df.Y;
    B0.Z = -df.Z;
    B0.exportASCII(filename);
end
