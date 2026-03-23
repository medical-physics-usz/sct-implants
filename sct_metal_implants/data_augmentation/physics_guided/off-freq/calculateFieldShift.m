%% Fourier Based Field-shift Calculation for MRI
% - k-space Filtering (KF) implementation
% - implemented with 'virtual' zero-padding

%% Written 22-06-2012 by Job Bouwman (jgbouwman@hotmail.com)
% Update 14-06-2013: to cope with anisotropic voxel size
% Update 26-01-2014: implemented a numerical trick to further increase the 
%           calculation speed with ~30% . As all distributions are real
%           in the spatial domain, the imaginary channel of the convolution
%           can be used for a second convolution. This channel is exploited 
%           to calculate the aliasing distribution.
% Update 20-10-2014: the use of nested functions allows larger matrices
%           again 13% faster, by using more efficient nested functions 

%% method description (including scientific references)
% This is an efficient implementation of the Fourier-based procedure to
% calculate the susceptibility-induced field-shift if an object 
% with known susceptibility distribution (dChi_3D) is placed within 
% a strong homogeneous magnetic field (B0).   

% This method is based on the publication of: 
% Salomir et al, "A fast calculation method for magnetic field 
% inhomogeneity due to an arbitrary distribution of bulk susceptibility"
% Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering, 
% Volume 19B, Issue 1, pages 26–34, 2003

% However, the original method needs (inefficient zero-padding)
% Cheng YCN, Neelavalli J, Haacke EM. "Limitations of calculating field
% distributions and magnetic susceptibilities in MRI using a Fourier
% based method". Phys Med Biol 2009;54:1169–1189

% In the proposed method here, aliasing is removed by estimating 
% this artifact using an additional convolution in a lower resolution, 
% which is feasible due to the smooth decay of the dipole function. 
% The article describing the principle of low resolution aliasing 
% subtraction:
% Job G. Bouwman, Chris J.G. Bakker. "Alias subtraction more efficient than
% conventional zero-padding in the Fourier-based calculation of the 
% susceptibility induced perturbation of the magnetic field in MR" 
% 2012, Magnetic Resonance in Medicine 68:621630 (2012). 
% Please refer to this article when using this script.


%% How to use this file:
% dChi_3D is your 3D susceptibility distribution
%   * the third dimension of dChi_3D is the one parallel to B0.
%   * the values of your dChi_3D are assumed to be relative to the 
%     dChi of the embedding medium. In practice this will either be: 
%     - relative to the dChi of air (0.36 ppm) if you want to calculate the 
%       field for example in a complete head model
%     - relative to the dChi of tissue (-9.05 ppm) if you want to calculate
%       the local effect of for example a small implant embedded in tissue.
% voxelSize (either in mm, centimeter, or inches)
%    * we use the matlabconvention: [dy, dx, dz]

% Example:
%   B0 = 3; % tesla;
%   voxelSize = [1 1 2]; % mm
%   gyromagneticRatio = 2*pi*42.58e6

%   dB = B0*calculateFieldShift(dChi_3D, voxelSize);
%   dOhmega = dB*gyromagneticRatio;

function dField_3D = calculateFieldShift(varargin)

  
    % Checking the input parameters:
    if or(nargin < 1, nargin > 2)
        error('Please supply two arguments (dChi_3D and voxelSize).');
    end
    dChi_3D = varargin{1};
    if or(length(size(dChi_3D)) ~= 3,  min(size(dChi_3D)) < 2)
        error('Input dChi_3D must be of dimension 3.');
    end
    
    if nargin == 2
        voxelSize = squeeze(varargin{2});
        if or(size(voxelSize,2)~=3, size(voxelSize,1)~=1)
            error('voxelSize must be a vector of length 3.');
        end
    else
        display(['In calculateFieldShift.m the voxelSize was not '...
                 'specified. The script assumed it to be isotropic:']);
        voxelSize = [1,1,1]
    end

    % making sure that the size of the distribution in each dimension 
    % is a multiple of four:
    [NyInput, NxInput, NzInput] = size(dChi_3D);
    if mod(NxInput,4)+mod(NyInput,4)+mod(NzInput,4) > 0
        conventionalPad = mod(4-...
            [mod(NyInput,4),mod(NxInput,4),mod(NzInput,4)], 4);
        dChi_3D = padarray(dChi_3D,conventionalPad,'post');
    end
    FOV = voxelSize.*size(dChi_3D);
    
    %% The Fourier-based convolution:
    % 1) The forward Fourier Transform (FT):
        % NB: Instead of doing an extra convolution to calculate the 
        % effect of aliasing (as proposed in the article), here we use 
        % the 'imaginary channel' to do this. 
        % This is done by downscaling the artificial environment, and
        % adding this as an imaginary component to the original 
        % (unpadded) susceptibility distribution. Together this is 
        % referred to as the "DUAL" distribution:

        %% flip 

        dChi_3D = flip(dChi_3D, 3); % if B0 is along the 3rd dimension

        FT_dChi_3D_DUAL = fftn(dChi_3D + 1i*Zc(D(dChi_3D))); 
                clear dChi_3D;
    % 2) multiplication with the dipole function: 
        FT_dField_3D = FT_dChi_3D_DUAL.*KF_kernel(FOV,size(FT_dChi_3D_DUAL)); 
                clear FT_dChi_3D_DUAL;
    % 3) inverse FT to spatial domain:
        dField_3D_DUAL = ifftn(FT_dField_3D);
                clear FT_dField_3D;
    % 4) Subtracting the upscaled, cropped center of the aliasing: 
        dField_3D = real(dField_3D_DUAL) - UC(imag(dField_3D_DUAL));
               clear dField_3D_DUAL;
  
    % returning the field shift in the same size:
    if mod(NxInput,4)+mod(NyInput,4)+mod(NzInput,4) > 0
        dField_3D = dField_3D(1:NyInput, 1:NxInput, 1:NzInput);
    end
    
    % END OF MAIN FUNCTION
    
    
    % NESTED functions:
    % (before the update of 20-10-2014 these were subfunctions)
    function kernel = KF_kernel(FOV,N)
        % FOV : field of view in x, y, and z directions
        % N   : no of samples in kx, ky, kz
        kx_squared = ifftshift((-N(2)/2:N(2)/2-1)/FOV(2)).^2;
        ky_squared = ifftshift((-N(1)/2:N(1)/2-1)/FOV(1)).^2;
        kz_squared = ifftshift((-N(3)/2:N(3)/2-1)/FOV(3)).^2;

        [kx2_3D,ky2_3D,kz2_3D] = meshgrid(kx_squared,ky_squared,kz_squared);

        kernel = 1/3 - kz2_3D./(kx2_3D + ky2_3D + kz2_3D);    
        kernel(1,1,1) = 0;
    end

    function DX = D(X)
        % downscales a 3D distribution by factor two, by collapsing each cube  
        % of eight voxels into one single voxel (average value) 
        DX =      X(1:2:end,1:2:end,1:2:end);
        DX = DX + X(1:2:end,1:2:end,2:2:end);
        DX = DX + X(1:2:end,2:2:end,1:2:end);
        DX = DX + X(1:2:end,2:2:end,2:2:end);
        DX = DX + X(2:2:end,1:2:end,1:2:end);
        DX = DX + X(2:2:end,1:2:end,2:2:end);
        DX = DX + X(2:2:end,2:2:end,1:2:end);
        DX = DX + X(2:2:end,2:2:end,2:2:end);
        DX = DX/8;
    end

    function ZcX = Zc(X)
        % This operator is complementary to zero-padding: the result is 
        % a void space embedded in an environment of copies of X. 
        % The resulting 3D distribution is 2 times as large 
        % as the original in each dimension
        Nx = 2*size(X, 2);  mid_x   = (Nx/4+1):(Nx*3/4);     
        Ny = 2*size(X, 1);  mid_y   = (Ny/4+1):(Ny*3/4);   
        Nz = 2*size(X, 3);  mid_z   = (Nz/4+1):(Nz*3/4);   
        ZcX = repmat(circshift(X, [Ny/4, Nx/4, Nz/4]), [2 2 2]); 
        ZcX(mid_y, mid_x, mid_z) = 0;
    end

    function center = UC(domain)
        % UC = Upscaling + Cropping:
        % both crops as upscales (factor 2) the central part of a larger 
        % distribution. Therefore the size of input ('domain') and output
        % ('center') are the same. 
        Nx = size(domain,2); Mx=round(1/4+Nx/4+(1:Nx)/2); Sx=Mx+(-1).^(1:Nx);
        Ny = size(domain,1); My=round(1/4+Ny/4+(1:Ny)/2); Sy=My+(-1).^(1:Ny);
        Nz = size(domain,3); Mz=round(1/4+Nz/4+(1:Nz)/2); Sz=Mz+(-1).^(1:Nz);

        center =          27/64*domain(My,Mx,Mz);
        center = center +  9/64*domain(My,Mx,Sz);
        center = center +  9/64*domain(My,Sx,Mz);
        center = center +  3/64*domain(My,Sx,Sz);
        center = center +  9/64*domain(Sy,Mx,Mz);
        center = center +  3/64*domain(Sy,Mx,Sz);
        center = center +  3/64*domain(Sy,Sx,Mz);
        center = center +  1/64*domain(Sy,Sx,Sz);
    end
end

