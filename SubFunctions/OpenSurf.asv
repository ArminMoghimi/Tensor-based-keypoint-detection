function ipts=OpenSurf(img,Options)
% OPEN SURF (Open Source Speeded Up Robust Features) MATLAB Implementation
%--------------------------------------------------------------------------
% This function, OpenSurf, is an implementation of the SURF (Speeded Up
% Robust Features) algorithm. SURF detects landmark points in an image and
% describes these points with a vector that is robust against rotation,
% scaling, and noise. It can be used for tasks such as image alignment,
% registration, and 3D reconstructions.
%
% This MATLAB implementation of Surf is a direct translation of the
% OpenSurf C# code authored by Chris Evans. The MATLAB code provides
% exactly the same results as the original C# implementation. Chris Evans
% has graciously permitted the publication of this code under the
% (MathWorks) BSD license.
%
% Chris Evans' OpenSURF implementation is well-structured and inclusive,
% making it one of the best implementations of SURF. Further details,
% evaluations, and the original C# and C++ code can be found on Chris
% Evans' website: http://www.chrisevansdev.com/opensurf/
%
% Authorship and Modification:
% - Original Function by D. Kroon, University of Twente (July 2010)
% - Modified by A. Moghimi to generate WSST-SURF, as presented in the paper:
% Armin Moghimi, Turgay Celik, Ali Mohammadzadeh (2022)
% "Tensor-based keypoint detection and switching regression model for
% relative radiometric normalization of bitemporal multispectral images,"
% International Journal of Remote Sensing, 43:11, 3927-3956,
% DOI: 10.1080/01431161.2022.2102951
%
% Usage:
%   Ipts = OpenSurf(I, Options)
%
% Inputs:
%   I       - The 2D input image (color or grayscale)
%   Options - A struct with options (see below)
%
% Outputs:
%   Ipts - A structure with information about all detected landmark points
%          - Ipts.x, Ipts.y: Landmark position
%          - Ipts.scale: Scale of the detected landmark
%          - Ipts.laplacian: Laplacian of the landmark neighborhood
%          - Ipts.orientation: Orientation in radians
%          - Ipts.descriptor: Descriptor for corresponding point matching
%
% Options:
%   Options.verbose: If true, useful information is displayed (default false)
%   Options.upright: Boolean for non-rotation invariant result (default false)
%   Options.extended: Add extra landmark point information to the descriptor (default false)
%   Options.tresh: Hessian response threshold (default 0.0002)
%   Options.octaves: Number of octaves to analyze (default 5)
%   Options.init_sample: Initial sampling step in the image (default 2)

% Example 1, Basic Surf Point Detection
% % Load image
%   I=imread('TestImages/test.png');
% % Set this option to true if you want to see more information
%   Options.verbose=false; 
% % Get the Key Points
%   Ipts=OpenSurf(I,Options);
% % Draw points on the image
%   PaintSURF(I, Ipts);
%
% Example 2, Corresponding points
% % See, example2.m
%
% Example 3, Affine registration
% % See, example3.m
%
% Function is written by D.Kroon University of Twente (July 2010)
% Function is modified by A.Moghimi to generate WSST-SURF as presented in
% following paper:




% Add subfunctions to Matlab Search path
functionname='OpenSurf.m';
functiondir=which(functionname);
functiondir=functiondir(1:end-length(functionname));
addpath([functiondir '/SubFunctions'])
       
    % Process inputs
    defaultOptions = struct('tresh', 0.0002, 'octaves', 5, 'init_sample', 2, 'upright', false, 'extended', false, 'verbose', false);
    
    if ~exist('Options', 'var')
        Options = defaultOptions;
    else
        tags = fieldnames(defaultOptions);
        for i = 1:length(tags)
            if ~isfield(Options, tags{i})
                Options.(tags{i}) = defaultOptions.(tags{i});
            end
        end
        if length(tags) ~= length(fieldnames(Options))
            warning('OpenSurf:unknownoption', 'unknown options found');
        end
    end
    
    % Create Integral Image
    for i = 1:size(img, 3)
        i_img(:, :, i) = IntegralImage_IntegralImage(img(:, :, i));
    end
    
    % Calculating spectral band weights
    if size(img, 3) > 1
        [lambda1, ~] = sMulti_Corr2G(img, 512);
        lambda1 = abs(lambda1);
    else
        lambda1 = 1;
    end
    
    % Construct a single weighted spectral image for fast calculating SURF Descriptor
    Z = zeros(size(i_img, 1), size(i_img, 2));
    for i = 1:size(img, 3)
        Z = Z + (100 * lambda1(:, i)) .* img(:, :, i);
    end
    Out2 = IntegralImage_IntegralImage(Z / max(Z(:)));
    
    % FastHessian parameters
    FastHessianData.thresh = Options.tresh;
    FastHessianData.octaves = Options.octaves;
    FastHessianData.init_sample = Options.init_sample;
    FastHessianData.img = i_img;
    FastHessianData.lambda = lambda1;
    
    % Keypoint extraction
    ipts = FastHessian_getIpoints(FastHessianData, Options.verbose);
    
    % Describe the keypoints
    if ~isempty(ipts)
        ipts = SurfDescriptor_DecribeInterestPoints(ipts, Options.upright, Options.extended, i_img, lambda1, Out2, Options.verbose);
    end
end