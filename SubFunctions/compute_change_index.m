%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Code Header
% -------------------------------------------------------------------------
% Author: Dr. Armin Moghimi
% Title: Optimizing "Relative Radiometric Modeling: Fine-Tuning Strategies
%        Using Trust-Region Reflective and Genetic Algorithms for Residual
%        Error Minimization"
% Description: This MATLAB script performs various image processing tasks,
%              including image normalization, keypoint detection, regression
%              with different algorithms, and RMSE calculation for evaluation.
% 
% Components:
% - Path setup
% - Image loading and normalization
% - Keypoint detection using WSST_SURF
% - Descriptor matching
% - Regression with Trust-Region Reflective and Genetic Algorithms
% - RMSE calculation and comparison
% 
% Last Updated: December 2024
% 
% Notes:
% - Ensure that all required libraries and functions are available in the
%   specified paths.
% 
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ChangeIndex, Mask2] = compute_change_index(im_new, im_lwir1, Mask, alpha, beta)
    % COMPUTE_CHANGE_INDEX - Computes a fused change index using CVA and cosine similarity
    %
    % Syntax: [ChangeIndex, Mask2] = compute_change_index(im_new, im_lwir1, Mask, alpha, beta)
    %
    % Inputs:
    %   im_new   - New image (MxNxC array)
    %   im_lwir1 - Baseline image (MxNxC array)
    %   Mask     - Binary mask (MxN array)
    %   alpha    - Weight for CVA (scalar, optional, default=0.5)
    %   beta     - Weight for cosine similarity (scalar, optional, default=0.5)
    %
    % Outputs:
    %   ChangeIndex - Final fused change index (MxN array)
    %   Mask2       - Refined mask based on the change index (MxN array)

    % Default values for alpha and beta if not provided
    if nargin < 4
        alpha = 0.5;  % Default weight for CVA
    end
    if nargin < 5
        beta = 0.5;   % Default weight for cosine similarity
    end

    % Step 1: Ensure both images are of type 'double' for accurate computation
    if ~isa(im_new, 'double')
        im_new = double(im_new);
    end
    if ~isa(im_lwir1, 'double')
        im_lwir1 = double(im_lwir1);
    end

    % Step 2: Compute Change Vector Analysis (CVA)
    CVA = sqrt(sum((im_new - im_lwir1).^2, 3));  % Euclidean distance across channels
    CVA=CVA./max(CVA(:));
    CVA = CVA.* Mask;  % Avoid zero values and apply the mask

    % Step 3: Compute Cosine Similarity
    dot_product = sum(im_new .* im_lwir1, 3);   % Element-wise dot product along the 3rd dimension
    norm_new = sqrt(sum(im_new.^2, 3));         % L2 norm of the first image
    norm_lwir1 = sqrt(sum(im_lwir1.^2, 3));     % L2 norm of the second image

    % Handle division by zero by adding a small constant (epsilon)
    epsilon = 1e-10;  % Small constant to avoid division by zero
    cos_similarity = dot_product ./ (norm_new .* norm_lwir1 + epsilon);  % Cosine similarity

    % Step 4: Normalize the cosine similarity to the range [0, 1]
    cos_similarity_normalized = ((cos_similarity + 1) / 2);  % Convert cosine similarity from [-1, 1] to [0, 1]

    % Step 5: Fuse CVA and cosine similarity into a combined change index
    % Compute the final change index by fusing CVA and cosine similarity
    ChangeIndex = alpha * CVA + beta * ((1 - cos_similarity_normalized).*Mask);  % Combine with inverse similarity for dissimilarity

    % Step 6: Rescale the change index to the range [0, 1] for better visualization
    ChangeIndex = rescale(ChangeIndex, 0, 1);

    % Step 7: Apply thresholding to highlight areas of significant change
    ChangeIndex_nonzero = nonzeros(ChangeIndex);  % Remove zero values for thresholding
    thresh = multithresh(ChangeIndex_nonzero, 2);  % Multi-level thresholding to find two thresholds

    % Step 8: Refine the mask based on the threshold
    Mask2 = (ChangeIndex < ((thresh(2)+thresh(1))./2)) .* Mask;  % Create the refined mask based on the second threshold

    % Optional: Visualize or output the change index and the refined mask
    imshow((1- cos_similarity_normalized).*Mask,[])
    figure, 
    imshow(CVA,[])
    figure, 
    imshow(ChangeIndex,[]);
    % imshow(Mask2);
end
