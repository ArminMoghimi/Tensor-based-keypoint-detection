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
function [fused_img, num_levels]= fusion_laplacian_pyramid_new(norm_img1, norm_img2, ref_image)
    % Fuses two images using Laplacian pyramids and a reference image
    % Inputs:
    %   norm_img1 - First normalized input image
    %   norm_img2 - Second normalized input image
    %   ref_image - Reference image for weight calculation

    % Calculate the entropy of the images
    entropy_img1 = entropy(norm_img1);
    entropy_img2 = entropy(norm_img2);
    avg_entropy = (entropy_img1 + entropy_img2) / 2;
    min_entropy=min(entropy_img1,entropy_img2);
    max_entropy=max(entropy_img1,entropy_img2);
    % Determine the number of pyramid levels based on the entropy and image size
    max_levels = floor(log2(min(size(norm_img1, 1), size(norm_img1, 2))))/4;  % Max levels based on size
    num_levels = 1+abs(floor((max_levels - max_entropy) / (avg_entropy+min_entropy+1)));  % Formula based on entropy
    num_levels = max(num_levels,2);  % Ensure number of levels is between 2 and the max

    % Build the Laplacian pyramids for the two images and the reference image
    pyr1 = build_laplacian_pyramid(norm_img1, num_levels);
    pyr2 = build_laplacian_pyramid(norm_img2, num_levels);
    ref_pyr = build_laplacian_pyramid(ref_image, num_levels);

    % Initialize the fused pyramid
    fused_pyr = cell(num_levels, 1);

    % Loop through pyramid levels and fuse using gradients
    for i = 1:num_levels
        % Compute differences from the reference image
        diff1_ref = pyr1{i} - ref_pyr{i};
        diff2_ref = pyr2{i} - ref_pyr{i};
        % Compute fusion weights based on the gradients
        weight1 = (1 ./ ((diff1_ref).^2+ eps));
        weight2 = (1 ./ ((diff2_ref).^2+ eps));
        % Normalize weights
        total_weights = weight1 + weight2;
        weight1 = weight1 ./ total_weights;
        weight2 = weight2 ./ total_weights;

        % Fuse the current pyramid level
        fused_pyr{i} = weight1 .* pyr1{i} + weight2 .* pyr2{i};
    end

    % Reconstruct the fused image from the fused pyramid
    fused_img = reconstruct_laplacian_pyramid(fused_pyr, num_levels);

    % Compute additional weights for final image fusion
    weight1 = 1 ./ ((fused_img - ref_image).^2 + eps);
    weight2 = 1 ./ ((norm_img1 - ref_image).^2 + eps);
    weight3 = 1 ./ ((norm_img2 - ref_image).^2 + eps);

    % Normalize final weights
    total_weights = weight1 + weight2 + weight3;
    weight1 = weight1 ./ total_weights;
    weight2 = weight2 ./ total_weights;
    weight3 = weight3 ./ total_weights;

    % Combine fused image with input images using weights
    fused_img = weight1 .* fused_img + weight2 .* norm_img1 + weight3 .* norm_img2 ;
end


%Pyramid construction function (same as before)
function pyr = build_laplacian_pyramid(img, num_levels)
    pyr = cell(num_levels, 1);
    current_img = img;

    for i = 1:num_levels
        next_img = imresize(current_img, 0.5);  % Downsample the image
        pyr{i} = current_img - imresize(next_img, size(current_img, [1, 2]));  % Create Laplacian layer
        current_img = next_img;  % Update current image for next level
    end
    pyr{num_levels} = current_img;  % Store the last level
end

% Pyramid reconstruction function (same as before)
function img = reconstruct_laplacian_pyramid(pyr, num_levels)
    img = pyr{num_levels};  % Start with the last level
    for i = num_levels-1:-1:1
        img = pyr{i} + imresize(img, size(pyr{i}, [1, 2]));  % Add each level to reconstruct the image
    end
end
