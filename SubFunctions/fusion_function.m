function fused_img = fusion_function(norm_img1, norm_img2, mask, ref_image)
    % Get the dimensions of the input images
    [rows, cols, bands] = size(norm_img1);
    
    % Compute distances for weight calculations
    weight1 = 1 ./ ((norm_img1 - ref_image).^2 + eps);  % Added eps to avoid division by zero
    weight2 = 1 ./ ((norm_img2 - ref_image).^2 + eps);  % Added eps to avoid division by zero

    % Normalize weights
    total_weights = weight1 + weight2;
    weight1 = weight1 ./ total_weights;
    weight2 = weight2 ./ total_weights;

    % Fuse the images
    fused_img = (weight1 .* norm_img1 + weight2 .* norm_img2);

    % Apply the mask - if mask is 1, use fusion; if mask is 0, use norm_img1
    fused_img = ((mask == 1) .* fused_img) + ((mask == 0) .* norm_img1);
end

% Example usage:
% norm_img1 = rand(100, 100, 3); % 3-band normalized image
% norm_img2 = rand(100, 100, 3); % 3-band normalized image
% mask = randi([0, 1], 100, 100); % Binary mask
% ref_image = rand(100, 100, 3); % Reference image

% fused_img = fusion_function(norm_img1, norm_img2, mask, ref_image);
% imshow(fused_img, []);
 
