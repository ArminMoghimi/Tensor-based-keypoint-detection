function fused_img = fusion_function(norm_img1, norm_img2, mask, ref_image)
% Get the dimensions of the input images
[~,~,bands] = size(norm_img1);
for i=1:bands
    % Compute distances for weight calculations
    weight1 = 1 ./ ((norm_img1(:,:,i) - ref_image(:,:,i)).^2 + eps);  % Added eps to avoid division by zero
    weight2 = 1 ./ ((norm_img2(:,:,i) - ref_image(:,:,i)).^2 + eps);  % Added eps to avoid division by zero
    % Normalize weights
    total_weights = weight1 + weight2;
    weight1 = weight1 ./ total_weights;
    weight2 = weight2 ./ total_weights;
    % Fuse the images
    fused_img(:,:,i) = (weight1 .* norm_img1(:,:,i) + weight2 .* norm_img2(:,:,i));
    % Apply the mask - if mask is 1, use fusion; if mask is 0, use norm_img1
    fused_img(:,:,i) = ((mask == 1) .* fused_img(:,:,i)) + ((mask == 0) .* norm_img1(:,:,i));
end
end
