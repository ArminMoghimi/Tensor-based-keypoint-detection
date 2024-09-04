function [rmseValues,avgRMSE]= calculateAverageRMSE(image1, image2, mask)
%  The code wriiten by Armin Moghimi (2022)
% Reference:
% Armin Moghimi, Turgay Celik, Ali Mohammadzadeh (2022)
% "Tensor-based keypoint detection and switching regression model for
% relative radiometric normalization of bitemporal multispectral images,"
% International Journal of Remote Sensing, 43:11, 3927-3956,
% DOI: 10.1080/01431161.2022.2102951
    % Input:
    %   - image1: First image
    %   - image2: Second image
    %   - mask: Binary mask (1 for regions of interest, 0 for background)

    % Ensure all inputs have the same size
    assert(all(size(image1) == size(image2)) && all(size(mask) == size(mask)), 'Inputs must have the same size.');

    % Calculate RMSE for each band in masked regions
    numBands = size(image1, 3);
    rmseValues = zeros(1, numBands);

    for band = 1:numBands
        % Extract individual bands
        band1 = double(image1(:, :, band));
        band2 = double(image2(:, :, band));

        % Apply the mask to the bands
        maskedBand1 = band1(mask == 1);
        maskedBand2 = band2(mask == 1);

        % Calculate RMSE for the current band in masked regions
        rmseValues(band) = sqrt(mean((maskedBand1 - maskedBand2).^2));
    end

    % Calculate the average RMSE across all bands
    avgRMSE = mean(rmseValues);
end
