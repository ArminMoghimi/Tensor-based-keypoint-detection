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
% Last Updated: September 4, 2024
% 
% Notes:
% - Ensure that all required libraries and functions are available in the
%   specified paths.
% 
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;                
clear;              
close all;         

% ------------------------------------ Set path
fp = mfilename('fullpath');           % Get the full path of the current script
rootdir = fileparts(fp);              % Extract the directory path
p{1} = fullfile(rootdir, 'vlfeat');   % Define the path to the 'vlfeat' folder
p{2} = fullfile(rootdir, 'SubFunctions'); % Define the path to 'SubFunctions' folder
p{3} = fullfile(rootdir, 'Dataset');  % Define the path to the 'Dataset' folder
for i = 1:numel(p)                    % Loop over each path
    addpath(genpath(p{i}));           % Add the paths (and their subdirectories) to MATLAB
end

% Compile and configure mexopencv to work with the specified OpenCV installation
mexopencv.make('opencv_path', 'C:\Users\moghimi\Desktop\armin\opencv-3.4.1-vc14_vc15_2\opencv\build')
% Add path to the mexopencv library, enabling MATLAB to access OpenCV functions
addpath('C:\Users\moghimi\Desktop\armin\mexopencv-master')

%% Read multispectral Image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[name, path] = uigetfile('*.tif', 'Select a Sub. image', 'Subject'); 
I11 = (imread(fullfile(path, name))) + 0.025;    % Read the image and add a small offset for processing
[name, path] = uigetfile('*.tif', 'Select a Ref. image', 'Reference'); 
I22 = (imread(fullfile(path, name))) + 0.025;    % Read the image and add a small offset for processing
[filename, pathname] = uigetfile('*.mat', 'Select a MAT File', 'Test'); 
load(fullfile(pathname, filename));  % Load the selected .mat file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Image Normalization in [0 ,1]
I1 = im2double(I11);    % Normalize the Subject image to the range [0, 1]
I2 = im2double(I22);    % Normalize the Reference image to the range [0, 1]
disp('Image was loaded successfully'); % Display success message

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% WSST_SURF parameter setting
Options.upright = false;    % Set the SURF option to consider rotation
Options.tresh = 0.000001;   % Set the threshold for SURF keypoints
Options.octaves = 5;        % Set the number of octaves for SURF (scale levels)
disp('WSST_SURF parameters were set'); % Display success message

%% Keypoint detection and description using WSST_SURF
disp('Starting keypoint detection and description for Image pair');
Ipts1 = OpenSurf(I1, Options);  % Detect SURF keypoints in the Subject image
disp('Keypoints detected for Subject image successfully');
Ipts2 = OpenSurf(I2, Options);  % Detect SURF keypoints in the Reference image
disp('Keypoints detected for Reference image successfully');

%% Put the landmark descriptors in a matrix
D1 = reshape([Ipts1.descriptor], 64, []);  % Extract and reshape descriptors from the Subject image
D2 = reshape([Ipts2.descriptor], 64, []);  % Extract and reshape descriptors from the Reference image
A = cell2mat(struct2cell(Ipts1'));          % Convert Subject keypoint locations to a matrix
A = A(1:2, :);                              % Keep only the X and Y coordinates
B = cell2mat(struct2cell(Ipts2'));          % Convert Reference keypoint locations to a matrix
B = B(1:2, :);                              % Keep only the X and Y coordinates

%% Matching descriptor
disp('Matching procedure.....');
[ptsObj1, ptsScene1] = matching(A, B, D1, D2);   % Match descriptors between Subject and Reference images
disp('Matching was successfully done.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('RNN procedure......');
appendimages(I1(:,:,1), I2(:,:,1), ptsObj1, ptsScene1); % Display the matched keypoints
[Normalized_Image, ~, ~, ~, ~, ~] = RCS_Regression(round(ptsScene1), round(ptsObj1), double(I11) + 0.0025, double(I22) + 0.0025); % Apply regression-based normalization
[im_new, R_2, B, M, ptsObj2, ptsScene2, t_score] = RCS_jointRegression(round(ptsScene1), round(ptsObj1), double(I11) + 0.0025, double(I22) + 0.0025); % Apply joint regression
disp('RNN was successfully done.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure, imshow(uint8(I11(:,:,1:3)));    % Display the Subject image
title('Subject Image');
figure, imshow(uint8(I22(:,:,1:3)));    % Display the Reference image
title('Reference Image');
figure, imshow(uint8(Normalized_Image(:,:,1:3))); % Display the normalized Subject image
title('Normalized Subject Image');

%% Optimization process; find Mask for No data in reference image
Mask1 = I22(:,:,1) > 0;    % Create a mask for valid data in the Reference image
Mask = imcomplement(im_new(:,:,1) == 0);  % Create a mask for valid data in the normalized image
Mask = Mask1 .* Mask;  % Combine masks to find regions with valid data in both images
im_lwir1 = Mask .* (double(I22) + 0.025); % Apply mask to the Reference image
[m1, m2, m3] = size(im_lwir1);    % Get the dimensions of the masked image

%% CVA image generation 
CVA = (sqrt(sum((double(im_new) - double(im_lwir1)).^2, 3)) + 1) .* Mask; % Compute Change Vector Analysis (CVA) image
CVA = rescale(CVA, 0, 1);   % Rescale CVA image to the range [0, 1]
thresh = multithresh(nonzeros(CVA), 2);  % Apply multi-level thresholding
Mask2 = (CVA < thresh(2)) .* Mask;  % Refine the mask based on thresholding
ptsScene1_test = Mask2 .* im_lwir1;  % Apply refined mask to the Reference image
ptsScene1_test = reshape(ptsScene1_test, m1 * m2, m3);  % Reshape the masked image into a 2D matrix
ptsObj1_test = Mask2 .* im_new;  % Apply refined mask to the normalized image
ptsObj1_test = reshape(ptsObj1_test, m1 * m2, m3);  % Reshape the masked normalized image into a 2D matrix

Num = 10000;  % Set the number of samples for regression

% Perform Regression with Trust-Region Reflective Algorithm (Linear)
[normalizedImg_trainTRR_Li, R_squared, slope, intercept, fitParameters, Initial_ImageT_Li] = RCS2_RegressionTRR(ptsScene1_test, ptsObj1_test, im_new, im_lwir1, Num, 'linear');

% Perform Regression with Trust-Region Reflective Algorithm (Nonlinear)
[normalizedImg_trainTRR_nLi, R_squared, slope, intercept, fitParameters, Initial_ImageT_nLi] = RCS2_RegressionTRR(ptsScene1_test, ptsObj1_test, im_new, im_lwir1, Num, 'nonlinear');

% Perform Regression with Genetic Algorithm (Linear)
[normalizedImg_trainGA_Li, R_squared, slope, intercept, fitParameters, Initial_ImageG_Li] = RCS2_RegressionGA37(ptsScene1_test, ptsObj1_test, im_new, Num, 'linear');

% Perform Regression with Genetic Algorithm (Nonlinear)
[normalizedImg_trainGA_nLi, R_squared, slope, intercept, fitParameters, Initial_ImageG_nLi] = RCS2_RegressionGA37(ptsScene1_test, ptsObj1_test, im_new, Num, 'nonlinear');

% Perform Regression with Ransac
[normalizedImg_ransack] = RCS_RegressionRansack(ptsScene1_test, ptsObj1_test, im_new, Num);

% Perform Fusion of Genetic Algorithms
normalizedImg_trainGA_Fusion = fusion_function(normalizedImg_trainGA_Li, normalizedImg_trainGA_nLi, Mask, im_lwir1(:, :, 1:m3));
% Perform Fusion of Trust-Region Reflective Algorithms
normalizedImg_trainTRR_Fusion = fusion_function(normalizedImg_trainTRR_Li, normalizedImg_trainTRR_nLi, Mask, im_lwir1(:, :, 1:m3));
%% Calculate RMSE for each method
[rmseValues1, avgRMSE1] = calculateAverageRMSE(normalizedImg_trainTRR_Li, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues2, avgRMSE2] = calculateAverageRMSE(Initial_ImageT_Li, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues22, avgRMSE22] = calculateAverageRMSE(Initial_ImageT_nLi, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues3, avgRMSE3] = calculateAverageRMSE(normalizedImg_trainTRR_nLi, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues33, avgRMSE33] = calculateAverageRMSE(Initial_ImageT_nLi, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues4, avgRMSE4] = calculateAverageRMSE(Initial_ImageG_Li, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues44, avgRMSE44] = calculateAverageRMSE(normalizedImg_trainGA_Li, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues5, avgRMSE5] = calculateAverageRMSE(Initial_ImageG_nLi, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues55, avgRMSE55] = calculateAverageRMSE(normalizedImg_trainGA_nLi, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues6, avgRMSE6] = calculateAverageRMSE(normalizedImg_ransack, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues7, avgRMSE7] = calculateAverageRMSE(normalizedImg_trainTRR_Fusion, double(im_lwir1(:, :, 1:m3)), Test);
[rmseValues8, avgRMSE8] = calculateAverageRMSE(normalizedImg_trainGA_Fusion, double(im_lwir1(:, :, 1:m3)), Test);

% Create table with names and RMSE values for comparison
MethodNames = {
    'Simple Linear Regression', ...
    'Initial_TRR Linear Regression', ...
    'TRR Linear Regression', ...
    'Initial_TRR Nonlinear Regression', ...
    'TRR Nonlinear Regression', ...
    'Initial_GA Linear Regression', ...
    'GA Linear Regression', ...
    'Initial_GA Nonlinear Regression', ...
    'GA Nonlinear Regression', ...
    'Ransac Regression', ...
    'TRR_Fusion Liner/Nonlinear', ...
    'GA _Fusion Liner/Nonlinear'
};

% Compile RMSE values into an array
RMSE_Values = [
    avgRMSE1;
    avgRMSE2;
    avgRMSE22;
    avgRMSE3;
    avgRMSE33;
    avgRMSE4;
    avgRMSE44;
    avgRMSE5;
    avgRMSE55;
    avgRMSE6;
    avgRMSE7;
    avgRMSE8
];

% Convert method names and RMSE values into a table
rmseComparisonTable = table(MethodNames', RMSE_Values, 'VariableNames', {'Method', 'Avg_RMSE'});

% Create a more detailed RMSE table with values for each method and overall average
rmseComparisonTable2 = [
    rmseValues1, avgRMSE1;
    rmseValues2, avgRMSE2;
    rmseValues22, avgRMSE22;
    rmseValues3, avgRMSE3;
    rmseValues33, avgRMSE33;
    rmseValues4, avgRMSE4;
    rmseValues44, avgRMSE44;
    rmseValues5, avgRMSE5;
    rmseValues55, avgRMSE55;
    rmseValues6, avgRMSE6;
    rmseValues7, avgRMSE7;
    rmseValues8, avgRMSE8
];

% Display the RMSE Comparison Table
disp('RMSE Comparison Table:');
disp(rmseComparisonTable);   % Display the first RMSE comparison table
disp(rmseComparisonTable2);  % Display the second, more detailed RMSE comparison table

% Perform affine transformation for image registration
tform = fitgeotrans(round(ptsObj1), round(ptsScene1), 'affine');
movingRegistered = imwarp(I11, tform, "OutputView", imref2d(size(I22)));

% Calculate the RMSE for the registered image
[rmseValues0, avgRMSE0] = calculateAverageRMSE(movingRegistered, double(im_lwir1(:, :, 1:m3)), Test);

% Compile final results including the affine transformation
Finel_1 = [
    rmseValues0, avgRMSE0;
    rmseValues2, avgRMSE2;
    rmseValues3, avgRMSE3;
    rmseValues22, avgRMSE22;
    rmseValues33, avgRMSE33;
    rmseValues7, avgRMSE7;
    rmseValues44, avgRMSE44;
    rmseValues55, avgRMSE55;
    rmseValues8, avgRMSE8
];