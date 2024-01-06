clc
clear all
close all

% ------------------------------------ Set path
fp = mfilename('fullpath');
rootdir = fileparts(fp);
p{1} = fullfile(rootdir, 'vlfeat');
p{2} = fullfile(rootdir, 'SubFunctions');
p{3} = fullfile(rootdir, 'Dataset');
for i = 1:size(p, 2)
    addpath(genpath(p{i}));
end

%% Read multispectral Image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[name, path] = uigetfile('*.tif', 'Select a Sub. image', 'Subject');
I11 = (imread([path, name])) + 0.025;
[name, path] = uigetfile('*.tif', 'Select a Ref. image', 'reference');
I22 = (imread([path, name])) + 0.025;
[filename, pathname] = uigetfile('*.mat', 'Select a MAT File', 'Test');
load([pathname,filename]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Image Normalization in [0 ,1]
I1 = im2double(I11);
I2 = im2double(I22);
disp('Image was loaded successfully')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% WSST_SURF parameter setting
Options.upright = false;
Options.tresh = 0.000001;
Options.octaves = 5;

disp('WSST_SURF parameters were set')

%% Keypoint detection and description using WSST_SURF
%% Keypoint detection and discription using Wighted-Tensor spectral SURF
disp('Starting for Keypoint detection and discription for Image pair')
Ipts1=OpenSurf(I1,Options);
disp('keypoints were detected for Sub. image successfully')
Ipts2=OpenSurf(I2,Options);
disp('keypoints were detected for Ref. image successfully')
%% Put the landmark descriptors in a matrix
D1 = reshape([Ipts1.descriptor],64,[]);
D2 = reshape([Ipts2.descriptor],64,[]);
A=cell2mat(struct2cell(Ipts1'));
A=A(1:2,:);
B=cell2mat(struct2cell(Ipts2'));
B=B(1:2,:);
%% Matching descriptor
disp('Matching procedure.....')
[ptsObj1, ptsScene1] = matching(A, B, D1, D2);
disp('Matching was successfully done.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('RNN procedure......')
appendimages(I1(:,:,1), I2(:,:,1), ptsObj1, ptsScene1);
[Normalized_Image, ~, ~, ~, ~, ~] = RCS_Regression(round(ptsScene1), round(ptsObj1), (double(I11) + 0.0025), (double(I22) + 0.0025));
[im_new, R_2, B, M, ptsObj2, ptsScene2, t_score] = RCS_jointRegression(round(ptsScene1), round(ptsObj1), (double(I11) + 0.0025), (double(I22) + 0.0025));
disp('RNN was successfully done.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure, imshow(uint8(I11(:,:,1:3)))
title('Subject Image')
figure, imshow(uint8(I22(:,:,1:3)))
title('Reference Image')
figure, imshow(uint8(Normalized_Image(:,:,1:3)))
title('Normalized Subject Image')

%% Optimization process; find Mask for No data in reference image
Mask1 = I22(:,:,1)+1 > 0;
Mask = imcomplement(im_new(:,:,1) == 0);
Mask = Mask1 .* Mask;
im_lwir1 = Mask .* (double(I22) + 0.025);
[m1, m2, m3] = size(im_lwir1);

%% CVA image generation 
CVA = (sqrt(sum((double(im_new) - double(im_lwir1)).^2, 3)) + 1) .* Mask;
CVA = rescale(CVA, 0, 1);
thresh = multithresh(nonzeros(CVA), 2);
Mask2 = (CVA < thresh(2)) .* Mask;
ptsScene1_test = Mask2 .* im_lwir1;
ptsScene1_test = reshape(ptsScene1_test, m1 * m2, m3);
ptsObj1_test = Mask2 .* im_new;
ptsObj1_test = reshape(ptsObj1_test, m1 * m2, m3);

Num = 10000;

[normalizedImg_train, R_squared, slope, intercept, fitParameters] = RCS_RegressionTRR(ptsScene1_test, ptsObj1_test, im_new, Num);
[normalizedImg_train1, R_squared, slope, intercept, fitParameters] = RCS_RegressionGA37(ptsScene1_test, ptsObj1_test, im_new, Num);

%%%%%%%%%%%%% RMSE evaluation
[rmseValues2, avgRMSE2] = calculateAverageRMSE(double(im_new), double(im_lwir1(:,:,1:m3)), Test)
[rmseValues3, avgRMSE3] = calculateAverageRMSE(double(normalizedImg_train), double(im_lwir1(:,:,1:m3)), Test)
[rmseValues4, avgRMSE4] = calculateAverageRMSE(normalizedImg_train1, double(im_lwir1(:,:,1:m3)), Test)
