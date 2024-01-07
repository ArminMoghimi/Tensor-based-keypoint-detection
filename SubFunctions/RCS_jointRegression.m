function [im_new_transformed, R_squared, Slope, Intercept, ptsObj2, ptsScene2, t_score] = RCS_jointRegression(ptsScene1, ptsObj1, imgObj, imgScene)
    [m1, m2, m3] = size(imgObj);

    % Initialize arrays
    R_DN = zeros(size(ptsScene1, 1), m3);
    S_DN = zeros(size(ptsObj1, 1), m3);

    for k = 1:m3
        for i = 1:max(size(ptsScene1))
            R_DN(i, k) = imgScene(ptsScene1(i, 2), ptsScene1(i, 1), k);
        end
    end

    for k = 1:m3
        for i = 1:max(size(ptsObj1))
            S_DN(i, k) = imgObj(ptsObj1(i, 2), ptsObj1(i, 1), k);
        end
    end

    % Reshape imgObj
    imgObj = reshape(imgObj, m1, m2, m3);

    % Linear Regression based on RCS with joint optimization
    im_new = zeros(size(imgObj));
    R_squared = zeros(1, m3);
    Slope = zeros(1, m3);
    Intercept = zeros(1, m3);
    lambda_value = 0.05; % Adjust this value based on your preference

    % Initialize affine parameters
    affine_params_init = zeros(1, 6);

    for i = 1:m3
        % Initialize radiometric parameters from linear regression
        %mdl = fitlm((nonzeros(S_DN(:, i))), (nonzeros(R_DN(:, i))), 'interactions', 'RobustOpts', 'off');
         %radiometric_params_init = [mdl.Coefficients.Estimate(2), mdl.Coefficients.Estimate(1)];
        lb_radiometric = [-Inf,-Inf ]; % Lower bounds for radiometric parameters
        ub_radiometric = [Inf,Inf]; % Upper bounds for radiometric parameters
        
        % Define lower and upper bounds for the radiometric parameters in each iteration
        lb = [lb_radiometric, -Inf(1, 6)]; % Lower bounds for all parameters
        ub = [ub_radiometric, Inf(1, 6)]; % Upper bounds for all parameters

        % Initial guess for joint optimization
        initial_params = [1,0, affine_params_init];

        objective_function = @(params) joint_residuals(params, S_DN(:, i), R_DN(:, i), ptsObj1, ptsScene1, lambda_value);

   % Perform joint optimization using Trust-Region Reflective algorithm
    options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'iter', 'MaxIterations', 500, 'FunctionTolerance', 1e-20);

        
        optimized_params = lsqnonlin(objective_function, initial_params, lb, ub, options);

        % Extract optimized parameters
        radiometric_params = optimized_params(1:2);
        affine_params = optimized_params(3:end);

        % Store regression results
        R_squared(1, i) = 1 - var(R_DN(:, i) - radiometric_params(1) * S_DN(:, i) - radiometric_params(2)) / var(R_DN(:, i));
        Slope(1, i) = affine_params(5);
        Intercept(1, i) = affine_params(6);

        % Apply the linear model to normalize subject image
        im_new(:, :, i) = radiometric_params(1) * double(imgObj(:, :, i)) + radiometric_params(2);
    end

    % Set dummy values for ptsObj2, ptsScene2, and t_score (they are not calculated in this function)
    ptsObj2 = [];
    ptsScene2 = [];
    t_score = [];

    % Transform the normalized subject image using the affine transformation
    tform = affine2d([affine_params(1) affine_params(3) 0; ...
                      affine_params(2) affine_params(4) 0; ...
                      affine_params(5) affine_params(6) 1]);

    im_new_transformed = imwarp(im_new, tform, 'OutputView', imref2d(size(imgScene)));

    % Display or save the results
figure;

subplot(1, 4, 1);
imshow(uint8(imgObj(:, :, 1:3)));
title('Original Subject Image');

subplot(1, 4, 2);
imshow(uint8(im_new(:, :, 1:3)));
title('Normalized Subject Image');

subplot(1, 4, 3);
imshow(uint8(im_new_transformed(:, :, 1:3)));
title('Transformed Image');

subplot(1, 4, 4);
imshow(uint8(imgScene(:, :, 1:3)));
title('Reference Image');
end

% Define residuals and joint_residuals functions (as provided in previous responses)
function radiometric_residuals = residuals_radiometric(params, KS, KR, lambda)
    radiometric_params = params(1:2);
    KS_normalized = radiometric_params(1) * KS + radiometric_params(2);
    
    % Calculate residuals with a weighted combination
    linear_residuals = KR - KS_normalized;
    
    % Combine radiometric and linear regression residuals
    radiometric_residuals = (1 - lambda) * linear_residuals;
end

function affine_residuals = residuals_affine(params, xS, yS, xR, yR)
    a = params(1);
    b = params(2);
    c = params(3);
    d = params(4);
    tx = params(5);
    ty = params(6);

    xS_transformed = a * xS + b * yS + tx;
    yS_transformed = c * xS + d * yS + ty;

    affine_residuals = [xS_transformed - xR; yS_transformed - yR];
end

function residuals = joint_residuals(params, S_DN, R_DN, ptsObj1, ptsScene1, lambda_value)
    radiometric_residuals = residuals_radiometric(params, S_DN, R_DN, lambda_value);
    affine_residuals = residuals_affine(params(3:end), ptsObj1(:, 1), ptsObj1(:, 2), ptsScene1(:, 1), ptsScene1(:, 2));
    residuals = [radiometric_residuals; sqrt(lambda_value) * affine_residuals];
end
