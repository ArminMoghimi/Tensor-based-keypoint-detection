function [im_new_transformed, R_squared, Slope, Intercept, ptsObj2, ptsScene2, t_score] = RCS_jointRegression1(ptsScene1, ptsObj1, imgObj, imgScene)
    [m1, m2, m3] = size(imgObj);

    % Initialize arrays
    R_DN = zeros(size(ptsScene1, 1), m3);
    S_DN = zeros(size(ptsObj1, 1), m3);

    % Extract pixel values from the object and scene images
    for k = 1:m3
        for i = 1:max(size(ptsScene1))
            R_DN(i, k) = imgScene(round(ptsScene1(i, 2)), round(ptsScene1(i, 1)), k);
        end
        for i = 1:max(size(ptsObj1))
            S_DN(i, k) = imgObj(round(ptsObj1(i, 2)), round(ptsObj1(i, 1)), k);
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

    % Initialize affine parameters using geometric transformation between points
    tform_initial = fitgeotrans(ptsObj1, ptsScene1, 'affine');
    
    % Extract affine parameters (a, b, c, d, tx, ty) from transformation matrix
    affine_matrix = tform_initial.T;
    a = affine_matrix(1, 1);
    b = affine_matrix(1, 2);
    c = affine_matrix(2, 1);
    d = affine_matrix(2, 2);
    tx = affine_matrix(3, 1);
    ty = affine_matrix(3, 2);
    affine_params_init = [a, b, c, d, tx, ty];

    disp('Initial Affine Parameters (from geometric transformation):');
    disp(affine_params_init);

    % Loop through each channel (m3)
    for i = 1:m3
        % Perform linear regression to initialize radiometric parameters
        mdl = fitlm(nonzeros(S_DN(:, i)), nonzeros(R_DN(:, i)), 'RobustOpts', 'off');
        initial_slope = mdl.Coefficients.Estimate(2); % Slope (coefficient)
        initial_intercept = mdl.Coefficients.Estimate(1); % Intercept
        radiometric_params_init = [initial_slope, initial_intercept];

        disp(['Initial Radiometric Parameters for Channel ', num2str(i), ':']);
        disp(['Slope: ', num2str(initial_slope), ', Intercept: ', num2str(initial_intercept)]);

        % Initial parameters: radiometric (slope, intercept) + affine
        initial_params = [radiometric_params_init, affine_params_init];

        % Define lower and upper bounds (-0.05 and +0.5 around initial values)
        lb_radiometric = (radiometric_params_init - 0.05); % Lower bound for radiometric [-5%]
        ub_radiometric = (radiometric_params_init+ 0.05);  % Upper bound for radiometric [+50%]
        
        % Apply the same logic for affine parameters
        lb_affine = (affine_params_init - 100); % Lower bound for affine [-5%]
        ub_affine = (affine_params_init + 100);  % Upper bound for affine [+50%]

        % Combine the bounds
        lb = [lb_radiometric, lb_affine];
        ub = [ub_radiometric, ub_affine];

        disp('Bounds for optimization:');
        disp('Lower bounds:');
        disp(lb);
        disp('Upper bounds:');
        disp(ub);

        % Objective function for joint optimization
        objective_function = @(params) joint_residuals(params, S_DN(:, i), R_DN(:, i), ptsObj1, ptsScene1, lambda_value);

        % Optimization options for lsqnonlin
        options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
                               'Display', 'iter', 'MaxIterations', 500, 'FunctionTolerance', 1e-20);

        % Perform joint optimization using lsqnonlin
        optimized_params = lsqnonlin(objective_function, initial_params, lb, ub, options);

        % Extract optimized radiometric and affine parameters
        radiometric_params = optimized_params(1:2); % Extract the first two values
        affine_params = optimized_params(3:end);    % Extract the affine parameters

        % Store regression results for R-squared, slope, and intercept
        R_squared(1, i) = 1 - var(R_DN(:, i) - radiometric_params(1) * S_DN(:, i) - radiometric_params(2)) / var(R_DN(:, i));
        Slope(1, i) = radiometric_params(1);   % Store the optimized slope
        Intercept(1, i) = radiometric_params(2); % Store the optimized intercept

        % Apply the linear model to normalize the subject image
        im_new(:, :, i) = radiometric_params(1) * double(imgObj(:, :, i)) + radiometric_params(2);
    end

    % Set dummy values for ptsObj2, ptsScene2, and t_score (they are not calculated in this function)
    ptsObj2 = [];
    ptsScene2 = [];
    t_score = [];

    % Perform affine transformation with optimized affine parameters
    tform = affine2d([affine_params(1) affine_params(3) 0; ...
                      affine_params(2) affine_params(4) 0; ...
                      affine_params(5) affine_params(6) 1]);

    % Apply transformation to the normalized image
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
