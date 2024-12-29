function [normalizedImg_train, R_squared_train, slope_train, intercept_train, fitParameters, Initial_Image] = RCS_RegressionPSO(ptsScene1_test, ptsObj1_test, imgObj_test, Num, regressionType)
    % The code written by Armin Moghimi (2022)

    % Determine the size of the input test image
    [m1, m2, m3] = size(imgObj_test);

    % Initialize matrices to store non-zero values
    R_DN = [];
    S_DN = [];
    for ii = 1:size(ptsScene1_test, 2)
        R_DN = [R_DN, nonzeros(ptsScene1_test(:, ii))];
        S_DN = [S_DN, nonzeros(ptsObj1_test(:, ii))];
    end

    % Randomly select indices for the subset
    selectedIndices = randperm(size(R_DN, 1), Num);
    R_DN1 = R_DN(selectedIndices, :);
    S_DN1 = S_DN(selectedIndices, :);

    % Initialize output matrices and variables
    normalizedImg_train = zeros(size(imgObj_test));
    R_squared_train = zeros(1, m3);
    slope_train = zeros(1, m3);
    intercept_train = zeros(1, m3);
    fitParameters = cell(1, m3);

    % PSO options
    options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 50, 'MaxIterations', 100);

    % Process each channel of the image
    for i = 1:m3
        % Ensure there are no zero-only channels
        if ~all(R_DN(:, i) == 0) && ~all(S_DN(:, i) == 0)
            % Initial guess for the parameters
            if strcmpi(regressionType, 'linear')
                % Linear Regression initial guess and bounds
                model = polyfit(S_DN(:, i), R_DN(:, i), 1);
                initialGuess = [model(2), model(1)];
                Initial_Image = model(1) .* imgObj_test + model(2);
                lb = [model(2)-0.05, model(1)-0.05];
                ub = [model(2)+0.05, model(1)+0.05];

                % Define the objective function
                objectiveFcn = @(params) sqrt(sum((R_DN1(:, i) - (params(1) + params(2) .* S_DN1(:, i))).^2));

            elseif strcmpi(regressionType, 'nonlinear')
                % Non-linear Regression initial guess and bounds (quadratic model)
                model = polyfit(S_DN(:, i), R_DN(:, i), 2);
                initialGuess = [model(3), model(2), model(1)];
                Initial_Image = model(3) + model(2) .* imgObj_test + model(1) .* imgObj_test.^2;
                lb = [model(3)-0.05, model(2)-0.05, model(1)-0.05];
                ub = [model(3)+0.05, model(2)+0.05, model(1)+0.05];

                % Define the objective function
                objectiveFcn = @(params) sqrt(sum((R_DN1(:, i) - (params(1) + params(2) .* S_DN1(:, i) + params(3) .* S_DN1(:, i).^2)).^2));

            else
                error('Unsupported regression type. Use "linear" or "nonlinear".');
            end

            % Fit the model using Particle Swarm Optimization
            [fitParameters{i}, ~] = particleswarm(objectiveFcn, length(initialGuess), lb, ub, options);

            % Compute normalization based on the regression type
            if strcmpi(regressionType, 'linear')
                slope_train(i) = fitParameters{i}(2);
                intercept_train(i) = fitParameters{i}(1);
                normalizedImg_train(:, :, i) = intercept_train(i) + slope_train(i) * imgObj_test(:, :, i);
            elseif strcmpi(regressionType, 'nonlinear')
                slope_train(i) = NaN;  % Not used in non-linear regression
                intercept_train(i) = NaN;  % Not used in non-linear regression
                fitImg = @(img) fitParameters{i}(1) + fitParameters{i}(2) * img + fitParameters{i}(3) * img.^2;
                normalizedImg_train(:, :, i) = fitImg(imgObj_test(:, :, i));
            end

            % Compute R-squared
            residuals = nonzeros(R_DN(:, i)) - (fitParameters{i}(1) + fitParameters{i}(2) * nonzeros(S_DN(:, i)));
            if strcmpi(regressionType, 'nonlinear')
                residuals = residuals - fitParameters{i}(3) * nonzeros(S_DN(:, i)).^2;
            end
            R_squared_train(i) = 1 - sum(residuals.^2) / sum((nonzeros(R_DN(:, i)) - mean(nonzeros(R_DN(:, i)))).^2);

        else
            % Handle zero-only channels
            R_squared_train(i) = NaN;
            slope_train(i) = NaN;
            intercept_train(i) = NaN;
            fitParameters{i} = NaN;
            normalizedImg_train(:, :, i) = zeros(size(imgObj_test(:, :, i)));
        end
    end
end

function objVal = objectiveFunction(params, xdata, ydata)
    % Main objective function
    objVal = sqrt(sum((ydata - (params(1) + params(2) .* xdata)).^2));
end
