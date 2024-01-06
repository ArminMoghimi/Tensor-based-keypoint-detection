function [normalizedImg_train, R_squared_train, slope_train, intercept_train, fitParameters] = RCS_RegressionTRR(ptsScene1_test, ptsObj1_test, imgObj_test,Num)
%  The code wriiten by Armin Moghimi (2022)

[m1, m2, m3] = size(imgObj_test);

    for ii = 1:size(ptsScene1_test, 2)
        R_DN(:, ii) = nonzeros(ptsScene1_test(:, ii));
        S_DN(:, ii) = nonzeros(ptsObj1_test(:, ii));
    end

    selectedIndices = randperm(size(R_DN, 1), Num);
    R_DN1 = R_DN(selectedIndices', 1:m3);
    S_DN1 = S_DN(selectedIndices', 1:m3);

    % Linear Regression based on the RCS
    normalizedImg_train = zeros(size(imgObj_test));
    R_squared_train = zeros(1, m3);
    slope_train = zeros(1, m3);
    intercept_train = zeros(1, m3);
    fitParameters = cell(1, m3);

    % Trust-Region Reflective options
    trOptions = optimoptions('lsqcurvefit', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');

    for i = 1:m3
        % Check for zero-division
        if ~all(R_DN(:, i) == 0) && ~all(S_DN(:, i) == 0)
            % Use Trust-Region Reflective algorithm for optimization
            initialGuess = [0, 1]; % Initial guess for intercept and slope

            % Define the objective function with additional penalty terms
            objectiveFcn = @(params, xdata) objectiveFunction(params, xdata, S_DN1(:, i), R_DN1(:, i));

            % Set bounds for parameters
            lb = [-5, -20];
            ub = [5, 20];

            % Fit the model using Trust-Region Reflective algorithm
            [fitParameters{i}, ~, ~, ~, ~] = lsqcurvefit(@(params, xdata) params(1) + params(2) * xdata, initialGuess, S_DN1(:, i), R_DN1(:, i), lb, ub, trOptions);

            % Compute R-squared using a different formula
            R_squared_train(i) = 1 - sum((nonzeros(R_DN(:, i)) - (fitParameters{i}(1) + fitParameters{i}(2) * nonzeros(S_DN(:, i)))).^2) / sum((nonzeros(R_DN(:, i)) - mean(nonzeros(R_DN(:, i)))).^2);
            slope_train(i) = fitParameters{i}(2);
            intercept_train(i) = fitParameters{i}(1);
            normalizedImg_train(:,:,i) = intercept_train(i) + slope_train(i) * (imgObj_test(:,:,i));
        else
            % Handle the case where all values are zero
            R_squared_train(i) = NaN;
            slope_train(i) = NaN;
            intercept_train(i) = NaN;
            normalizedImg_train(:,:,i) = zeros(size(imgObj_test(:,:,i)));
        end
    end
end

function objVal = objectiveFunction(params, xdata, ydata, expectedGain, expectedOffset)
    % Additional penalty terms to enforce proximity to expectedGain and expectedOffset
    % Main objective function
    objVal = sqrt(sum((expectedOffset - (params(1) + params(2) .* expectedGain)).^2));
end
