function [normalizedImg_train, R_squared_train, slope_train, intercept_train, fitParameters] = RCS1_RegressionTRR(ptsScene1_test, ptsObj1_test, imgObj_test, imgScene_test, Num)

%  The code written by Armin Moghimi (2022)

% Determine the size of the input test image
[m1, m2, m3] = size(imgObj_test);

% Initialize necessary matrices to store non-zero values
R_DN = [];
S_DN = [];

% Extract non-zero values from the point matrices
for ii = 1:size(ptsScene1_test, 2)
    R_DN = [R_DN, nonzeros(ptsScene1_test(:, ii))];
    S_DN = [S_DN, nonzeros(ptsObj1_test(:, ii))];
end

% Randomly select indices for subset
selectedIndices = randperm(size(R_DN, 1), Num);
R_DN1 = R_DN(selectedIndices, :);
S_DN1 = S_DN(selectedIndices, :);

% Initialize output matrices and variables
normalizedImg_train = zeros(size(imgObj_test));
R_squared_train = zeros(1, m3);
slope_train = zeros(1, m3);
intercept_train = zeros(1, m3);
fitParameters = cell(1, m3);

% Trust-Region Reflective options
trOptions = optimoptions('lsqcurvefit', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');

% Process each channel of the image
for i = 1:m3
    % Ensure there are no zero-only channels
    if ~all(R_DN(:, i) == 0) && ~all(S_DN(:, i) == 0)
        % Initial guess for intercept and slope
        initialGuess = [0, 1];

        % Define linear model bounds based on initial linear regression
        model = polyfit(S_DN(:, i), R_DN(:, i), 1);
        lb = [model(2)-0.005, model(1)-0.05];
        ub = [model(2)+0.005, model(1)+0.05];

        % Fit the model using Trust-Region Reflective algorithm
        [fitParameters{i}, ~, ~, ~, ~] = lsqcurvefit(@(params, xdata) params(1) + params(2) * xdata, initialGuess, S_DN1(:, i), R_DN1(:, i), lb, ub, trOptions);

        % Compute R-squared
        residuals = nonzeros(R_DN(:, i)) - (fitParameters{i}(1) + fitParameters{i}(2) * nonzeros(S_DN(:, i)));
        R_squared_train(i) = 1 - sum(residuals.^2) / sum((nonzeros(R_DN(:, i)) - mean(nonzeros(R_DN(:, i)))).^2);
        slope_train(i) = fitParameters{i}(2);
        intercept_train(i) = fitParameters{i}(1);
        normalizedImg_train(:,:,i) = intercept_train(i) + slope_train(i) * imgObj_test(:,:,i);
    else
        % Set NaN for zero-only channels
        R_squared_train(i) = NaN;
        slope_train(i) = NaN;
        intercept_train(i) = NaN;
        normalizedImg_train(:,:,i) = zeros(size(imgObj_test(:,:,i)));
    end
end

end