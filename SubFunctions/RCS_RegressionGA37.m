function [normalizedImg_train, R_squared_train, slope_train, intercept_train, fitParameters] = RCS_RegressionGA37(ptsScene1_test, ptsObj1_test, imgObj_test,Num)
%  The code wriiten by Armin Moghimi (2022)

[m1, m2, m3] = size(imgObj_test);    
for ii = 1:size(ptsScene1_test, 2)
        R_DN(:, ii) = nonzeros(ptsScene1_test(:, ii));
        S_DN(:, ii) = nonzeros(ptsObj1_test(:, ii));
end
selectedIndices = randperm(size(R_DN, 1), Num);
R_DN1 = (R_DN(selectedIndices',1:m3));
S_DN1 = (S_DN(selectedIndices',1:m3));

% Linear Regression based on the RCS
normalizedImg_train = zeros(size(imgObj_test));
R_squared_train = zeros(1, m3);
slope_train = zeros(1, m3);
intercept_train = zeros(1, m3);
fitParameters = cell(1, m3);

% Genetic Algorithm options
gaOptions = optimoptions(@ga, 'Display', 'iter', 'PlotFcn', @gaplotbestf, 'ConstraintTolerance', 1e-6, 'MaxStallGenerations', 20);

% Define optimizers for each band
optimizers = {'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga', 'ga'}; % Use GA for all bands
    
    for i = 1:m3
        % Check for zero-division
        if ~all(R_DN(:, i) == 0) && ~all(S_DN(:, i) == 0)
            % Use genetic algorithm for optimization
            optimizer = optimizers{i};
            initialGuess = [0, 1]; % Initial guess for intercept and slope

            % Define the objective function with additional penalty terms
            objectiveFcn = @(params) objectiveFunction(params,S_DN1(:, i),(R_DN1(:, i)));

            % Set bounds for parameters
            lb=[-5, -20];
            ub=[5, 20];

            % Fit the model using the Genetic Algorithm
            [fitParameters{i}, ~, ~, ~, ~] = ga(objectiveFcn, 2, [], [], [], [], lb, ub, [], gaOptions);

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

function objVal = objectiveFunction(params, xdata, ydata)
    % Main objective function
    objVal = sqrt(sum((ydata - (params(1) + params(2) .* xdata)).^2));
end
