function [normalizedImg_train, R_squared_train, slope_train, intercept_train, fitParameters, Initial_Image] = RCS_RegressionTRR_direct(ptsScene1_test, ptsObj1_test, imgObj_test, imgScene_test, regressionType)
    % Determine the size of the input test image
    [m1, m2, m3] = size(imgObj_test);

    % Extract non-zero values from the point matrices
    for k=1:m3
        for i=1:max(size(ptsScene1_test))
            R_DN(i,k)=imgScene_test(ptsScene1_test(i,2),ptsScene1_test(i,1),k);
        end
    end


    for k=1:m3
        for i=1:max(size(ptsObj1_test))
            S_DN(i,k)=imgObj_test(ptsObj1_test(i,2),ptsObj1_test(i,1),k);
        end
    end
    % Initialize output matrices and variables
    normalizedImg_train = zeros(size(imgObj_test));
    R_squared_train = zeros(1, m3);
    slope_train = zeros(1, m3);
    intercept_train = zeros(1, m3);
    fitParameters = cell(1, m3);

    % Trust-Region Reflective options
    trOptions = optimoptions('lsqcurvefit', 'Display', 'iter', 'Algorithm', 'trust-region-reflective', 'MaxIterations', 1000);

    % Process each channel of the image
    for i = 1:m3
        % Ensure there are no zero-only channels
        if ~all(R_DN(:, i) == 0) && ~all(S_DN(:, i) == 0)
            if strcmpi(regressionType, 'linear')
                % Initial guess from simple linear regression
                model = polyfit(S_DN(:, i), R_DN(:, i), 1);
                initialGuess = [model(2), model(1)];
                Initial_Image = model(2) + model(1) .* imgObj_test;
                % Set bounds
                lb = [model(2)-0.05, model(1)-0.05];
                ub = [model(2)+0.05, model(1)+0.05];
                % Fit the model using TRR
                [parameters, ~, ~, ~, ~] = lsqcurvefit(@(params, xdata) params(1) + params(2) * xdata, initialGuess, S_DN1(:, i), R_DN1(:, i), lb, ub, trOptions);
            elseif strcmpi(regressionType, 'nonlinear')
                % Initial guess from simple quadratic regression
                model = polyfit(S_DN(:, i), R_DN(:, i), 2);
                initialGuess = [model(3), model(2), model(1)];
                Initial_Image = model(3) + model(2) .* imgObj_test + model(1) .* imgObj_test.^2;

                % Set bounds
                lb = [model(3)-0.05, model(2)-0.05, model(1)-0.05];
                ub = [model(3)+0.05, model(2)+0.05, model(1)+0.05];
                % Fit the model using TRR
                [parameters, ~, ~, ~, ~] = lsqcurvefit(@(params, xdata) params(1) + params(2) * xdata + params(3) * xdata.^2, initialGuess, S_DN1(:, i), R_DN1(:, i), lb, ub, trOptions);
            else
                error('Unsupported regression type. Use "linear" or "nonlinear".');
            end

            % Store parameters and compute normalization
            fitParameters{i} = parameters;
            if strcmpi(regressionType, 'linear')
                slope_train(i) = parameters(2);
                intercept_train(i) = parameters(1);
                normalizedImg_train(:,:,i) = intercept_train(i) + slope_train(i) * imgObj_test(:,:,i);
            elseif strcmpi(regressionType, 'nonlinear')
                fitImg = @(img) parameters(1) + parameters(2) * img + parameters(3) * img.^2;
                normalizedImg_train(:,:,i) = fitImg(imgObj_test(:,:,i));
            end

            % Calculate R-squared
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
            normalizedImg_train(:,:,i) = zeros(size(imgObj_test(:,:,i)));
        end
    end

end

function objVal = objectiveFunction(params, xdata, ydata)
    % Main objective function
    objVal = sqrt(sum((ydata - (params(1) + params(2) .* xdata)).^2));
end