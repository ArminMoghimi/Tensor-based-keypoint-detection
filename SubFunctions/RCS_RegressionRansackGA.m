function [bestParams] = RCS_RegressionRansackGA(ptsScene1_test, ptsObj1_test, imgObj_test, Num)
    [m1, m2, m3] = size(imgObj_test);

[m1, m2, m3] = size(imgObj_test);

    for ii = 1:size(ptsScene1_test, 2)
        R_DN(:, ii) = nonzeros(ptsScene1_test(:, ii));
        S_DN(:, ii) = nonzeros(ptsObj1_test(:, ii));
    end

    selectedIndices = randperm(size(R_DN, 1), Num);
    R_DN = R_DN(selectedIndices', 1:m3);
    S_DN = S_DN(selectedIndices', 1:m3);


    % Linear Regression based on the RCS   
    sampleSize = 5000; % number of points to sample per trial
    maxDistance = 50; % max allowable distance for inliers

    for i = 1:m3
        pts =abs( [S_DN(:,i), R_DN(:,i)]);

        % Define fitting and evaluation functions
        fitFcn = @(points) polyfit(points(:, 1), points(:, 2), 1);
        evalFcn = @(model, points) sum((points(:, 2) - polyval(model, points(:, 1))).^2, 2);

        % Specify parameter ranges, population size, and generations
        paramRanges.sampleSize = 100:500;
        paramRanges.maxDistance = 10:50;

        populationSize = 10;
        generations = 5;

        % Call the optimization function
        [bestParams, bestInlierIdx, bestModel] = optimizeRANSACParametersGA(pts, fitFcn, evalFcn, paramRanges, populationSize, generations);

        % Display the results or use them as needed
        disp(['Best Parameters for Channel ', num2str(i), ':']);
        disp(bestParams);
    end
end
