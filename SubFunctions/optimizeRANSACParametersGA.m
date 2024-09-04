function [bestParams, bestInlierIdx, bestModel] = optimizeRANSACParametersGA(pts, fitFcn, evalFcn, paramRanges, populationSize, generations)
    % Define the fitness function
    fitnessFcn = @(params) -evaluateFitness(params, pts, fitFcn, evalFcn);

    % Define the options for the genetic algorithm
    options = optimoptions('ga', 'PopulationSize', populationSize, 'MaxGenerations', generations);

    % Define the parameter ranges for GA
    lb = [paramRanges.sampleSize(1), paramRanges.maxDistance(1)];
    ub = [paramRanges.sampleSize(end), paramRanges.maxDistance(end)];

    % Run the genetic algorithm
    bestParams = ga(fitnessFcn, 2, [], [], [], [], lb, ub, [], options);

    % Run RANSAC with the best parameters
    [bestModel, bestInlierIdx] = ransac(pts, fitFcn, evalFcn, bestParams(1), bestParams(2));
end

function fitness = evaluateFitness(params, pts, fitFcn, evalFcn)
    sampleSize = params(1);
    maxDistance = params(2);

    % Perform RANSAC
    [~, inlierIdx] = ransac(pts, fitFcn, evalFcn, sampleSize, maxDistance);

    % Evaluate fitness (you may use a specific metric)
    % For example, you might use the number of inliers or a robustness measure.
    fitness = -numel(inlierIdx);
end
