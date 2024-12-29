function [normalizedImg_train] = RCS_RegressionRansack(ptsScene1_test, ptsObj1_test, imgObj_test, Num)
    [m1, m2, m3] = size(imgObj_test);

    for ii = 1:size(ptsScene1_test, 2)
        R_DN(:, ii) = nonzeros(ptsScene1_test(:, ii));
        S_DN(:, ii) = nonzeros(ptsObj1_test(:, ii));
    end

    selectedIndices = randperm(size(R_DN, 1), Num);
    R_DN1 = R_DN(selectedIndices', 1:m3);
    S_DN1 = S_DN(selectedIndices', 1:m3);

    % Linear Regression based on the RCS   
    for i = 1:m3
        points(:, 1) = S_DN1(:, i);
        points(:, 2) = R_DN1(:, i);

        % Parameter search space
        sampleSizeValues = [1000,2000,3000,4000,5000];
        maxDistanceValues =50000;

        bestInlierCount = 0;
        bestModelInliers = [];
        bestNormalizedImg_train = [];

        for sampleSize = sampleSizeValues
            for maxDistance = maxDistanceValues
                fitLineFcn = @(points) polyfit(points(:,1), points(:,2), 1);
                evalLineFcn = @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2, 2);

                [modelRANSAC, inlierIdx] = ransac(points, fitLineFcn, evalLineFcn, sampleSize, maxDistance);

                % Count inliers
                inlierCount = sum(inlierIdx);
                if inlierCount > bestInlierCount
                    bestInlierCount = inlierCount;
                    bestModelInliers = polyfit(points(inlierIdx, 1), points(inlierIdx, 2), 1);
                    bestNormalizedImg_train = bestModelInliers(1) .* imgObj_test(:,:,i) + bestModelInliers(2);
                end
            end
        end

        normalizedImg_train(:,:,i) = bestNormalizedImg_train;
    end
end
