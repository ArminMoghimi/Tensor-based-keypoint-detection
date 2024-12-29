function [normalizedImg_train] = RCS_Regression(ptsScene1_test, ptsObj1_test, imgObj_test, Num)
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

        % Linear Regression
        model = polyfit(points(:, 1), points(:, 2), 1);

        % Apply the model to the entire image
        normalizedImg_train(:,:,i) = model(1) .* imgObj_test(:,:,i) + model(2);
    end
end