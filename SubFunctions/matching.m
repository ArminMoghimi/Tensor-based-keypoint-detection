function [ptsObj1, ptsScene1] = matching(pos1, pos2, desc1, desc2)
    % Find matches using vl_ubcmatch
    [matches, ~] = vl_ubcmatch(single(desc1), single(desc2), 1.8);
    
    % Extract corresponding coordinates
    x1 = pos1(1, matches(1, :));
    y1 = pos1(2, matches(1, :));
    x2 = pos2(1, matches(2, :));
    y2 = pos2(2, matches(2, :));
    
    % Store matched points in Qt1 and Qt2
    Qt1 = [x1', y1'];
    Qt2 = [x2', y2'];
    
    % Call Homography function for further processing
    [ptsObj1, ptsScene1] = Homography(x1, y1, x2, y2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ransac for outlier removal
end

function [ptsObj1, ptsScene1] = Homography(x1, y1, x2, y2)
    % Combine x and y coordinates for input points
    ptsObj1 = [x1; y1];
    ptsScene1 = [x2; y2];
    
    a = ptsObj1';
    b = ptsScene1';
    
    if size(a, 1) >= 4
        % Compute homography using RANSAC
        [H, inliers] = cv.findHomography(a, b, 'Method', 'Ransac');
        
        % Check if homography matrix is empty
        if isempty(H)
            ptsScene1 = b;
            ptsObj1 = a;   
        else
            inliers = logical(inliers);
            
            % Extract inliers from input points
            ptsObj1 = (ptsObj1(:, inliers))';
            ptsScene1 = (ptsScene1(:, inliers))';
        end
    else
        % If not enough points, keep original points
        ptsScene1 = b;
        ptsObj1 = a;
    end
end
