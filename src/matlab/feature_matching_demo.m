function result = feature_matching_demo(imagePathA, imagePathB)
%FEATURE_MATCHING_DEMO ORB keypoint detection, descriptor extraction, and correspondence matching.
%
% THEORETICAL BACKGROUND:
%   ORB (Oriented FAST and Rotated BRIEF) is a rotation-invariant, scale-aware,
%   computationally efficient feature detector and binary descriptor.
%   This function:
%     1. Detects multiscale ORB keypoints across image pyramid levels.
%     2. Extracts 256-bit binary descriptors steered by local intensity centroid angles.
%     3. Establishes nearest-neighbor correspondences using Hamming distance with Lowe's ratio test.
%
% SYNTAX:
%   result = feature_matching_demo()
%   result = feature_matching_demo(imagePathA, imagePathB)
%
% INPUTS:
%   imagePathA - (Optional) Path to query image. Default: 'cameraman.tif'.
%   imagePathB - (Optional) Path to target image. Default: rotated 'cameraman.tif'.
%
% OUTPUTS:
%   result     - MATLAB struct containing:
%                  .imageA        - Grayscale matrix of Image A.
%                  .imageB        - Grayscale matrix of Image B.
%                  .pointsA       - Validated cornerPoints / ORBPoints in Image A.
%                  .pointsB       - Validated cornerPoints / ORBPoints in Image B.
%                  .matches       - M-by-2 index matrix of matched feature pairs.
%                  .matchedPtsA   - Spatial coordinates of matches in Image A.
%                  .matchedPtsB   - Spatial coordinates of matches in Image B.
%
% EXAMPLE:
%   res = feature_matching_demo();
%   figure; showMatchedFeatures(res.imageA, res.imageB, res.matchedPtsA, res.matchedPtsB, 'montage');

    if nargin < 1 || isempty(imagePathA) || exist(imagePathA, 'file') ~= 2
        rawA = imread('cameraman.tif');
    else
        rawA = imread(imagePathA);
    end

    if nargin < 2 || isempty(imagePathB) || exist(imagePathB, 'file') ~= 2
        rawB = imrotate(rawA, 15, 'bilinear', 'crop');
    else
        rawB = imread(imagePathB);
    end

    if size(rawA, 3) == 3; grayA = rgb2gray(rawA); else; grayA = rawA; end
    if size(rawB, 3) == 3; grayB = rgb2gray(rawB); else; grayB = rawB; end

    % Detect ORB Features
    pointsA = detectORBFeatures(grayA);
    pointsB = detectORBFeatures(grayB);

    % Extract Binary Descriptors
    [featuresA, validPointsA] = extractFeatures(grayA, pointsA);
    [featuresB, validPointsB] = extractFeatures(grayB, pointsB);

    % Match Features with Lowe's Ratio Test
    pairs = matchFeatures(featuresA, featuresB, 'Unique', true, 'MaxRatio', 0.75);

    matchedPtsA = validPointsA(pairs(:, 1));
    matchedPtsB = validPointsB(pairs(:, 2));

    result = struct();
    result.imageA      = grayA;
    result.imageB      = grayB;
    result.pointsA     = validPointsA;
    result.pointsB     = validPointsB;
    result.matches     = pairs;
    result.matchedPtsA = matchedPtsA;
    result.matchedPtsB = matchedPtsB;
end
