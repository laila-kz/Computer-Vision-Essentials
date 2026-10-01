%% TP4: Local Feature Extraction, Binary Descriptors, and Robust Matching
%
% THEORETICAL BACKGROUND:
%   - ORB (Oriented FAST and Rotated BRIEF):
%       * Fast interest point detector across scale pyramid levels.
%       * Intensity Centroid orientation assignment for rotation invariance:
%           theta = atan2(m01, m10)
%       * Steered BRIEF binary descriptor computation:
%           f_n(p) = sum_{1 <= i <= n} 2^(i-1) * tau(p; x_i, y_i)
%   - Feature Correspondence Matching:
%       * Hamming distance calculation: d_H(a, b) = count_ones(a XOR b).
%       * Lowe's Ratio Test: d(best_1) / d(best_2) < MaxRatio to eliminate ambiguous points.
%
% USAGE:
%   >> tp4_feature_matching_detection
%
% OUTPUT:
%   Figure showing detected keypoints in both views and matched feature
%   correspondence lines across rotation/scale variations.

clear; close all; clc;

%% 1. Image Acquisition & View Synthesis
imagePath1 = fullfile('..', '..', 'materials', 'sample_tp4_a.png');
imagePath2 = fullfile('..', '..', 'materials', 'sample_tp4_b.png');

if exist(imagePath1, 'file') == 2 && exist(imagePath2, 'file') == 2
    fprintf('[+] Loading query and target assets: %s, %s\n', imagePath1, imagePath2);
    imageA = imread(imagePath1);
    imageB = imread(imagePath2);
else
    fprintf('[!] Assets not found. Creating synthesized rotated view from built-in "cameraman.tif"...\n');
    imageA = imread('cameraman.tif');
    imageB = imrotate(imageA, 18, 'bilinear', 'crop');
end

if size(imageA, 3) == 3; grayA = rgb2gray(imageA); else; grayA = imageA; end
if size(imageB, 3) == 3; grayB = rgb2gray(imageB); else; grayB = imageB; end

%% 2. ORB Feature Detection
pointsA = detectORBFeatures(grayA, 'NumLevels', 8, 'ScaleFactor', 1.2);
pointsB = detectORBFeatures(grayB, 'NumLevels', 8, 'ScaleFactor', 1.2);

fprintf('--------------------------------------------------\n');
fprintf(' Image A ORB Keypoints Detected : %d\n', pointsA.Count);
fprintf(' Image B ORB Keypoints Detected : %d\n', pointsB.Count);
fprintf('--------------------------------------------------\n');

%% 3. Binary Descriptor Extraction
[featuresA, validPointsA] = extractFeatures(grayA, pointsA);
[featuresB, validPointsB] = extractFeatures(grayB, pointsB);

%% 4. Match Features using Hamming Distance and Lowe's Ratio Test
maxRatio = 0.75;
indexPairs = matchFeatures(featuresA, featuresB, 'MatchThreshold', 40.0, ...
                           'MaxRatio', maxRatio, 'Unique', true);

matchedPointsA = validPointsA(indexPairs(:, 1));
matchedPointsB = validPointsB(indexPairs(:, 2));

fprintf(' Validated Matched Feature Pairs : %d\n', size(indexPairs, 1));
fprintf('--------------------------------------------------\n');

%% 5. Visualization of Keypoints and Matches
fig = figure('Name', 'TP4 - ORB Feature Extraction and Correspondence Matching', ...
             'NumberTitle', 'off', 'Color', [1 1 1], 'Position', [100 100 1100 600]);

subplot(2, 2, 1);
imshow(imageA); hold on;
plot(pointsA.selectStrongest(100));
title(sprintf('1. Image A Keypoints (Top 100 of %d)', pointsA.Count), 'FontWeight', 'bold');
hold off;

subplot(2, 2, 2);
imshow(imageB); hold on;
plot(pointsB.selectStrongest(100));
title(sprintf('2. Image B Keypoints (Top 100 of %d)', pointsB.Count), 'FontWeight', 'bold');
hold off;

subplot(2, 2, [3 4]);
showMatchedFeatures(imageA, imageB, matchedPointsA, matchedPointsB, 'montage');
title(sprintf('3. ORB Feature Correspondences (%d Matched Pairs, Ratio < %.2f)', ...
      size(indexPairs, 1), maxRatio), 'FontWeight', 'bold');
