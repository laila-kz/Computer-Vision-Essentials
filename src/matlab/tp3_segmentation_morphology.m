%% TP3: Thresholding, Mathematical Morphology, and Region Segmentation
%
% THEORETICAL BACKGROUND:
%   - Otsu's Thresholding (graythresh): Selects the global threshold T* that
%     maximizes the between-class variance sigma_B^2(T) = w0(T)*w1(T)*[mu0(T) - mu1(T)]^2.
%   - Mathematical Morphology:
%       * Opening (imopen): Erosion followed by Dilation. Removes small isolated noise specs.
%       * Closing (imclose): Dilation followed by Erosion. Fills small internal holes and bridges gaps.
%       * Hole Filling (imfill): Reconstructs background holes enclosed within foreground regions.
%   - Connected Component Analysis (bwconncomp, regionprops):
%       Measures geometric invariants: Area, Centroids, Bounding Boxes, Circularity, and Eccentricity.
%
% USAGE:
%   >> tp3_segmentation_morphology
%
% OUTPUT:
%   Visual pipeline illustrating thresholding, morphology cleanup, label map,
%   and region bounding box annotations with tabular metrics.

clear; close all; clc;

%% 1. Image Acquisition & Grayscale Conversion
imagePath = fullfile('..', '..', 'materials', 'sample_tp3_image.png');
if exist(imagePath, 'file') == 2
    fprintf('[+] Loading workspace asset: %s\n', imagePath);
    inputImage = imread(imagePath);
else
    fprintf('[!] Sample asset not found. Falling back to built-in "coins.png"...\n');
    inputImage = imread('coins.png');
end

if size(inputImage, 3) == 3
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

%% 2. Otsu Automated Thresholding
thresholdLevel = graythresh(grayImage);
rawBinary = imbinarize(grayImage, thresholdLevel);

% Ensure foreground objects are logical 1 (white)
if mean(rawBinary(:)) > 0.5
    rawBinary = imcomplement(rawBinary);
end

fprintf('--------------------------------------------------\n');
fprintf(' Otsu Optimal Threshold Level : %.4f (Normalized [0, 1])\n', thresholdLevel);
fprintf(' Otsu Cutoff Intensity (uint8): %d / 255\n', round(thresholdLevel * 255));
fprintf('--------------------------------------------------\n');

%% 3. Mathematical Morphology Refinement Pipeline
seDisk = strel('disk', 3);

% Step A: Morphological Opening to remove small noise specks
openedMask = imopen(rawBinary, seDisk);

% Step B: Morphological Closing to seal perimeter fractures
closedMask = imclose(openedMask, seDisk);

% Step C: Binary Hole Filling to ensure solid object interiors
cleanedMask = imfill(closedMask, 'holes');

%% 4. Connected Component Labeling & Feature Extraction
connectedComps = bwconncomp(cleanedMask, 8);
regionPropsData = regionprops(connectedComps, 'Area', 'Centroid', 'BoundingBox', 'Circularity', 'Eccentricity');

numObjects = connectedComps.NumObjects;
fprintf(' Connected Components Detected : %d objects\n', numObjects);
fprintf('--------------------------------------------------\n');
fprintf(' %-6s | %-8s | %-12s | %-12s\n', 'ID', 'Area (px)', 'Circularity', 'Eccentricity');
fprintf('--------------------------------------------------\n');

for k = 1:numObjects
    fprintf(' #%-5d | %-8.0f | %-12.3f | %-12.3f\n', ...
        k, regionPropsData(k).Area, regionPropsData(k).Circularity, regionPropsData(k).Eccentricity);
end
fprintf('--------------------------------------------------\n');

%% 5. Multi-Panel Diagnostic Visualization
fig = figure('Name', 'TP3 - Thresholding, Morphology, and Object Segmentation', ...
             'NumberTitle', 'off', 'Color', [1 1 1], 'Position', [100 100 1000 700]);

subplot(2, 2, 1);
imshow(grayImage);
title('1. Input Grayscale', 'FontWeight', 'bold');

subplot(2, 2, 2);
imshow(rawBinary);
title(sprintf('2. Otsu Binary Mask (T = %.2f)', thresholdLevel * 255), 'FontWeight', 'bold');

subplot(2, 2, 3);
imshow(cleanedMask);
title('3. Morphologically Cleaned (Open + Close + Fill)', 'FontWeight', 'bold');

subplot(2, 2, 4);
imshow(grayImage);
title(sprintf('4. Detected Regions (%d Objects)', numObjects), 'FontWeight', 'bold');
hold on;

for k = 1:numObjects
    bb = regionPropsData(k).BoundingBox;
    cen = regionPropsData(k).Centroid;
    rectangle('Position', bb, 'EdgeColor', [0 0.8 0], 'LineWidth', 2);
    plot(cen(1), cen(2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    text(bb(1), max(10, bb(2) - 5), sprintf('#%d', k), 'Color', [1 1 0], ...
         'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0]);
end
hold off;
