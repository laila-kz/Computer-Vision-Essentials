%% TP2: Histogram Processing, Spatial Filtering, and Edge Detection
%
% THEORETICAL BACKGROUND:
%   - Histogram Equalization: Flattens probability density function p(r_k) = n_k / N,
%     yielding a uniform cumulative distribution function s_k = T(r_k) = (L-1) * sum_{j=0}^k p(r_j).
%   - Contrast-Limited Adaptive Histogram Equalization (CLAHE): Prevents noise
%     amplification in uniform regions by tile-based histogram equalization with clip limiting.
%   - 1st-Derivative Edge Detection (Sobel, Prewitt, Roberts):
%     Computes discrete spatial gradients Gx, Gy and gradient magnitude |grad I| = sqrt(Gx^2 + Gy^2).
%   - 2nd-Derivative Edge Detection (Laplacian of Gaussian - LoG):
%     Locates zero-crossings of the isotropic Laplacian operator del^2(G_sigma * I).
%   - Canny Edge Detector:
%     Multi-stage optimal detector: Gaussian smoothing -> Gradient magnitude & angle
%     -> Non-Maximum Suppression (NMS) -> Hysteresis thresholding.
%
% USAGE:
%   >> tp2_histogram_edges
%
% OUTPUT:
%   Comprehensive 6-panel visualization comparing histogram transformations
%   and differential edge detection operators.

clear; close all; clc;

%% 1. Image Acquisition & Grayscale Conversion
imagePath = fullfile('..', '..', 'materials', 'sample_tp2_image.png');
if exist(imagePath, 'file') == 2
    fprintf('[+] Loading workspace asset: %s\n', imagePath);
    inputImage = imread(imagePath);
else
    fprintf('[!] Sample asset not found. Falling back to built-in "cameraman.tif"...\n');
    inputImage = imread('cameraman.tif');
end

if size(inputImage, 3) == 3
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

%% 2. Histogram Enhancement Transformations
% Global Histogram Equalization
equalizedImage = histeq(grayImage);

% Contrast-Limited Adaptive Histogram Equalization (CLAHE)
claheImage = adapthisteq(grayImage, 'ClipLimit', 0.02, 'Distribution', 'uniform');

% Contrast stretching using linear saturation
contrastAdjusted = imadjust(grayImage);

%% 3. Spatial Gaussian Smoothing Preprocessing
sigma = 1.2;
smoothedImage = imgaussfilt(grayImage, sigma);

%% 4. Classical Differential Edge Detection
sobelEdges   = edge(smoothedImage, 'Sobel');
prewittEdges = edge(smoothedImage, 'Prewitt');
robertsEdges = edge(smoothedImage, 'Roberts');
logEdges     = edge(smoothedImage, 'log');
cannyEdges   = edge(smoothedImage, 'Canny', [0.08 0.20]);

fprintf('--------------------------------------------------\n');
fprintf(' Edge Detection Density Statistics:\n');
fprintf('   Sobel Edge Density    : %.2f%%\n', 100 * mean(sobelEdges(:)));
fprintf('   Prewitt Edge Density  : %.2f%%\n', 100 * mean(prewittEdges(:)));
fprintf('   Roberts Edge Density  : %.2f%%\n', 100 * mean(robertsEdges(:)));
fprintf('   LoG Edge Density      : %.2f%%\n', 100 * mean(logEdges(:)));
fprintf('   Canny Edge Density    : %.2f%%\n', 100 * mean(cannyEdges(:)));
fprintf('--------------------------------------------------\n');

%% 5. Multi-Panel Figure Generation
fig = figure('Name', 'TP2 - Histograms, Spatial Filtering, and Edge Detection', ...
             'NumberTitle', 'off', 'Color', [1 1 1], 'Position', [80 80 1050 700]);

subplot(2, 3, 1);
imshow(grayImage);
title('1. Original Grayscale', 'FontWeight', 'bold');

subplot(2, 3, 2);
imshow(equalizedImage);
title('2. Global Hist Equalization', 'FontWeight', 'bold');

subplot(2, 3, 3);
imshow(claheImage);
title('3. CLAHE Enhancement', 'FontWeight', 'bold');

subplot(2, 3, 4);
imshow(sobelEdges);
title('4. Sobel 1st Derivative', 'FontWeight', 'bold');

subplot(2, 3, 5);
imshow(logEdges);
title('5. LoG Zero-Crossing', 'FontWeight', 'bold');

subplot(2, 3, 6);
imshow(cannyEdges);
title('6. Canny Optimal Detector', 'FontWeight', 'bold');
