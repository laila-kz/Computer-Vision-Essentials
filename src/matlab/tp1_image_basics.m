%% TP1: Image Fundamentals, Color Models, and Intensity Transformations
%
% THEORETICAL BACKGROUND:
%   A digital image is mathematically modeled as a 2D discrete function f(x, y),
%   where (x, y) denote spatial coordinates and f represents intensity/radiance.
%   In standard 8-bit digital imaging:
%     - Grayscale conversion uses standard ITU-R BT.601 luminance weights:
%         Y = 0.2989*R + 0.5870*G + 0.1140*B
%     - Linear Min-Max Normalization maps intensity range [min(I), max(I)] -> [0, 1]:
%         I_norm(x, y) = (I(x, y) - min(I)) / (max(I) - min(I))
%     - Bitwise Photographic Negative / Complement:
%         I_inv(x, y) = 255 - I(x, y)
%     - Power-Law / Gamma Transformation:
%         s = c * r^gamma
%
% USAGE:
%   Run directly in MATLAB editor or command window:
%     >> tp1_image_basics
%
% OUTPUT:
%   Figure window displaying original, grayscale, normalized, inverted,
%   and gamma-adjusted images alongside quantitative channel statistics.

clear; close all; clc;

%% 1. Image Acquisition & Path Resolution
imagePath = fullfile('..', '..', 'materials', 'sample_tp1_image.png');
if exist(imagePath, 'file') == 2
    fprintf('[+] Loading workspace asset: %s\n', imagePath);
    originalImage = imread(imagePath);
else
    fprintf('[!] Sample asset not found. Falling back to built-in "peppers.png"...\n');
    originalImage = imread('peppers.png');
end

%% 2. Dimensional Analysis & Color Space Transformation
[height, width, channels] = size(originalImage);
fprintf('--------------------------------------------------\n');
fprintf(' Image Dimensions : %d x %d pixels\n', height, width);
fprintf(' Color Channels   : %d (%s)\n', channels, getChannelDesc(channels));
fprintf(' Data Type        : %s\n', class(originalImage));
fprintf('--------------------------------------------------\n');

if channels == 3
    grayImage = rgb2gray(originalImage);
else
    grayImage = originalImage;
end

%% 3. Fundamental Image Processing Operations
% A. Floating-point intensity normalization to [0, 1]
normalizedImage = mat2gray(grayImage);

% B. Photographic complement / intensity inversion
invertedImage = imcomplement(grayImage);

% C. Power-law (Gamma) dynamic range adjustment (gamma = 0.7: brighten midtones)
gammaBright = imadjust(grayImage, [], [], 0.7);

% D. Power-law (Gamma) dynamic range adjustment (gamma = 1.6: darken midtones)
gammaDark = imadjust(grayImage, [], [], 1.6);

%% 4. Statistical Metrics Extraction
meanVal = mean(double(grayImage(:)));
stdVal  = std(double(grayImage(:)));
minVal  = min(grayImage(:));
maxVal  = max(grayImage(:));

fprintf(' Grayscale Statistics:\n');
fprintf('   Min Intensity : %d\n', minVal);
fprintf('   Max Intensity : %d\n', maxVal);
fprintf('   Mean (mu)     : %.2f\n', meanVal);
fprintf('   Std Dev (sig) : %.2f\n', stdVal);
fprintf('--------------------------------------------------\n');

%% 5. Multi-Panel Diagnostic Visualization
fig = figure('Name', 'TP1 - Image Fundamentals and Transformations', ...
             'NumberTitle', 'off', 'Color', [1 1 1], 'Position', [100 100 950 650]);

subplot(2, 3, 1);
imshow(originalImage);
title('1. Original Input', 'FontWeight', 'bold');

subplot(2, 3, 2);
imshow(grayImage);
title(sprintf('2. Grayscale (mu=%.1f, sig=%.1f)', meanVal, stdVal), 'FontWeight', 'bold');

subplot(2, 3, 3);
imshow(normalizedImage);
title('3. Normalized [0.0, 1.0]', 'FontWeight', 'bold');

subplot(2, 3, 4);
imshow(invertedImage);
title('4. Inverted (255 - I)', 'FontWeight', 'bold');

subplot(2, 3, 5);
imshow(gammaBright);
title('5. Gamma = 0.7 (Brighten)', 'FontWeight', 'bold');

subplot(2, 3, 6);
imshow(gammaDark);
title('6. Gamma = 1.6 (Darken)', 'FontWeight', 'bold');

%% Helper Local Functions
function desc = getChannelDesc(numChannels)
    if numChannels == 1
        desc = 'Grayscale / Monochromatic';
    elseif numChannels == 3
        desc = 'Truecolor RGB';
    else
        desc = 'Multi-spectral';
    end
end
