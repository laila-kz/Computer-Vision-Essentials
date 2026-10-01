function result = image_segmentation_demo(imagePath)
%IMAGE_SEGMENTATION_DEMO Binarization, morphological filtering, and connected components pipeline.
%
% THEORETICAL BACKGROUND:
%   Implements classical region segmentation:
%     1. Otsu Variance Maximization: Calculates optimal binarization threshold T*.
%     2. Morphological Opening: Removes isolated noise artifacts using disk structuring element.
%     3. Morphological Closing: Bridges gaps and seals perimeter discontinuities.
%     4. Connected Component Labeling: Identifies contiguous 8-connected regions and measures area.
%
% SYNTAX:
%   result = image_segmentation_demo()
%   result = image_segmentation_demo(imagePath)
%
% INPUTS:
%   imagePath - (Optional) Path to input image file. Default: 'coins.png'.
%
% OUTPUTS:
%   result    - MATLAB struct containing:
%                 .grayscale  - Input grayscale matrix.
%                 .threshold  - Otsu threshold cutoff level [0, 1].
%                 .binary     - Raw binarized binary mask.
%                 .refined    - Morphologically cleaned binary mask.
%                 .labels     - 2D integer label matrix from bwlabel.
%                 .numObjects - Total count of detected foreground objects.
%                 .stats      - Region properties struct array (Area, Centroid, BoundingBox).
%
% EXAMPLE:
%   res = image_segmentation_demo('coins.png');
%   figure; imshow(label2rgb(res.labels));

    if nargin < 1 || isempty(imagePath) || exist(imagePath, 'file') ~= 2
        raw = imread('coins.png');
    else
        raw = imread(imagePath);
    end

    if size(raw, 3) == 3
        grayscale = rgb2gray(raw);
    else
        grayscale = raw;
    end

    % 1. Otsu Binarization
    thresholdLevel = graythresh(grayscale);
    binaryMask = imbinarize(grayscale, thresholdLevel);
    if mean(binaryMask(:)) > 0.5
        binaryMask = imcomplement(binaryMask);
    end

    % 2. Morphological Refinement
    se = strel('disk', 3);
    opened = imopen(binaryMask, se);
    closed = imclose(opened, se);
    filled = imfill(closed, 'holes');

    % 3. Connected Component Analysis
    [labeled, numObjects] = bwlabel(filled, 8);
    stats = regionprops(labeled, 'Area', 'BoundingBox', 'Centroid', 'Circularity');

    result = struct();
    result.grayscale  = grayscale;
    result.threshold  = thresholdLevel;
    result.binary     = binaryMask;
    result.refined    = filled;
    result.labels     = labeled;
    result.numObjects = numObjects;
    result.stats      = stats;
end
