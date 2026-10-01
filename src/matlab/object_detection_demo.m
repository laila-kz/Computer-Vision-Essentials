function result = object_detection_demo(imagePath, minArea)
%OBJECT_DETECTION_DEMO Region-based object detection, filtering, and bounding box localization.
%
% THEORETICAL BACKGROUND:
%   Classical object detection localizes candidate regions of interest (ROIs)
%   by combining adaptive intensity thresholding with morphological area filtering:
%     1. Automated binarization separates foreground candidates from background.
%     2. Morphological opening/closing eliminates sensor noise and bridges fragments.
%     3. Area thresholding (bwareaopen) filters out spurious small blobs.
%     4. Regionprops extracts spatial geometry (BoundingBox, Centroid, Area, Aspect Ratio).
%
% SYNTAX:
%   result = object_detection_demo()
%   result = object_detection_demo(imagePath)
%   result = object_detection_demo(imagePath, minArea)
%
% INPUTS:
%   imagePath - (Optional) Path to input image file. Default: 'coins.png'.
%   minArea   - (Optional) Minimum pixel area threshold. Default: 80.
%
% OUTPUTS:
%   result    - MATLAB struct containing:
%                 .image        - Original image array.
%                 .grayscale    - Single-channel grayscale matrix.
%                 .mask         - Filtered binary foreground mask.
%                 .objectCount  - Number of localized target objects.
%                 .statistics   - Struct array of region properties.
%
% EXAMPLE:
%   res = object_detection_demo('coins.png', 100);
%   figure; imshow(res.image); hold on;
%   for k = 1:res.objectCount
%       rectangle('Position', res.statistics(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);
%   end

    if nargin < 1 || isempty(imagePath) || exist(imagePath, 'file') ~= 2
        image = imread('coins.png');
    else
        image = imread(imagePath);
    end

    if nargin < 2 || isempty(minArea)
        minArea = 80;
    end

    if size(image, 3) == 3
        grayscale = rgb2gray(image);
    else
        grayscale = image;
    end

    % Thresholding
    thresholdLevel = graythresh(grayscale);
    mask = imbinarize(grayscale, thresholdLevel);
    if mean(mask(:)) > 0.5
        mask = imcomplement(mask);
    end

    % Morphological filtering and noise suppression
    mask = bwareaopen(mask, minArea);
    mask = imclose(mask, strel('disk', 3));
    mask = imfill(mask, 'holes');

    % Extract regional statistics
    stats = regionprops(mask, 'Area', 'BoundingBox', 'Centroid', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength');
    objectCount = numel(stats);

    result = struct();
    result.image       = image;
    result.grayscale   = grayscale;
    result.mask        = mask;
    result.objectCount = objectCount;
    result.statistics  = stats;
end
