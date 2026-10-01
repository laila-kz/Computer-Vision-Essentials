function result = edge_detection_comparison(imagePath)
%EDGE_DETECTION_COMPARISON Multi-operator spatial edge detection benchmark.
%
% THEORETICAL BACKGROUND:
%   Edge detection algorithms locate sharp intensity discontinuities corresponding
%   to object boundaries, surface orientation changes, or material properties.
%   This function computes and compares:
%     1. Sobel Gradient: 3x3 directional derivative kernels with smoothing.
%     2. Prewitt Gradient: 3x3 uniform directional derivative kernels.
%     3. Roberts Cross: 2x2 diagonal difference operator.
%     4. Laplacian of Gaussian (LoG): 2nd-derivative zero-crossing detector.
%     5. Canny Edge Detector: Multi-stage optimal edge detection with NMS and hysteresis.
%
% SYNTAX:
%   result = edge_detection_comparison()
%   result = edge_detection_comparison(imagePath)
%
% INPUTS:
%   imagePath - (Optional) String or char array containing path to input image file.
%               If omitted or not found, defaults to MATLAB built-in 'cameraman.tif'.
%
% OUTPUTS:
%   result    - MATLAB struct containing the following fields:
%                 .grayscale - Original single-channel grayscale matrix.
%                 .sobel     - Binary edge map from Sobel operator.
%                 .prewitt   - Binary edge map from Prewitt operator.
%                 .roberts   - Binary edge map from Roberts Cross operator.
%                 .log       - Binary edge map from Laplacian of Gaussian.
%                 .canny     - Binary edge map from Canny detector.
%
% EXAMPLE:
%   res = edge_detection_comparison('cameraman.tif');
%   figure; imshow(res.canny); title('Canny Edges');

    if nargin < 1 || isempty(imagePath) || exist(imagePath, 'file') ~= 2
        raw = imread('cameraman.tif');
    else
        raw = imread(imagePath);
    end

    if size(raw, 3) == 3
        grayscale = rgb2gray(raw);
    else
        grayscale = raw;
    end

    % Apply gentle Gaussian smoothing to stabilize gradients
    smoothed = imgaussfilt(grayscale, 1.0);

    % Compute edge maps
    sobelEdges   = edge(smoothed, 'sobel');
    prewittEdges = edge(smoothed, 'prewitt');
    robertsEdges = edge(smoothed, 'roberts');
    logEdges     = edge(smoothed, 'log');
    cannyEdges   = edge(smoothed, 'canny');

    % Pack results structure
    result = struct();
    result.grayscale = grayscale;
    result.sobel     = sobelEdges;
    result.prewitt   = prewittEdges;
    result.roberts   = robertsEdges;
    result.log       = logEdges;
    result.canny     = cannyEdges;
end
