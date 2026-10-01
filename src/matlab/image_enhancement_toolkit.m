function result = image_enhancement_toolkit(imagePath)
%IMAGE_ENHANCEMENT_TOOLKIT Dynamic range expansion and contrast enhancement suite.
%
% THEORETICAL BACKGROUND:
%   Provides foundational intensity enhancement transformations:
%     1. Contrast Stretching: Stretches pixel dynamic range linearly to [0, 255].
%     2. Global Histogram Equalization: Maps intensity PDF to a uniform distribution.
%     3. CLAHE (Contrast-Limited Adaptive Histogram Equalization): Equalizes contrast
%        locally in contextual tiles while clipping high frequency noise.
%     4. Gamma Power-Law Correction: Applies s = c * r^gamma for tone curve manipulation.
%
% SYNTAX:
%   result = image_enhancement_toolkit()
%   result = image_enhancement_toolkit(imagePath)
%
% INPUTS:
%   imagePath - (Optional) Path to input image file. Default: 'peppers.png'.
%
% OUTPUTS:
%   result    - MATLAB struct containing:
%                 .original        - Original input image matrix.
%                 .grayscale       - Grayscale converted matrix.
%                 .contrastStretch - Linearly normalized matrix.
%                 .equalized       - Global histogram equalized image.
%                 .clahe           - CLAHE enhanced image.
%                 .gammaBright     - Gamma corrected (gamma = 0.7) image.
%                 .gammaDark       - Gamma corrected (gamma = 1.5) image.
%
% EXAMPLE:
%   res = image_enhancement_toolkit('peppers.png');
%   figure; imshowpair(res.grayscale, res.clahe, 'montage');

    if nargin < 1 || isempty(imagePath) || exist(imagePath, 'file') ~= 2
        original = imread('peppers.png');
    else
        original = imread(imagePath);
    end

    if size(original, 3) == 3
        grayscale = rgb2gray(original);
    else
        grayscale = original;
    end

    % 1. Contrast Stretching
    contrastStretch = imadjust(grayscale);

    % 2. Global Histogram Equalization
    equalized = histeq(grayscale);

    % 3. Adaptive Histogram Equalization (CLAHE)
    claheImage = adapthisteq(grayscale, 'ClipLimit', 0.02, 'Distribution', 'uniform');

    % 4. Gamma Adjustments
    gammaBright = imadjust(grayscale, [], [], 0.7);
    gammaDark   = imadjust(grayscale, [], [], 1.5);

    result = struct();
    result.original        = original;
    result.grayscale       = grayscale;
    result.contrastStretch = contrastStretch;
    result.equalized       = equalized;
    result.clahe           = claheImage;
    result.gammaBright     = gammaBright;
    result.gammaDark       = gammaDark;
end
