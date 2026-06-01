function result = image_enhancement_toolkit(imagePath)
%IMAGE_ENHANCEMENT_TOOLKIT Enhancement utilities for course demos.

    original = imread(imagePath);
    grayscale = im2gray(original);
    equalized = histeq(grayscale);
    claheImage = adapthisteq(grayscale);
    gammaCorrected = imadjust(grayscale, [], [], 1.2);

    result = struct();
    result.original = original;
    result.grayscale = grayscale;
    result.equalized = equalized;
    result.clahe = claheImage;
    result.gammaCorrected = gammaCorrected;
end
