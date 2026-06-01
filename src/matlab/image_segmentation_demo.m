function result = image_segmentation_demo(imagePath)
%IMAGE_SEGMENTATION_DEMO Thresholding and morphology pipeline.

    grayscale = im2gray(imread(imagePath));
    thresholdLevel = graythresh(grayscale);
    binaryMask = imbinarize(grayscale, thresholdLevel);
    se = strel('disk', 3);
    opened = imopen(binaryMask, se);
    closed = imclose(opened, se);
    labeled = bwlabel(closed);

    result = struct();
    result.grayscale = grayscale;
    result.binary = binaryMask;
    result.refined = closed;
    result.labels = labeled;
end
