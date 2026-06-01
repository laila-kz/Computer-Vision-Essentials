function result = object_detection_demo(imagePath)
%OBJECT_DETECTION_DEMO Simple region-based detection demo.

    grayscale = im2gray(imread(imagePath));
    thresholdLevel = graythresh(grayscale);
    mask = imbinarize(grayscale, thresholdLevel);
    mask = bwareaopen(mask, 100);
    mask = imclose(mask, strel('disk', 2));
    stats = regionprops(mask, 'Area', 'BoundingBox', 'Centroid');

    result = struct();
    result.mask = mask;
    result.statistics = stats;
end
