% TP3: Thresholding, Morphology, and Segmentation

clear; clc;

imagePath = fullfile('..', '..', 'materials', 'sample_tp3_image.png');
if exist(imagePath, 'file')
    image = imread(imagePath);
else
    image = imread('coins.png');
end

if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end

thresholdLevel = graythresh(grayImage);
binaryMask = imbinarize(grayImage, thresholdLevel);
binaryMask = imcomplement(binaryMask);

se = strel('disk', 3);
cleanedMask = imopen(binaryMask, se);
cleanedMask = imclose(cleanedMask, se);
cleanedMask = imfill(cleanedMask, 'holes');

connectedComponents = bwconncomp(cleanedMask);
regionMeasurements = regionprops(connectedComponents, 'Area', 'Centroid', 'BoundingBox');

figure('Name', 'TP3 - Segmentation and Morphology');
subplot(2, 2, 1); imshow(grayImage); title('Gray Image');
subplot(2, 2, 2); imshow(binaryMask); title('Binary Mask');
subplot(2, 2, 3); imshow(cleanedMask); title('Cleaned Mask');
subplot(2, 2, 4); imshow(grayImage); title('Detected Regions'); hold on;
for k = 1:numel(regionMeasurements)
    rectangle('Position', regionMeasurements(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 1.5);
end
hold off;
