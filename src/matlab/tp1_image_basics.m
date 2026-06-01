% TP1: Image Fundamentals and Basic Operations

clear; clc;

imagePath = fullfile('..', '..', 'materials', 'sample_tp1_image.png');
if exist(imagePath, 'file')
    originalImage = imread(imagePath);
else
    originalImage = imread('peppers.png');
end

if size(originalImage, 3) == 3
    grayImage = rgb2gray(originalImage);
else
    grayImage = originalImage;
end

normalizedImage = mat2gray(grayImage);
invertedImage = imcomplement(grayImage);
brightenedImage = imadjust(grayImage, [], [], 0.8);

figure('Name', 'TP1 - Image Fundamentals');
subplot(2, 2, 1); imshow(originalImage); title('Original');
subplot(2, 2, 2); imshow(grayImage); title('Grayscale');
subplot(2, 2, 3); imshow(normalizedImage); title('Normalized');
subplot(2, 2, 4); imshowpair(invertedImage, brightenedImage, 'montage'); title('Inverted / Brightened');
