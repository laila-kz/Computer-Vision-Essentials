% TP2: Histogram Processing and Edge Detection

clear; clc;

imagePath = fullfile('..', '..', 'materials', 'sample_tp2_image.png');
if exist(imagePath, 'file')
    image = imread(imagePath);
else
    image = imread('cameraman.tif');
end

if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end

equalizedImage = histeq(grayImage);
contrastImage = imadjust(grayImage);
smoothedImage = imgaussfilt(grayImage, 1.2);

sobelEdges = edge(smoothedImage, 'Sobel');
logEdges = edge(smoothedImage, 'log');
cannyEdges = edge(smoothedImage, 'Canny');

figure('Name', 'TP2 - Histogram and Edges');
subplot(2, 3, 1); imshow(grayImage); title('Original Gray');
subplot(2, 3, 2); imshow(equalizedImage); title('Histogram Equalized');
subplot(2, 3, 3); imshow(contrastImage); title('Contrast Adjusted');
subplot(2, 3, 4); imshow(sobelEdges); title('Sobel');
subplot(2, 3, 5); imshow(logEdges); title('LoG');
subplot(2, 3, 6); imshow(cannyEdges); title('Canny');
