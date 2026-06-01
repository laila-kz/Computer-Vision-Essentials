% TP4: Feature Extraction and Object Detection

clear; clc;

imagePath1 = fullfile('..', '..', 'materials', 'sample_tp4_a.png');
imagePath2 = fullfile('..', '..', 'materials', 'sample_tp4_b.png');

if exist(imagePath1, 'file') && exist(imagePath2, 'file')
    imageA = imread(imagePath1);
    imageB = imread(imagePath2);
else
    imageA = imread('cameraman.tif');
    imageB = imrotate(imageA, 8, 'bilinear', 'crop');
end

grayA = im2gray(imageA);
grayB = im2gray(imageB);

pointsA = detectORBFeatures(grayA);
pointsB = detectORBFeatures(grayB);
[featuresA, validPointsA] = extractFeatures(grayA, pointsA);
[featuresB, validPointsB] = extractFeatures(grayB, pointsB);

indexPairs = matchFeatures(featuresA, featuresB, 'Unique', true, 'MaxRatio', 0.75);
matchedPointsA = validPointsA(indexPairs(:, 1));
matchedPointsB = validPointsB(indexPairs(:, 2));

figure('Name', 'TP4 - Feature Matching');
showMatchedFeatures(imageA, imageB, matchedPointsA, matchedPointsB, 'montage');
title('ORB Feature Matching');
