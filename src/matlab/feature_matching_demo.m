function result = feature_matching_demo(imagePathA, imagePathB)
%FEATURE_MATCHING_DEMO ORB-style feature comparison.

    imageA = im2gray(imread(imagePathA));
    imageB = im2gray(imread(imagePathB));

    pointsA = detectORBFeatures(imageA);
    pointsB = detectORBFeatures(imageB);
    [featuresA, validPointsA] = extractFeatures(imageA, pointsA);
    [featuresB, validPointsB] = extractFeatures(imageB, pointsB);
    pairs = matchFeatures(featuresA, featuresB, 'Unique', true);

    result = struct();
    result.pointsA = validPointsA;
    result.pointsB = validPointsB;
    result.matches = pairs;
end
