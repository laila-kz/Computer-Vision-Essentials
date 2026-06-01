function result = edge_detection_comparison(imagePath)
%EDGE_DETECTION_COMPARISON Compare classical edge detectors.

    grayscale = im2gray(imread(imagePath));
    sobelEdges = edge(grayscale, 'sobel');
    prewittEdges = edge(grayscale, 'prewitt');
    robertsEdges = edge(grayscale, 'roberts');
    cannyEdges = edge(grayscale, 'canny');

    result = struct();
    result.grayscale = grayscale;
    result.sobel = sobelEdges;
    result.prewitt = prewittEdges;
    result.roberts = robertsEdges;
    result.canny = cannyEdges;
end
