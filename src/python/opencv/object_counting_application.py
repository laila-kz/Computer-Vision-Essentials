from cv_portfolio.object_detection import count_contours
from cv_portfolio.segmentation import morphology_cleanup, otsu_threshold
from cv_portfolio.utils import read_image, to_gray


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Object counting application")
    parser.add_argument("image")
    parser.add_argument("--min-area", type=int, default=100)
    args = parser.parse_args()

    image = read_image(args.image)
    _, binary = otsu_threshold(image)
    cleaned = morphology_cleanup(binary)
    count, _ = count_contours(cleaned, min_area=args.min_area)
    print("count", count)


if __name__ == "__main__":
    main()
