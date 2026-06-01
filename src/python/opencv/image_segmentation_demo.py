from cv_portfolio.segmentation import segmentation_pipeline
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Image segmentation demo")
    parser.add_argument("image")
    args = parser.parse_args()

    image = read_image(args.image)
    result = segmentation_pipeline(image)
    print("threshold", result["threshold"])
    print("labels", result["num_labels"])


if __name__ == "__main__":
    main()
