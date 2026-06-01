from cv_portfolio.feature_matching import match_orb_features
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Feature matching demo")
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    args = parser.parse_args()

    image_a = read_image(args.image_a)
    image_b = read_image(args.image_b)
    result = match_orb_features(image_a, image_b)
    print("matches", len(result.matches))


if __name__ == "__main__":
    main()
