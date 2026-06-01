from cv_portfolio.enhancement import build_enhancement_summary
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Image enhancement toolkit")
    parser.add_argument("image")
    args = parser.parse_args()

    image = read_image(args.image)
    summary = build_enhancement_summary(image)
    for name, value in summary.items():
        print(name, value.shape)


if __name__ == "__main__":
    main()
