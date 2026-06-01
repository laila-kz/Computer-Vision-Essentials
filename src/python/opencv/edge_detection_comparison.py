from cv_portfolio.edges import compare_edge_detectors
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Edge detection comparison")
    parser.add_argument("image")
    args = parser.parse_args()

    image = read_image(args.image)
    outputs = compare_edge_detectors(image)
    for name, value in outputs.items():
        print(name, value.shape)


if __name__ == "__main__":
    main()
