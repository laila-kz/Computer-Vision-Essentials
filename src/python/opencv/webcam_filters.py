from cv_portfolio.webcam_filters import build_webcam_filter_map
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Webcam-style filter demo")
    parser.add_argument("frame_image")
    args = parser.parse_args()

    frame = read_image(args.frame_image)
    outputs = build_webcam_filter_map(frame)
    for name, value in outputs.items():
        print(name, value.shape)


if __name__ == "__main__":
    main()
