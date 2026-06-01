from cv_portfolio.object_detection import detect_faces
from cv_portfolio.utils import read_image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Face detection project")
    parser.add_argument("image")
    args = parser.parse_args()

    image = read_image(args.image)
    result = detect_faces(image)
    print("faces", len(result.boxes))


if __name__ == "__main__":
    main()
