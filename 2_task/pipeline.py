import argparse
from NER_inference import extract_animal
from Classification_inference import predict_animal


def check_text_image_match(text, image_path):
    #Extract the animal name from the user's natural language input
    text_animal = extract_animal(text)

    # Classify the animal present in the provided image
    image_animal = predict_animal(image_path)

    print("Text animal:", text_animal)
    print("Image animal:", image_animal)

    # If the NER model couldn't find any animal in the text,
    # we cannot confirm a match
    if text_animal is None:
        return False
    # Returns True if both models agree on the animal type
    return text_animal.lower() == image_animal.lower()


def main():
    parser = argparse.ArgumentParser(description="Check whether the text matches the animal in the image.")
    parser.add_argument("--text", type=str, required=True, help="Input text describing the animal")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")

    args = parser.parse_args()

    result = check_text_image_match(args.text, args.image)
    print("Match:", result)


if __name__ == "__main__":
    main()