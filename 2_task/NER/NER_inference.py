from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ner_model")
model = AutoModelForTokenClassification.from_pretrained("ner_model")

def extract_animal(text):
    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    # Get the predicted class index for each token (argmax along the label dimension)
    # 0 = Not an animal, 1 = Animal
    preds = torch.argmax(outputs.logits, dim=2)[0]
    # Convert numerical IDs back into human-readable tokens (including sub-words like '##ly')
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    animal = ""
    for token, label in zip(tokens, preds):
        # If the label is 1, the model has identified this token as part of an animal name
        if label.item() == 1:
            # If a token starts with '##', it is a continuation of the previous word
            if token.startswith("##"):
                animal += token[2:]
            else:
                # If it's a new word, we assign/reset the animal string
                animal = token

    return animal if animal else None
