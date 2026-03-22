from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
import torch

classes = ["dog", "horse", "elephant", "butterfly", "chicken",
           "cat", "cow", "sheep", "spider", "squirrel"]

model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

# Templates for generating synthetic datasets
templates = [
    "There is a {} in the picture",
    "I see a {}",
    "This {} is big",
    "The {} is here",
    "Look at that {}",
    "A {} is running",
    "Do you see the {}?",
    "I think it is a {}",
    "This looks like a {}",
    "It seems to be a {}",
    "Probably a {}",
    "The animal is a {}",
    "This picture shows a {}",
    "That must be a {}",
    "In the image there is a {}",
    "It is a {}",
    "It's a {}",
    "That is a {}",
    "That's a {}",
    "Here is a {}",
    "Here’s a {}",
    "This is a {}",
]


texts  = []
labels = []
# Sentence and label generation:
# 1 means "this is the name of an animal," 0 means a regular word
for animal in classes:
    for t in templates:
        sentence = t.format(animal)
        words    = sentence.split()
        label    = [1 if w.strip(".,?") == animal else 0 for w in words]
        texts.append(sentence)
        labels.append(label)

# dataset
class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # We need to assign a label only to the first sub-token of the word.
        word_ids    = self.encodings.word_ids(batch_index=idx)
        word_labels = self.labels[idx]
        aligned     = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD]) are ignored when calculating losses (-100)
                aligned.append(-100)
            elif word_id != prev_word_id:
                # This is the first sub-token of the new word—let's assign it a real label
                aligned.append(word_labels[word_id] if word_id < len(word_labels) else -100)
            else:
                # Subsequent sub-tokens of the same word are ignored (-100)
                aligned.append(-100)
            prev_word_id = word_id

        item["labels"] = torch.tensor(aligned, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = AnimalDataset(texts, labels, tokenizer)

# Creating a model
# num_labels=2 (0: not an animal, 1: an animal)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=2,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=50,
)
# Using Hugging Face Trainer to Simplify the Training Process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained("ner_model")
tokenizer.save_pretrained("ner_model")