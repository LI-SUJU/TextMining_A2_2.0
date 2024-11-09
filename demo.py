import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define paths to your files
train_file_path = './fold1/train.txt'
val_file_path = './fold1/val.txt'
test_file_path = './fold1/test.txt'

# Function to read and process the IOB data
def read_data(file_path):
    words, labels, sentences, sentence_labels = [], [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                if words:
                    sentences.append(words)
                    sentence_labels.append(labels)
                    words, labels = [], []
            else:
                word, _, label = line.strip().split()
                words.append(word)
                labels.append(label)
        if words:  # Add the last sentence if file does not end with a newline
            sentences.append(words)
            sentence_labels.append(labels)
    return sentences, sentence_labels

# Read train, validation, and test datasets
train_sentences, train_labels = read_data(train_file_path)
val_sentences, val_labels = read_data(val_file_path)
test_sentences, test_labels = read_data(test_file_path)

# Define a label map
unique_labels = set(label for labels in train_labels for label in labels)
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Load tokenizer and model
model_checkpoint = "distilbert-base-uncased"  # Replace with your model if needed
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize and align labels with tokens
def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True)
    label_all_tokens = True
    new_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])  # Convert label to ID
            else:
                label_ids.append(label2id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        new_labels.append(label_ids)
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# Tokenize datasets
train_encodings = tokenize_and_align_labels(train_sentences, train_labels)
val_encodings = tokenize_and_align_labels(val_sentences, val_labels)
test_encodings = tokenize_and_align_labels(test_sentences, test_labels)

# Convert data into Hugging Face Dataset
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)
test_dataset = Dataset.from_dict(test_encodings)

# Define model for token classification
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label2id))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = [label for labels in labels for label in labels if label != -100]
    true_predictions = [pred for preds, labels in zip(predictions, labels) for pred, label in zip(preds, labels) if label != -100]
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="weighted")
    accuracy = accuracy_score(true_labels, true_predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate(test_dataset)
print(test_results)
