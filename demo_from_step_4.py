import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report

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
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label2id))

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

# Convert data into Hugging Face Dataset and DataLoader
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)
test_dataset = Dataset.from_dict(test_encodings)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=DataCollatorForTokenClassification(tokenizer))
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=DataCollatorForTokenClassification(tokenizer))
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=DataCollatorForTokenClassification(tokenizer))

# Set up AdamW optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Define training loop with validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluate on validation set at the end of each epoch
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_preds.extend(predictions.cpu().numpy().tolist())
            val_labels.extend(batch["labels"].cpu().numpy().tolist())

    # Flatten lists and filter out `-100` tokens
    val_labels_flat = [label for labels in val_labels for label in labels if label != -100]
    val_preds_flat = [pred for preds, labels in zip(val_preds, val_labels) for pred, label in zip(preds, labels) if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(val_labels_flat, val_preds_flat, average="weighted")
    print(f"Epoch {epoch + 1} - Validation Precision: {precision}, Recall: {recall}, F1-score: {f1}")

from sklearn.metrics import classification_report

def evaluate_test_set():
    model.eval()
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            test_preds.extend(predictions.cpu().numpy().tolist())
            test_labels.extend(batch["labels"].cpu().numpy().tolist())

    # Flatten lists and filter out `-100` tokens
    test_labels_flat = [label for labels in test_labels for label in labels if label != -100]
    test_preds_flat = [pred for preds, labels in zip(test_preds, test_labels) for pred, label in zip(preds, labels) if label != -100]

    # Ensure that label_names only includes labels that are actually in the test set
    labels_in_test_set = sorted(set(test_labels_flat))
    label_names = [id2label[i] for i in labels_in_test_set]

    # Get classification report
    report = classification_report(
        test_labels_flat, 
        test_preds_flat, 
        target_names=label_names, 
        labels=labels_in_test_set,  # Specify the exact labels present in test set
        zero_division=1
    )
    print("Test set evaluation:")
    print(report)

# Run the extended evaluation on the test set
evaluate_test_set()

