from collections import Counter

import optuna
from jedi.inference import references
from sklearn.metrics import precision_score, recall_score, f1_score
from sympy.core.random import sample
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification
from datasets import load_dataset, ClassLabel, Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch


# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=13)


# def tokenize_function(example):
#     return tokenizer(example["tokens"], truncation=True)

def creat_dataset(file_path):
    x = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        paragraphs = text.split('\n\n')

        split_paragraphs = [paragraph.splitlines() for paragraph in paragraphs]
    for line in split_paragraphs:
        tokens = []
        POS = []
        ner_tags = []
        for sentence in line:
            token, pos, label = sentence.split()
            tokens.append(token)
            POS.append(pos)
            ner_tags.append(label)
        data = {"tokens": tokens, "pos_tags": POS, "labels": ner_tags}
        x.append(data)
    dataset = Dataset.from_list(x)
    return dataset


def filter_nones(example):
    return example["text"] != ''


def process_data(example):
    example["tokens"] = example["text"].split()[0]
    example["POS"] = example["text"].split()[1]
    example["label"] = example["text"].split()[2]
    return example


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def encode_labels(example):
    example["labels"] = [class_label.str2int(label) for label in example["labels"]]
    return example


data_files = {"train": "train.txt", "test": "test.txt", "validation": "val.txt"}
dataset_sum = DatasetDict({"train": creat_dataset(data_files["train"]), "test": creat_dataset(data_files["test"]),
                           "validation": creat_dataset(data_files["validation"])})
print(dataset_sum)
# squad_it_dataset = load_dataset("text", data_files=data_files,sample_by= "paragraph")
# dataset=squad_it_dataset
# print(dataset["train"][0])
# dataset=dataset.filter(filter_nones)
# dataset=dataset.map(process_data)
# dataset=dataset.remove_columns("text")
label_names = np.unique(sum(dataset_sum["train"]["labels"], [])).tolist()
print(label_names)
class_label = ClassLabel(names=np.unique(sum(dataset_sum["train"]["labels"], [])).tolist())
dataset = dataset_sum.map(encode_labels)

dataset = dataset.rename_column("labels", "ner_tags")
# print(dataset)
y = dataset["train"][0]
labels = dataset["train"][0]["ner_tags"]
# print(labels)
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# print(dataset["train"][0]["ner_tags"])
# print(",,,,,,,,,,")
# ner_feature = dataset["train"].features["ner_tags"]
# print(ner_feature)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


metric = evaluate.load("seqeval")

# training_args = TrainingArguments("test-trainer")

id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id,
# )
# print(model.config.num_labels)

# args = TrainingArguments(
#     "bert-finetuned-ner",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     # push_to_hub=True,
# )

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer,
# )
#
# trainer.train()
# test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
# print("----------------------------")
# print(test_results)

# predictions = trainer.predict(tokenized_datasets["validation"])
#
# metric=evaluate.load(dataset)
# print(predictions.predictions.shape, predictions.label_ids.shape)
# print(dataset["train"]["label"])
# lable2id={"B-ART": 0, "B-CON": 1, "B-LOC": 2,  "B-MAT": 3, "B-PER": 4, "B-SPE": 5, "I-ART": 6,  "I-CON": 7, "I-LOC": 8,
# "I-MAT": 9,"I-PER": 10,"I-SPE": 11,"0": 12}
#
# dataset_aligned= dataset.align_labels_with_mapping(lable2id,"label")
# print(dataset)
# x=dataset['train'][3]
# print(x)


#  task3


train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=8)
#
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# optimizer = AdamW(model.parameters(), lr=2e-5)
#
#
#
# accelerator = Accelerator()
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
#
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )
#
output_dir = "bert-finetuned-ner-accelerate"


# # # repo = Repository(output_dir, clone_from=repo_name)
# #
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


#
#
# progress_bar = tqdm(range(num_training_steps))
#
# for epoch in range(num_train_epochs):
#     # Training
#     model.train()
#     for batch in train_dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#
#     # Evaluation
#     model.eval()
#     for batch in eval_dataloader:
#         with torch.no_grad():
#             outputs = model(**batch)
#
#         predictions = outputs.logits.argmax(dim=-1)
#         labels = batch["labels"]
#
#         # Necessary to pad predictions and labels for being gathered
#         predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
#         labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
#
#         predictions_gathered = accelerator.gather(predictions)
#         labels_gathered = accelerator.gather(labels)
#
#         true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
#         metric.add_batch(predictions=true_predictions, references=true_labels)
#
#     results = metric.compute()
#     print(
#         f"epoch {epoch}:",
#         {
#             key: results[f"overall_{key}"]
#             for key in ["precision", "recall", "f1", "accuracy"]
#         },
#     )
#
#     # Save and upload
#     accelerator.wait_for_everyone()
#     unwrapped_model = accelerator.unwrap_model(model)
#     unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
#     if accelerator.is_main_process:
#         tokenizer.save_pretrained(output_dir)
#         # repo.push_to_hub(
#         #     commit_message=f"Training in progress epoch {epoch}", blocking=False
#         # )


#         test
# def find_lable(predictions, references):
#     tp, fp, fn = 0, 0, 0
#
#
#     for pred_seq, ref_seq in zip(predictions, references):
#         for pred, ref in zip(pred_seq, ref_seq):
#             # 忽略 "O" 标签，仅计算非 "O" 标签的统计
#             if ref != "O" and pred != "O":
#                 if pred == ref:
#                     tp += 1  # True Positive：预测正确的标签
#                 else:
#                     fp += 1  # False Positive：预测错误的标签
#                     fn += 1  # False Negative：漏掉的正确标签
#             elif pred != "O" and ref == "O":
#                 fp += 1  # False Positive：预测为非 "O"，但真实标签是 "O"
#             elif pred == "O" and ref != "O":
#                 fn += 1  # False Negative：真实标签是非 "O"，但预测为 "O"
#
#
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     print(tp, fp, fn)
#     print("Overall Non-O Precision:", precision)
#     print("Overall Non-O Recall:", recall)
#     print("Overall Non-O F1 Score:", f1_score)
#
#
#
# accelerator = Accelerator()

model = AutoModelForTokenClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)


#
#
# model.eval()
# model.to(accelerator.device)
#
# test_dataloader = accelerator.prepare(test_dataloader)
#
#
# all_predictions = []
# all_labels = []
#
# for batch in tqdm(test_dataloader, desc="Evaluating on test set"):
#
#     batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#
#     with torch.no_grad():
#
#         outputs = model(**batch)
#
#
#     predictions = outputs.logits.argmax(dim=-1)
#     labels = batch["labels"]
#
#
#     predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
#     labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
#
#
#     predictions_gathered = accelerator.gather(predictions)
#     labels_gathered = accelerator.gather(labels)
#
#
#
#     true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
#
#
#     all_predictions.extend(true_predictions)
#     all_labels.extend(true_labels)
# # find_lable(all_predictions, all_labels)
#
# results = metric.compute(predictions=all_predictions, references=all_labels)
#
#
# print("Test Set Results:")
# for key, value in results.items():
#     if isinstance(value, (float, int)):
#         print(f"{key}: {value:.4f}")
#     else:
#         print(f"{key}: {value}")

def optimization(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    num_training_steps = num_train_epochs * len(train_dataloader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
                              num_training_steps=num_training_steps)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
        return results["overall_accuracy"]


# study = optuna.create_study(direction="maximize")
# study.optimize(optimization, n_trials=50)
# print("Best hyperparameters:", study.best_params)


def res_test():
    learning_rate = 2.3817788032560365e-05
    batch_size = 32
    weight_decay = 0.0011565812915545555
    model = AutoModelForTokenClassification.from_pretrained(output_dir)

    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      "weight_decay": weight_decay},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0},
    # ]
    optimizer = AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)

    num_training_steps = num_train_epochs * len(train_dataloader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps),
                              num_training_steps=num_training_steps)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("new-model", save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained("new-model")

        # model.eval()
        # model.to(accelerator.device)
        # # 准备测试集数据加载器
        # test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=8)
        # test_dataloader = accelerator.prepare(test_dataloader)
        #
        # # 评估循环
        # all_predictions = []
        # all_labels = []
        #
        # for batch in tqdm(test_dataloader, desc="Evaluating on test set"):
        #     # 将 batch 中的所有张量移动到相同设备
        #     batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        #
        #     with torch.no_grad():
        #         # 前向传播，得到预测输出
        #         outputs = model(**batch)
        #
        #     # 获取预测值
        #     predictions = outputs.logits.argmax(dim=-1)
        #     labels = batch["labels"]
        #
        #     # 将 predictions 和 labels 填充并进行多进程收集
        #     predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        #     labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        #
        #     # 收集所有进程的 predictions 和 labels
        #     predictions_gathered = accelerator.gather(predictions)
        #     labels_gathered = accelerator.gather(labels)
        #
        #
        #     # 将收集到的预测和标签转化为实际的标签
        #     true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        #
        #
        #     all_predictions.extend(true_predictions)
        #     all_labels.extend(true_labels)
        # # find_lable(all_predictions, all_labels)
        # # 计算测试集的评估指标
        # results = metric.compute(predictions=all_predictions, references=all_labels)
        #
        # # 输出评估结果
        # print("Test Set Results:")
        # for key, value in results.items():
        #     if isinstance(value, (float, int)):
        #         print(f"{key}: {value:.4f}")
        #     else:
        #         print(f"{key}: {value}")


# res_test()

def expand_result():
    accelerator = Accelerator()
    model = AutoModelForTokenClassification.from_pretrained("new-model")
    model.eval()
    model.to(accelerator.device)

    test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=8)
    test_dataloader = accelerator.prepare(test_dataloader)

    all_predictions = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Evaluating on test set"):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)

        all_predictions.extend(true_predictions)
        all_labels.extend(true_labels)
    # test_title(all_predictions, all_labels,"B-")
    # test_title(all_predictions, all_labels,"I-")
    cal_entire(all_predictions, all_labels)


def test_title(predictions, references, title):
    tp_b_class = 0
    fp_b_class = 0
    fn_b_class = 0

    for pred_seq, ref_seq in zip(predictions, references):
        for pred, ref in zip(pred_seq, ref_seq):

            is_pred_b_class = pred.startswith(title)
            is_ref_b_class = ref.startswith(title)

            if is_pred_b_class and is_ref_b_class and pred == ref:
                tp_b_class += 1
            elif is_pred_b_class and not is_ref_b_class:
                fp_b_class += 1
            elif not is_pred_b_class and is_ref_b_class:
                fn_b_class += 1

    precision_b_class = tp_b_class / (tp_b_class + fp_b_class) if (tp_b_class + fp_b_class) > 0 else 0
    recall_b_class = tp_b_class / (tp_b_class + fn_b_class) if (tp_b_class + fn_b_class) > 0 else 0
    f1_score_b_class = 2 * precision_b_class * recall_b_class / (precision_b_class + recall_b_class) if (
                                                                                                                    precision_b_class + recall_b_class) > 0 else 0

    print(title + " Class Metrics:")
    print("Precision:", precision_b_class)
    print("Recall:", recall_b_class)
    print("F1 Score:", f1_score_b_class)
    return f1_score_b_class, tp_b_class, fp_b_class, fn_b_class


def cal_entire(predictions, references):
    tp, fp, fn = 0, 0, 0

    for pred_seq, ref_seq in zip(predictions, references):
        for pred, ref in zip(pred_seq, ref_seq):

            if ref != "O" and pred != "O":
                if pred == ref:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif pred != "O" and ref == "O":
                fp += 1
            elif pred == "O" and ref != "O":
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Overall Non-O Precision:", precision)
    print("Overall Non-O Recall:", recall)
    print("Overall Non-O F1 Score:", f1_score)
    return f1_score, tp, fp, fn


# expand_result()

def cal_label(predictions, references):
    tp, fp, fn = Counter(), Counter(), Counter()

    for pred_seq, ref_seq in zip(predictions, references):
        for pred, ref in zip(pred_seq, ref_seq):
            if ref != "O" and pred != "O":
                if pred == ref:
                    tp[ref] += 1  # True Positive
                else:
                    fp[pred] += 1  # False Positive：
                    fn[ref] += 1  # False Negative
            elif ref != "O" and pred == "O":
                fn[ref] += 1  # False Negative
            elif ref == "O" and pred != "O":
                fp[pred] += 1  # False Positive

    return tp, fp, fn


def compute_metrics_for_each_entity(predictions, references):
    tp, fp, fn = Counter(), Counter(), Counter()

    for pred_seq, ref_seq in zip(predictions, references):
        for pred, ref in zip(pred_seq, ref_seq):
            if ref != "O" and pred != "O":
                entity_ref = ref.split("-")[1]
                entity_pred = pred.split("-")[1]
                if entity_pred == entity_ref:
                    tp[entity_ref] += 1  # True Positive
                else:
                    fp[entity_pred] += 1  # False Positive
                    fn[entity_ref] += 1  # False Negative
            elif ref != "O" and pred == "O":
                entity_ref = ref.split("-")[1]
                fn[entity_ref] += 1  # False Negative
            elif ref == "O" and pred != "O":
                entity_pred = pred.split("-")[1]
                fp[entity_pred] += 1  # False Positive
    return tp, fp, fn


def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def calculate_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def cal_average_f1(path):
    accelerator = Accelerator()
    model = AutoModelForTokenClassification.from_pretrained(path)
    model.eval()
    model.to(accelerator.device)

    test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=8)
    test_dataloader = accelerator.prepare(test_dataloader)

    all_predictions = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Evaluating on test set"):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)

        all_predictions.extend(true_predictions)
        all_labels.extend(true_labels)
    label_tp, label_fp, label_fn = cal_label(all_predictions, all_labels)
    print(label_tp, label_fp, label_fn)
    metrics_per_label = {}
    sum_f1 = 0
    # for label in tp.keys() | fp.keys() | fn.keys():
    #     precision, recall, f1 = calculate_precision_recall_f1(tp[label], fp[label], fn[label])
    #     # metrics_per_label[label] = {"Precision": precision, "Recall": recall, "F1 Score": f1}
    #     sum_f1=sum_f1+f1
    # for label, metrics in metrics_per_label.items():
    #     print(f"Metrics for {label}:")
    #     print(f"  Precision: {metrics['Precision']:.2f}")
    #     print(f"  Recall: {metrics['Recall']:.2f}")
    #     print(f"  F1 Score: {metrics['F1 Score']:.2f}")
    tp, fp, fn = compute_metrics_for_each_entity(all_predictions, all_labels)
    tp = dict(tp)
    fp = dict(fp)
    fn = dict(fn)
    #
    # F1_B,tp_b,fp_b,fn_b= test_title(all_predictions, all_labels,"B-")
    # F1_I,tp_I,fp_I,fn_I = test_title(all_predictions, all_labels,"I-")
    F1_Full, tp_f, fp_f, fn_f = cal_entire(all_predictions, all_labels)
    # tp.update([("B-", tp_b),("I-",tp_I),("Full",tp_f)])
    # fp.update([("B-", fp_b), ("I-", fp_I), ("Full", fp_f)])
    # fn.update([("B-", fn_b), ("I-", fn_I), ("Full", fn_f)])
    total_tp = sum(tp.values()) + sum(label_fp.values())
    total_fp = sum(fp.values()) + sum(label_fn.values())
    total_fn = sum(fn.values()) + sum(label_fn.values())
    # micro_f1 = calculate_f1(total_tp, total_fp, total_fn)
    # calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    print(calculate_precision_recall_f1(total_tp, total_fp, total_fn))
    # results = metric.compute(predictions=all_predictions, references=all_labels)
    # for label,score in results.items():
    #     if label.startswith("ART"):
    #         F1_ART= score['f1']
    #     if label.startswith("CON"):
    #         F1_CON= score['f1']
    #     if label.startswith("LOC"):
    #         F1_LOC= score['f1']
    #     if label.startswith("MAT"):
    #         F1_MAT= score['f1']
    #     if label.startswith("PER"):
    #         F1_PER= score['f1']
    #     if label.startswith("SPE"):
    #         F1_SPE= score['f1']
    # macro_average=(sum_f1  + F1_MAT + F1_SPE + F1_PER + F1_LOC+ F1_CON +F1_ART) /17
    # print(macro_average)


cal_average_f1(output_dir)

print("-----------------")
cal_average_f1("new-model")





