import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate

# 환경 변수 설정 (필요시)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 또는 "true"

# 체크포인트와 데이터셋 설정
checkpoint = "bert-base-uncased"
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 데이터 전처리 함수
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 데이터 전처리
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False
)

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 평가 메트릭 함수
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 모델 훈련
trainer.train()
