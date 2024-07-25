import os
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

# 모델과 토크나이저 로드
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 데이터셋 로드
raw_datasets = load_dataset("glue", "mrpc")

# 데이터셋 전처리 함수
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 데이터셋 전처리
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# DataCollatorWithPadding 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 샘플 가져오기 및 패딩 테스트
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# 동적 패딩 확인
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

# 배치 데이터 설정
batch["labels"] = torch.tensor([1] * len(samples["input_ids"]))

# 옵티마이저 설정
optimizer = AdamW(model.parameters())

# 모델 학습
model.train()
loss = model(**batch).loss
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
