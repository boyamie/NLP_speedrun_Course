import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터셋 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 평균 0.5, 표준편차 0.5로 정규화
])

# FashionMNIST 데이터셋 불러오기
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 설정
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 데이터셋이 올바르게 로드되었는지 확인
print(f'훈련 데이터셋 크기: {len(train_loader.dataset)}')
print(f'테스트 데이터셋 크기: {len(test_loader.dataset)}')

# 첫 번째 배치 불러오기 예시
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(f'첫 번째 배치 크기: {example_data.shape}')
print(f'첫 번째 배치 레이블: {example_targets}')
