import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *
"""
1. Package load
torch: PyTorch 패키지를 불러옵니다.
torchvision: PyTorch 에서 이미지 데이터 로드와 관련된 여러가지 편리한 함수들을 제공합니다.
matplotlib.pyplot: 데이터 시각화를 위해 사용합니다.
numpy: Scientific computing과 관련된 여러 편리한 기능들을 제공해주는 라이브러리입니다.
"""
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#import check_util.checker as checker 
#%matplotlib inline

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

"""
2. 하이퍼파라미터 세팅
학습에 필요한 하이퍼파리미터의 값을 초기화해줍니다. 
하이퍼파라미터는 뉴럴네트워크를 통하여 학습되는 것이 아니라 학습율(learning rate), 
    사용할 레이어의 수 등 설계자가 결정해줘야 하는 값들을 의미합니다.

미니배치의 크기(batch_size), 학습 할 세대(epoch) 수(num_epochs), 
    학습률(learning_rate) 등의 값들을 다음과 같이 정했습니다.
"""
batch_size = 100
num_epochs = 5
learning_rate = 0.001

"""
3. Dataset 및 DataLoader 할당
"""
from torch.utils.data import DataLoader

root = './data'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = dset.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_data = dset.FashionMNIST(root=root, train=False, transform=transform, download=True)
## 코드 시작 ##
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False)
## 코드 종료 ##

"""
4. 데이터 샘플 시각화
"""
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
columns = 5
rows = 5
fig = plt.figure(figsize=(8,8))

for i in range(1, columns*rows+1):
    data_idx = np.random.randint(len(train_data))
    img = train_data[data_idx][0][0,:,:].numpy() # numpy()를 통해 torch Tensor를 numpy array로 변환
    label = labels_map[train_data[data_idx][1]] # item()을 통해 torch Tensor를 숫자로 변환
    
    fig.add_subplot(rows, columns, i)
    plt.title(label)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show()


"""
5. 네트워크 설계
Multi Layer Perceptron(MLP) 레이어를 2개 쌓아 네트워크를 설계할 것입니다.
MLP는 한 레이어의 모든 뉴런이 다음 레이어의 뉴런과 완전히 연결된 계층(Fully connected layer)입니다.

한편, MLP의 레이어를 깊게 쌓을 때에는 반드시 비선형 activation function이 필요합니다.
이번 실습에서는 ReLU를 사용 할 것입니다.

첫번째 fully connected layer(FC layer)의 입력 feature 갯수는 입력 이미지의 픽셀 갯수인 28x28로, 
    출력 feature 갯수는 512로 하겠습니다.
두번째 FC layer의 출력 feature 갯수는 데이터의 class 갯수인 10으로 지정해야 합니다.
그리고 첫번째 FC layer 이후에는 ReLU 함수를 적용해보세요.
두번째 FC layer 이후에는 ReLU activation function을 적용하지 않습니다. 
Classification 네트워크의 마지막 activation function은 주로 softmax 함수가 적용되기 때문입니다. 
그렇다면 왜 softmax function은 네트워크 구현에 포함시키지 않는걸까요? 
그 이유는 우리가 이후에 선언할 크로스엔트로피(Cross Entropy) loss function에 
    softmax function이 포함되도록 Pytorch에 구현이 되어있기 때문입니다. 따라서 softmax function은 우리가 따로 선언하지 않아도 됩니다.
첫번째 FC Layer와 ReLU 사이에 Batch normalization('Lab-09-4')을 적용해보세요.
"""
class DNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            ## 코드 시작 ##
            nn.Linear(784,512,bias=True).to(device),    # Linear_1 해당하는 층
            nn.BatchNorm1d(512),    # BatchNorm_1 해당하는 층
            nn.ReLU()     # ReLU_1 해당하는 층
            ## 코드 종료 ##
        )
        self.layer2 = nn.Sequential(
            ## 코드 시작 ##
            nn.Linear(512,10,bias=True).to(device)    # Linear_2 해당하는 층 
            ## 코드 종료 ##
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        return x_out



"""
6. Weight initialization
이번 실습에서 우리는 네트워크의 weight를 xavier_normal으로 초기화할 것입니다. 
nn.init 모듈에는 다양한 초기화 기법들이 정의되어 있습니다. 
가중치 초기화(Weight initialization)에 대해 잘 기억이 나지 않는다면 
    'Lab-09-2'강의를 참고하시기 바랍니다.
"""
def weights_init(m):
    if isinstance(m, nn.Linear): # 모델의 모든 MLP 레이어에 대해서
        nn.init.xavier_normal_(m.weight) # Weight를 xavier_normal로 초기화
        print(m.weight)


"""
7. 모델 생성
model.apply()을 통해 가중치 초기화를 적용할 수 있습니다. 
apply 함수의 인자로 <6. Weight initialization> 에서 정의한 weights_init 함수를 주면 됩니다.
"""
torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정
model = DNN().to(device)
model.apply(weights_init) # 모델에 weight_init 함수를 적용하여 weight를 초기화


"""
8. Loss function 및 Optimizer 정의
생성한 모델을 학습 시키기 위해서 손실함수를 정의해야 합니다.
효과적인 경사하강 방법을 적용하기 위해 옵티마이져를 함께 사용할 겁니다.

criterion 변수에 Classification에서 자주 사용되는 Cross Entropy Loss를 정의하세요. 
Cross Entropy에 대해 잘 기억이 나지 않는다면 'Lab-06' 강의를 참고하시기 바랍니다.

어떤 optimizer를 사용할 것인지는 설계자의 몫입니다. 
Adam optimizer는 많은 경우에 잘 작동하는 훌륭한 optimizer입니다. 
optimizer" 변수에 Adam optimizer를 정의하세요. 
<2. 하이퍼파라미터 세팅> 에서 정의한 learning_rate 를 사용하세요.
"""
## 코드 시작 ##
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
## 코드 종료 ##



"""
9. Training

데이터를 공급해주는 train_loader가 for 문을 통해 미니 배치만큼 데이터를 가져오면, 
    <7. 모델 생성> 에서 작성한 model에게 전달하고, 
    출력 값을 <8. Loss function 및 Optimizer 정의> 에서 작성한 손실함수 criterion 를 통해 
    손실 값을 얻습니다. 
해당 손실 값을 기준으로 모델은 손실값이 적어지는 방향으로 매개변수(parameters)를 업데이트합니다. 
업데이트를 수행하는 것은 optimizer 객체입니다.

1. 모델에 imgs 데이터를 주고, 그 출력을 outputs 변수에 저장하세요.
2. 모델의 outputs과 train_loader에서 제공된 labels를 통해 손실값을 구하고, 
    그 결과를 loss 변수에 저장하세요.
3. 이전에 계산된 gradient를 모두 clear 해줍니다.
4. Gradient를 계산합니다.
5. Optimizer를 통해 파라미터를 업데이트합니다.
"""
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        ## 코드 시작 ##
        outputs = model(imgs)  # 위의 설명 1. 을 참고하여 None을 채우세요.
        loss = criterion(outputs,labels)     # 위의 설명 2. 를 참고하여 None을 채우세요.
        
        optimizer.zero_grad()            # Clear gradients: 위의 설명 3. 을 참고하여 None을 채우세요.
        loss.backward()            # Gradients 계산: 위의 설명 4. 를 참고하여 None을 채우세요.
        optimizer.step()            # Parameters 업데이트: 위의 설명 5. 를 참고하여 None을 채우세요.
        ## 코드 종료 ##
        
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax).float().mean()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                epoch+1, num_epochs, i+1, len(train_loader), loss.item(), accuracy.item() * 100))


"""
10. Test
마지막으로 학습된 모델의 성능을 테스트할 차례입니다.

model.eval()은 모델을 평가(evaluation) 모드로 설정하겠다는 의미입니다. 
평가 모드 가 필요한 이유는, batch normalization과 dropout이 training을 할 때와 
    test를 할 때 작동하는 방식이 다르기 때문입니다. 
평가 모드를 설정해주어야 test를 할 때 일관된 결과를 얻을 수 있습니다.

torch.no_grad()는 torch.Tensor의 requires_grad를 False로 만들어줍니다. 
Test 때는 backpropagation을 통해 gradient를 계산할 필요가 없기 때문에, 
    Tensor의 requires_grad를 False로 바꿔줌을 통해 메모리를 낭비하지 않을 수 있습니다.
Test를 마친 이후에 training을 더 진행하길 원하면 
    model.train()을 통해 다시 training 모드로 설정을 해주면 됩니다.
"""
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, argmax = torch.max(outputs, 1) # max()를 통해 최종 출력이 가장 높은 class 선택
        total += imgs.size(0)
        correct += (labels == argmax).sum().item()
    
    print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))






























