import mxnet as mx
from mxnet import nd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mxnet import autograd
import random

#make dataset
num_inputs = 2
num_examples = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(features[0], labels[0])


def use_svg_display():
    # save in vector graphics
    plt.rcParams['savefig.format'] = 'svg'

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted
    plt.rcParams['figure.figsize'] = figsize

def save_plot(filename, format='svg'):
    plt.savefig(filename, format=format)
    plt.close()

set_figsize()
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
save_plot('scatter_plot.png')

# 임의로 선택된 특성(feature)들과 태그(tag)들을 배치 크기(batch size)의 개수만큼 리턴
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

# 작은 배치를 읽어서 출력
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

#모델 파라미터들 초기화
#가중치는 평균값이 0이고 표준편차가 0.01인 정규분포를 따르는 난수값들로 초기화합니다. 편향(bias)은 0으로 설정
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
#손실(loss) 값을 줄이는 방향으로 각 파라미터를 업데이트
w.attach_grad()
b.attach_grad()

#모델 정의
def linreg(X, w, b):
    return nd.dot(X, w) + b

#손실함수 정의
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#최적화 알고리즘 정의
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

#학습
lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))