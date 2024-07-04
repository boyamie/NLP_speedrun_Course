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