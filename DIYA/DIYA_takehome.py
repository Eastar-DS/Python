import numpy as np
import tensorflow as tf
tf.random.set_seed(0)
print(tf.__version__)

x = [[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]]
y = [[0],
    [1],
    [1],
    [0]]

dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(len(x))

[i for i in dataset]

def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels

preprocess_data(x,y)

W1 = tf.Variable(tf.random.normal((2, 10)), name='weight1')
b1 = tf.Variable(tf.random.normal((1,)), name='bias1')

W2 = tf.Variable(tf.random.normal((10, 10)), name='weight2')
b2 = tf.Variable(tf.random.normal((1,)), name='bias2')

W3 = tf.Variable(tf.random.normal((10, 10)), name='weight3')
b3 = tf.Variable(tf.random.normal((1,)), name='bias3')

W4 = tf.Variable(tf.random.normal((10, 1)), name='weight4')
b4 = tf.Variable(tf.random.normal((1,)), name='bias4')


def deep_nn(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
    return hypothesis


#오차율 함수
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost
#정확도 함수
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy
#경사하강법 함수
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(deep_nn(features),features,labels)
    return tape.gradient(loss_value, [W1, W2, W3, W4, b1, b2, b3, b4])



optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


EPOCHS = 5000

for step in range(EPOCHS+1):
    for features, labels  in dataset:
        features, labels = preprocess_data(features, labels)
        grads = grad(deep_nn(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W1, W2, W3,W4, b1, b2, b3, b4]))
        if step % 500 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(deep_nn(features),features,labels)))


x_data, y_data = preprocess_data(x, y)
test_acc = accuracy_fn(deep_nn(x_data),y_data)
print("Testset Accuracy: {:.4f}".format(test_acc))





























