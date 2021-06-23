# -*- coding: utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2
Conv2D = tf.keras.layers.Conv2D
TransConv2D = tf.keras.layers.Conv2DTranspose
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
Maxpool2D = tf.keras.layers.MaxPool2D
ZeroPadd2D = tf.keras.layers.ZeroPadding2D
ReLU = tf.keras.layers.ReLU
LeakReLU = tf.keras.layers.LeakyReLU

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def residual_block(input, ori_img, weight_decay, c):

    h = ZeroPadd2D((1,1))(input)
    h = DepthwiseConv2D(kernel_size=3, strides=1, padding="valid", use_bias=False, depthwise_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = ReLU()(h)

    h = ZeroPadd2D((1,1))(h)
    h = DepthwiseConv2D(kernel_size=3, strides=1, padding="valid", use_bias=False, depthwise_regularizer=l2(weight_decay))(h)

    if c == 0:
        h = InstanceNormalization()(h + input)
        h = ReLU()(h)
        return h
    else:
        h = InstanceNormalization()(h + ori_img)

    return ReLU()(h) + input

Trans_1 = TransConv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(0.00001))
Trans_2 = TransConv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(0.00001))
Trans_3 = TransConv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(0.00001))
Trans_4 = TransConv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(0.00001))
Trans_5 = TransConv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(0.00001))

def new_generator(input_shape=(416, 416, 3), weight_decay=0.00001):

    h = inputs = tf.keras.Input(input_shape)

    h = ZeroPadd2D((3,3))(h)
    h = Conv2D(filters=64, kernel_size=7, strides=1, padding="valid", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = ReLU()(h)   # 52, 26, 13 로 grid를 설정하자(detection의 grid와는 별개)

    h = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = ReLU()(h)   # 208 x 208 x 128

    h = Conv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = ReLU()(h)   # 104 x 104 x 256

    h_1 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h_2 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h_3 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)

    h_1 = Conv2D(filters=512, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h_1)
    h_1 = InstanceNormalization()(h_1)  # 52 x 52 x 512

    h_2 = Conv2D(filters=512, kernel_size=3, strides=4, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h_2)
    h_2 = InstanceNormalization()(h_2)  # 26 x 26 x 512

    h_3 = Conv2D(filters=512, kernel_size=3, strides=8, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h_3)
    h_3 = InstanceNormalization()(h_3)  # 13 x 13 x 512

    h_1_ori = h_1
    for i in range(3):
        if i == 0:
            h_1 = residual_block(h_1, h_1_ori, weight_decay, i)
        else:
            h_1 = residual_block(h_1 * 0.5, h_1_ori * 0.5, weight_decay, i)

    h_2_ori = h_2
    for i in range(3):
        if i == 0:
            h_2 = residual_block(h_2, h_2_ori, weight_decay, i)
        else:
            h_2 = residual_block(h_2 * 0.5, h_2_ori * 0.5, weight_decay, i)

    h_3_ori = h_3
    for i in range(3):
        if i == 0:
            h_3 = residual_block(h_3, h_3_ori, weight_decay, i)
        else:
            h_3 = residual_block(h_3 * 0.5, h_3_ori * 0.5, weight_decay, i)

    ##############################################################################################################
    
    h_1 = Trans_1(h_1)
    h_1 = InstanceNormalization()(h_1)
    h_1 = ReLU()(h_1)   # 104 x 104 x 256
    h_1 = Trans_2(h_1)
    h_1 = InstanceNormalization()(h_1)
    h_1 = ReLU()(h_1)   # 208 x 208 x 128
    h_1 = Trans_3(h_1)
    h_1 = InstanceNormalization()(h_1)  # 416 x 416 x 64

    h_2 = Trans_4(h_2)
    h_2 = InstanceNormalization()(h_2)
    h_2 = ReLU()(h_2)   # 52 x 52 x 256
    h_2 = Trans_5(h_2)
    h_2 = InstanceNormalization()(h_2)
    h_2 = ReLU()(h_2)   # 104 x 104 x 256
    h_2 = Trans_2(h_2)
    h_2 = InstanceNormalization()(h_2)
    h_2 = ReLU()(h_2)   # 208 x 208 x 128
    h_2 = Trans_3(h_2)
    h_2 = InstanceNormalization()(h_2)  # 416 x 416 x 64

    h_3 = TransConv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h_3)
    h_3 = InstanceNormalization()(h_3)
    h_3 = ReLU()(h_3)   # 26 x 26 x 256
    h_3 = TransConv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h_3)
    h_3 = InstanceNormalization()(h_3)
    h_3 = ReLU()(h_3)   # 52 x 52 x 256
    h_3 = Trans_5(h_3)
    h_3 = InstanceNormalization()(h_3)
    h_3 = ReLU()(h_3)   # 104 x 104 x 256
    h_3 = Trans_2(h_3)
    h_3 = InstanceNormalization()(h_3)
    h_3 = ReLU()(h_3)   # 208 x 208 x 128
    h_3 = Trans_3(h_3)
    h_3 = InstanceNormalization()(h_3)  # 416 x 416 x 64

    h = InstanceNormalization()(h_1 + h_2 + h_3)
    h = ReLU()(h)
    
    h = ZeroPadd2D((3,3))(h)
    h = Conv2D(filters=3, kernel_size=7, strides=1, padding="valid", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def new_discriminator(input_shape=(416, 416, 3), weight_decay=0.00001):

    dim = 64
    dim_ = dim
    h = inputs = tf.keras.Input(input_shape)
    inputs2 = tf.keras.Input(input_shape)

    h = tf.keras.layers.Concatenate()([h, inputs2])

    h = Conv2D(filters=dim, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = LeakReLU(0.2)(h)

    for _ in range(2):
        dim = min(dim * 2, dim_ * 8)
        h = Conv2D(filters=dim, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = LeakReLU(0.2)(h)

    dim = min(dim * 2, dim_ * 8)
    h = Conv2D(filters=dim, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = LeakReLU(0.2)(h)

    h = Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)

    return tf.keras.Model(inputs=[inputs, inputs2], outputs=h)
