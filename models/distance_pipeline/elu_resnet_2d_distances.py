# Import libraries
import numpy as np
import keras
import keras.backend as K
from keras.models import Model
# Activation and Regularization
from keras.regularizers import l2
from keras.activations import softmax
# Keras layers
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from keras.layers.pooling import MaxPooling2D, AveragePooling2D


def softMaxAxis2(x):
    """ Compute softmax on axis 2. """
    return softmax(x,axis=2)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def resnet_block(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='elu',
                 batch_normalization=True,
                 conv_first=False):
    """ # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    """

    x = inputs
    # down size
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(num_filters//2, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    # Strided Calculations - cyclic strides - half filters
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(num_filters//2, kernel_size=3, strides=strides, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    # Up size
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(num_filters, kernel_size=1, strides=strides, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    return x


def resnet_layer(inputs,
                 num_filters=32,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=False):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=4):
    """ Elu ResNet Model builder.
        * depth should be multiple of 4
    """
    if depth % 4 != 0:
        raise ValueError('depth should be 4n (eg 8 or 16)')
    # Start model definition.
    num_filters_in = 64
    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    striding = [1,2,4,8]
    for stage in range(depth):

        activation = 'elu'
        batch_normalization = True

        # bottleneck residual unit
        y = resnet_block(inputs=x,
                         num_filters=64,
                         kernel_size=3,
                         strides=striding[stage%4])

        x = keras.layers.add([x, y])

    # Add a linear Conv classifier on top.
    # v2 has BN-ReLU before Pooling
    y = Conv2D(num_classes, kernel_size=1, strides=1, padding='same',
    	       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    outputs = Activation(softMaxAxis2)(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Check it's working
if __name__ == "__main__":
    # Using AMSGrad optimizer for speed 
    kernel_size, filters = 3, 16
    adam = keras.optimizers.Adam(amsgrad=True)
    # Create model
    model = model = resnet_v2(input_shape=(200,200, 40), depth=8, num_classes=7)
    model.compile(optimizer=adam, loss=weighted_categorical_crossentropy(
                  np.array([0.01,1,0.9,0.8,0.7,0.6,0.5])), metrics=["accuracy"])
    model.summary()
    print("Model file works perfectly")
