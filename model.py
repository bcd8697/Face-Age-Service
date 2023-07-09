import tensorflow as tf


class FaceBlock(tf.keras.layers.Layer):
    '''
    Special block which is going to be repeated a few times in the model's architecture.
    Consists of convolution, batch norm, max pooling and relu activation.
    '''
    def __init__(self, conv_num_channels):
        super(FaceBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(conv_num_channels, (3, 3), padding="same", activation='relu')
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class FaceModel(tf.keras.Model):
    '''
    Model class implementation
    '''
    def __init__(self):
        super(FaceModel, self).__init__()
        self.fb1 = FaceBlock(conv_num_channels=16)
        self.fb2 = FaceBlock(conv_num_channels=32)
        self.fb3 = FaceBlock(conv_num_channels=32)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.fb1(inputs)
        x = self.fb2(x)
        x = self.fb3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x