import tensorflow as tf
import numpy as np


input_shape = (256, 256, 3)

class Generator(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()

        self._set_up_layers(**kwargs)
    
    def _set_up_layers(self, **kwargs):
        self.input_dense = tf.keras.layers.Dense(
            units=np.prod(kwargs["origin_output_shape"])
        )
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=kwargs["filters"][i],
                kernel_size=kwargs["kernel_size"][i],
                strides=kwargs["strides"][i],
                padding="same",
                name=f"generator_conv_{i}"
            ) for i in range(kwargs["num_of_conv"])
        ]
        self.batch_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(kwargs["num_of_conv"])
        ]
        
        self.reshape = tf.keras.layers.Reshape(kwargs["origin_output_shape"])

    def call(self, input):
        # We now use images as input then output images
        # therefore, we do not need dense layers and the reshape layer
        x = input
        #x = self.input_dense(x)
        #x = tf.keras.activations.relu(x)

        #x = self.reshape(x)
        for i in range(len(self.conv_layers)-1):
            #x = tf.keras.layers.UpSampling2D()(x)
            x = self.conv_layers[i](x)
            x = tf.keras.activations.relu(x)
            x = self.batch_norm[i](x)
        
        x = self.conv_layers[-1](x)
        x = tf.keras.activations.sigmoid(x)

        return x * 255


class Discriminator(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        self._set_up_layers(**kwargs)
    
    def _set_up_layers(self, **kwargs):
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=kwargs["filters"][i],
                kernel_size=kwargs["kernel_size"][i],
                strides=kwargs["strides"][i],
                padding="same",
                name=f"discriminator_conv_{i}",
            ) for i in range(kwargs["num_of_conv"])
        ]
        self.dense_layers = [
            tf.keras.layers.Dense(
                units=kwargs["dense_units"][i],
                activation="relu",
                name=f"discriminator_{i}"
            ) for i in range(kwargs["num_of_dense"])
        ]
        self.output_layer = tf.keras.layers.Dense(units=1, activation="sigmoid")
        
    def call(self, input):
        x = input / 255
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = tf.keras.activations.relu(x)
        
        x = tf.keras.layers.Flatten()(x)

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
        
        return self.output_layer(x)


class model(tf.keras.models.Model):
    pass


if __name__ == "__main__":
    generator = Generator(
        origin_output_shape=(32, 32, 3),
        num_of_conv=5,
        filters=[32, 64, 128, 64, 3],
        kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        strides=[1, 1, 1, 1, 1],
    )

    discriminator = Discriminator(
        num_of_dense=3,
        dense_units=[64, 64, 32],
        num_of_conv=4,
        filters=[64, 128, 128, 64],
        kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
        strides=[2, 2, 2, 1]
    )