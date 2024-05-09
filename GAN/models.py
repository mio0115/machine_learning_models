import tensorflow as tf


class Generator(tf.keras.layers.Layer):
    pass


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        self.get_model(kwargs)
    
    def get_model(self, **kwargs):
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
                units=kwargs["dense_units"],
                activation="relu",
                name=f"discriminator_{i}"
            ) for i in range(kwargs["num_of_dense"])
        ]
        self.output_layer = tf.keras.layers.Dense(units=2, activation="softmax")
        
    def call(self, input):
        x = input
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = tf.keras.activations.relu(x)
        
        x = tf.keras.layers.Flatten()(x)

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
        
        return self.output_layer(x)


class model(tf.keras.models.Model):
    pass