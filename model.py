import tensorflow as tf
from tensorflow.keras import layers, Model
from config import LATENT_DIM

# Encoder
def build_encoder():
    encoder_inputs = tf.keras.Input(shape=(64, 64, 3), name="encoder_inputs")
    x = layers.Conv2D(16, 2, strides=2, activation='relu')(encoder_inputs)
    x = layers.Conv2D(32, 2, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 2, strides=2, activation='relu')(x)
    x = layers.Conv2D(128, 2, strides=2, activation='relu')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

    return Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var], name="encoder")

# Decoder
def build_decoder():
    decoder_inputs = tf.keras.Input(shape=(LATENT_DIM,), name="decoder_inputs")
    x = layers.Dense(4 * 4 * 128, activation='relu')(decoder_inputs)
    x = layers.Reshape((4, 4, 128))(x)
    x = layers.Conv2DTranspose(64, 2, strides=2, activation='relu')(x)
    x = layers.Conv2DTranspose(32, 2, strides=2, activation='relu')(x)
    x = layers.Conv2DTranspose(16, 2, strides=2, activation='relu')(x)
    decoder_outputs = layers.Conv2DTranspose(3, 2, strides=2, activation='sigmoid')(x)

    return Model(inputs=decoder_inputs, outputs=decoder_outputs, name="decoder")

# VAE Model
class VAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        self.add_loss(self.beta * kl_loss)
        return reconstruction
