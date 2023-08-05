"""
Autor:  JacintoDR/vae_oversampler 
        [https://github.com/JacintoDR/vae_oversampler]
forked from dyanni3/vae_oversampler 
        [https://github.com/dyanni3/vae_oversampler]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from sklearn.preprocessing import StandardScaler as SS
import numpy as np


class VAEOversampler:
    """
    Class for variational autoencoder oversampling of a dataset
    """

    def __init__(self, epochs=10, intermediate_dim=12,
                 weights=None, batch_size=64, original_dim=None,
                 latent_dim=1, minority_class_id=1,
                 rescale=False, verbose=True,
                 random_state=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.weights = weights
        self.latent_dim = latent_dim
        self.minority_class_id = minority_class_id
        self.rescale = rescale
        self.verbose = verbose
        self.random_state = random_state
        if random_state is not None:
            try:
                np.random.seed(random_state)
            except TypeError:
                print("random state must be int")
                print("setting random state None")
                np.random.seed(None)
                self.random_state = None

    def get_weights(self):
        return self.vae.get_weights()

    def set_weights(self, parameters):
        self.vae.set_weights(parameters)

    def save_weights(self, weights):
        if weights:
            self.vae.save_weights(weights)

    def sampling(self, args):
        """Reparameterization trick by sampling for an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build(self, *args, **kwargs):
        """
        Builds a variational autoencoder using Keras
            and then compiles

        #Returns
            none.
            sets self.vae, self.encoder, and self.decoder
        """
        if self.original_dim is None:
            print("especify original_dim parameter")
        else:
            original_dim = self.original_dim
            input_shape = (original_dim, )
            inputs = Input(shape=input_shape, name='encoder_input')
            x = Dense(self.intermediate_dim, activation='relu')(inputs)
            z_mean = Dense(self.latent_dim, name='z_mean')(x)
            z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
            z = Lambda(self.sampling,
                       output_shape=(self.latent_dim, ),
                       name='z')([z_mean, z_log_var])
            encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            if self.verbose:
                encoder.summary()
            latent_inputs = Input(shape=(self.latent_dim, ), name='z_sampling')
            x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
            outputs = Dense(original_dim, activation='sigmoid')(x)
            decoder = Model(latent_inputs, outputs, name='decoder')
            if self.verbose:
                decoder.summary()
            outputs = decoder(encoder(inputs)[2])
            self.vae = Model(inputs, outputs, name='vae_mlp')
            reconstruction_loss = mse(inputs, outputs)
            reconstruction_loss *= original_dim
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self.vae.add_loss(vae_loss)
            self.vae.compile(optimizer='adam')
            if self.verbose:
                self.vae.summary()
            if self.weights:
                self.vae.load_weights(self.weights)

            self.encoder = encoder
            self.decoder = decoder
        return

    def build_train(self, x_train, x_test=None,
                    *args, **kwargs):
        """
        Builds a variational autoencoder using Keras
            and then compiles and trains it

        #Arguments
            x_train: array like, shape == (num_samples, num_features)
            x_test: optional validation data

        #Returns
            none.
            sets self.vae, self.encoder, and self.decoder
        """
        original_dim = x_train.shape[1]
        input_shape = (original_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling,
                   output_shape=(self.latent_dim, ),
                   name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if self.verbose:
            encoder.summary()
        latent_inputs = Input(shape=(self.latent_dim, ), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        if self.verbose:
            decoder.summary()
        outputs = decoder(encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        if self.verbose:
            self.vae.summary()
        if self.weights:
            self.vae.load_weights(self.weights)
        else:
            if x_test is not None:
                hist = self.vae.fit(x_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             validation_data=(x_test, None),
                             verbose=self.verbose)
            else:
                hist = self.vae.fit(x_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             verbose=self.verbose)
            #self.vae.save_weights('vae.h5')
        self.encoder = encoder
        self.decoder = decoder

        return hist

    def fit(self, Xtrain, ytrain, validation_data=None, **vae_kwargs):
        """
        Fits a vae oversampler

        Arguments:
            Xtrain: training data
            ytrain: training labels
            validation_data = (Xtest,ytest)
                optional
            variational autoencoder kwargs: passed to keras

        Returns: none
        """
        if validation_data is not None:
            Xtest, ytest = validation_data
        if self.rescale:
            self.ss = SS()
            self.ss.fit(Xtrain[ytrain == self.minority_class_id])
            X = self.ss.transform(Xtrain[ytrain == self.minority_class_id])
            if validation_data is not None:
                x_test = self.ss.transform(Xtest
                                           [ytest == self.minority_class_id])
        else:
            X = Xtrain[ytrain == self.minority_class_id]
            if validation_data is not None:
                x_test = Xtest[ytest == self.minority_class_id]
        if validation_data is not None:
            hist = self.build_train(X, x_test=x_test, **vae_kwargs)
        else:
            hist = self.build_train(X, **vae_kwargs)

        return hist

    def fit_resample(self, Xtrain, ytrain, validation_data=None, sampling_strategy=1, **vae_kwargs):
        """
        Fits a vae oversampler and returns resampled dataset

        Arguments:
            Xtrain: training data
            ytrain: training labels
            validation_data = (Xtest,ytest)
                optional
            sampling_strategy: (N_M - N_m) * sampling_strategy
            	N_M: number of instances of majority class
            	N_m: number of instances of minority class
            variational autoencoder kwargs: passed to keras

        Returns:
            Xres,yres: resampled data and labels.
            attempts to balance the dataset to 50% minority class
        """
        if validation_data is not None:
            Xtest, ytest = validation_data
        num_samples_to_generate = max(int((Xtrain[ytrain != self.minority_class_id].shape[0]
                                      - Xtrain[ytrain == self.minority_class_id].shape[0])
                                      * sampling_strategy), 
                                      100)
        if self.rescale:
            self.ss = SS()
            self.ss.fit(Xtrain[ytrain == self.minority_class_id])
            X = self.ss.transform(Xtrain[ytrain == self.minority_class_id])
            if validation_data is not None:
                x_test = self.ss.transform(
                                           Xtest
                                           [ytest == self.minority_class_id])
        else:
            X = Xtrain[ytrain == self.minority_class_id]
            if validation_data is not None:
                x_test = Xtest[ytest == self.minority_class_id]
        if validation_data is not None:
            self.build_train(X, x_test=x_test, **vae_kwargs)
        else:
            self.build_train(X, **vae_kwargs)
        z_sample = np.random.normal(0, 1,
                                    (num_samples_to_generate,
                                     self.latent_dim))
        outputs = self.decoder.predict(z_sample)
        if self.rescale:
            oversampled_X = self.ss.inverse_transform(outputs)
        else:
            oversampled_X = outputs
        oversampled_y = np.ones(num_samples_to_generate)\
            * self.minority_class_id
        X_all = np.concatenate((Xtrain, oversampled_X))
        y_all = np.concatenate((ytrain, oversampled_y))
        return(X_all, y_all)

    def resample(self, Xtrain, ytrain, sampling_strategy=1, **vae_kwargs):
        """
        Generates samples from vae model and returns resampled dataset

        Arguments:
            Xtrain: training data
            ytrain: training labels
            sampling_strategy: (N_M - N_m) * sampling_strategy
            	N_M: number of instances of majority class
            	N_m: number of instances of minority class
            variational autoencoder kwargs: passed to keras

        Returns:
            Xres,yres: resampled data and labels.
            attempts to balance the dataset to 50% minority class
        """
        num_samples_to_generate = max(int((Xtrain[ytrain != self.minority_class_id].shape[0]
                                      - Xtrain[ytrain == self.minority_class_id].shape[0])
                                      * sampling_strategy), 
                                      100)

        z_sample = np.random.normal(0, 1,
                                    (num_samples_to_generate,
                                     self.latent_dim))
        outputs = self.decoder.predict(z_sample)
        if self.rescale:
            oversampled_X = self.ss.inverse_transform(outputs)
        else:
            oversampled_X = outputs
        oversampled_y = np.ones(num_samples_to_generate)\
            * self.minority_class_id
        X_all = np.concatenate((Xtrain, oversampled_X))
        y_all = np.concatenate((ytrain, oversampled_y))
        return(X_all, y_all)


    def evaluate(self, Xtest=None, ytest=None, batch_size=1, **kwargs):
        """
        Evaluates a vae oversampler

        Arguments:
            Xtest: test data (optional)
            ytest: test labels (optional)
            variational autoencoder kwargs: passed to keras

        Returns: evaluation
        """
        if Xtest is not None:
            loss = self.vae.evaluate(Xtest, ytest, batch_size)
        else:
            loss, acc = self.vae.evaluate()

        return loss
