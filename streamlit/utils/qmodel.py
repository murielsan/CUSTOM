import keras
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tf.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling1D,
)


class QModel:
    def __init__(self):
        self.time_steps = 10
        self.freq_bins = 128

        # Hyper parameters
        self.hidden_units = 1024
        self.drop_rate = 0.5
        self.batch_size = 500

        # Embedded layers
        input_layer = Input(shape=(self.time_steps, self.freq_bins))

        a1 = Dense(self.hidden_units)(input_layer)
        a1 = BatchNormalization()(a1)
        a1 = Activation("relu")(a1)
        a1 = Dropout(self.drop_rate)(a1)

        a2 = Dense(self.hidden_units)(a1)
        a2 = BatchNormalization()(a2)
        a2 = Activation("relu")(a2)
        a2 = Dropout(self.drop_rate)(a2)

        a3 = Dense(self.hidden_units)(a2)
        a3 = BatchNormalization()(a3)
        a3 = Activation("relu")(a3)
        a3 = Dropout(self.drop_rate)(a3)
        cla = Dense(self.hidden_units, activation="linear")(a3)
        att = Dense(self.hidden_units, activation="sigmoid")(a3)

        b1 = Lambda(self.attention_pooling, output_shape=self.pooling_shape)([cla, att])
        b1 = BatchNormalization()(b1)
        b1 = Activation(activation="relu")(b1)
        b1 = Dropout(self.drop_rate)(b1)

        output_layer = Dense(self.time_steps, activation="sigmoid")(b1)

        model = keras.Model(
            inputs=input_layer, outputs=output_layer, name="qiuqiangkong"
        )
        return model.compile(
            loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )

    def attention_pooling(self, inputs, **kwargs):
        [out, self.att] = inputs

        epsilon = 1e-7
        self.att = K.clip(self.att, epsilon, 1.0 - epsilon)
        normalized_att = self.att / K.sum(self.att, axis=1)[:, None, :]

        return K.sum(out * normalized_att, axis=1)

    def pooling_shape(self, input_shape):

        if isinstance(input_shape, list):
            (self.sample_num, self.time_steps, self.freq_bins) = input_shape[0]

        else:
            (self.sample_num, self.time_steps, self.freq_bins) = input_shape

        return (self.sample_num, self.freq_bins)
