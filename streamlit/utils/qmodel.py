import keras
import keras.backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, Input, Lambda)


# Auxiliary model function 1
def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1.0 - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


# Auxiliary model function 2


def configure_pooling_shape(config):
    def pooling_shape(input_shape):

        if isinstance(input_shape, list):
            (
                config["sample_num"],
                config["time_steps"],
                config["freq_bins"],
            ) = input_shape[0]

        else:
            (
                config["sample_num"],
                config["time_steps"],
                config["freq_bins"],
            ) = input_shape

        return (config["sample_num"], config["freq_bins"])

    return pooling_shape


# DEFAULT MODEL CONFIGURATION
MODEL_BASE_CONFIG = {
    "time_steps": 10,
    "freq_bins": 128,
    "hidden_units": 1024,
    "drop_rate": 0.5,
}


def topology(override_config={}):
    """
    Define model topology based on config parameters
    """

    config = {**MODEL_BASE_CONFIG, **override_config}
    print("Loading model with config", config)

    # Embedded layers
    input_layer = Input(shape=(config["time_steps"], config["freq_bins"]))

    a1 = Dense(config["hidden_units"])(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation("relu")(a1)
    a1 = Dropout(config["drop_rate"])(a1)

    a2 = Dense(config["hidden_units"])(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation("relu")(a2)
    a2 = Dropout(config["drop_rate"])(a2)

    a3 = Dense(config["hidden_units"])(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation("relu")(a3)
    a3 = Dropout(config["drop_rate"])(a3)
    cla = Dense(config["hidden_units"], activation="linear")(a3)
    att = Dense(config["hidden_units"], activation="sigmoid")(a3)

    b1 = Lambda(attention_pooling, output_shape=configure_pooling_shape(config))(
        [cla, att]
    )
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation="relu")(b1)
    b1 = Dropout(config["drop_rate"])(b1)

    output_layer = Dense(config["time_steps"], activation="sigmoid")(b1)
    return (input_layer, output_layer)


def compile_model(layers=topology(), name="qiuqiangkong"):
    """
    Compile model and select optimizer
    """
    input_layer, output_layer = layers

    # Create model instance
    model = keras.Model(inputs=input_layer, outputs=output_layer, name=name)
    # Define the optimizer
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

    return model
