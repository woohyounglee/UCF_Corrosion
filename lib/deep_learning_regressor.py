from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import tensorflow.keras.layers as KL
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


class DeepLearningRegressor():
    def __init__(self, type='custom'):
        self.type = type

    def nn_block(self, input_layer, size, dropout_rate, activation):
        out_layer = KL.Dense(size, activation=None)(input_layer)
        #out_layer = KL.BatchNormalization()(out_layer)
        out_layer = KL.Activation(activation)(out_layer)
        # out_layer = KL.Dropout(dropout_rate)(out_layer)
        return out_layer

    def nn_block_simple(self, input_layer, size, activation):
        out_layer = KL.Dense(size, activation=activation)(input_layer)
        return out_layer

    def larger_model(self, input_size):
        # create model
        self.model = Sequential()
        self.model.add(Dense(13, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(6, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def wider_model(self, input_size):
        # create model
        self.model = Sequential()
        self.model.add(Dense(20, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def make_custom_model(self, input_size):
        inp = KL.Input(shape=input_size)

        # hidden_layer = self.nn_block(inp, 64, 0.0, "relu")
        # gate_layer = self.nn_block(hidden_layer, 32, 0.0, "sigmoid")
        # hidden_layer = self.nn_block(hidden_layer, 32, 0.0, "relu")
        # hidden_layer = KL.multiply([hidden_layer, gate_layer])

        # hidden_layer = self.nn_block_simple(inp, 400, "relu")
        # hidden_layer = self.nn_block_simple(hidden_layer, 400, "relu")
        # hidden_layer = self.nn_block_simple(hidden_layer, 400, "relu")

        hidden_layer = self.nn_block_simple(inp, 64, "relu")
        hidden_layer = self.nn_block_simple(hidden_layer, 64, "relu")
        hidden_layer = self.nn_block_simple(hidden_layer, 64, "relu")

        # out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)
        out = KL.Dense(1, activation="linear")(hidden_layer)

        self.model = tf.keras.models.Model(inputs=[inp], outputs=out)

        return self.model

    def fit(self, X_train, y_train):
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        input_size = X_train.shape[1]
        if self.type == 'custom':
            self.make_custom_model(input_size)
        elif self.type == 'larger':
            self.larger_model(input_size)
        elif self.type == 'wider':
            self.wider_model(input_size)

        print(self.model.summary())

        # model_json = self.model.to_json()
        # with open("DL_model.json", "w") as json_file:
        #     json_file.write(model_json)

        self.model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))
        hist = self.model.fit(X_train, y_train, batch_size=5, epochs=500, verbose=0, shuffle=True)
        # KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0))
        return self.model

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score
