import gpflow
from sklearn.metrics import r2_score


class GPFLOWRegressor():
    def __init__(self):
        self.k = gpflow.kernels.Matern52()
        # print_summary(k)

    def fit(self, X_train, y_train):

        y_train = y_train.to_numpy()
        self.model = gpflow.models.GPR(data=(X_train, y_train), kernel=self.k, mean_function=None)
        # print_summary(model)

        # The first two lines correspond to the kernel parameters, and the third one gives the likelihood parameter (the noise variance $\tau^2$ in our model).
        self.model.likelihood.variance.assign(0.01)
        self.model.kernel.lengthscales.assign(0.3)

        # ## Optimize the model parameters
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=100))
        # print_summary(self.model)

    def score(self, X_test, y_test):
        mean, var = self.model.predict_f(X_test)
        score = r2_score(y_test, mean)
        return score
