from typing import Protocol

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt


class Metric(Protocol):
    @tf.function
    def __call__(self, y_true, y_pred) -> tf.Tensor: ...


class MeanAbsoluteError:
    @tf.function
    def __call__(self, y_true, y_pred) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(y_true - y_pred))


class MeanSquaredError:
    @tf.function
    def __call__(self, y_true, y_pred) -> tf.Tensor:
        return tf.reduce_mean(tf.square(y_true - y_pred))


class Optimizer(Protocol):
    @tf.function
    def __call__(
        self,
        x,
        y_true,
        model,
        tape: tf.GradientTape,
        loss_fun: Metric,
        variables,
        learning_rate,
    ) -> (tf.Tensor, tf.Tensor): ...


class GradientDescent:
    def __call__(
        self,
        x,
        y_true,
        model,
        tape: tf.GradientTape,
        loss_fun,
        variables,
        learning_rate,
    ):
        y_pred = model(x)
        loss = loss_fun(y_true, y_pred)
        dw = tape.gradient(loss, variables)
        for i in range(len(variables)):
            variables[i].assign_sub(learning_rate * dw[i])
        return loss, dw


class StochasticGradientDescent:
    def __call__(self, x, y_true, model, tape, loss_fun, variables, learning_rate):
        i = tf.random.uniform(shape=[], minval=0, maxval=len(x), dtype=tf.int32)
        x_sample = x[i]
        y_sample = y_true[i]
        gradient_descent = GradientDescent()
        return gradient_descent(
            x_sample, y_sample, model, tape, loss_fun, variables, learning_rate
        )


class LinearRegression(tf.Module):
    def __init__(self, random_initialization: bool = False):
        # random initialization
        if random_initialization:
            rands = tf.random.uniform(
                shape=[3], minval=0.0, maxval=5.0, dtype=tf.float64
            )
        else:
            rands = tf.constant(0.0, shape=[3], dtype=tf.float64)
        self.w_0 = tf.Variable(rands[0], name="w_0")
        # self.w_1 = tf.Variable(rands[1])
        self.bias = tf.Variable(rands[2], name="b")

    @tf.function
    def __call__(self, x_0: tf.Variable) -> tf.Variable:
        return self.w_0 * x_0 + self.bias


class LinearRegressionWithDegree(tf.Module):
    def __init__(
        self, degree: int, dtype=tf.float64, random_initialization: bool = False
    ):
        self._degree = degree
        # n, n-1, n-2, ... 1, 0
        self.powers = [i for i in range(degree + 1)]
        self.coefficients = [
            tf.Variable(0.0, name=f"w_{i}", dtype=dtype) for i in range(degree + 1)
        ]

    @tf.function
    def __call__(self, x):
        summands = []
        for c, p in zip(self.coefficients, self.powers):
            x_i = tf.pow(x, p)
            summands.append(x_i * c)
        return tf.add_n(summands)


def generate_training_data(
    coefficients: [int], size: int = 100, delta: float = 0.0
) -> (np.ndarray, np.ndarray):
    x = np.linspace(-100, 100, size)

    poly = np.polynomial.Polynomial(list(reversed(coefficients)))
    y = poly(x)
    noise = np.random.uniform(-delta, delta, size)
    y_noisy = y + noise

    return x, y_noisy


def main():
    # hyper parameters
    learning_rate = 0.0001
    epochs = 1000
    log_every_n_lines = epochs // 10
    training_samples = 10000
    batch_size = 32
    early_stopping = False
    # y = sum([c * x^i for i, c in enumerate(coefficients)])
    coefficients = [0.5, 1]

    # test data
    x, y_true = generate_training_data(coefficients, size=training_samples, delta=1.0)

    # training
    lin_reg = LinearRegressionWithDegree(degree=len(coefficients) - 1, dtype=tf.float64)
    optimizer: Optimizer = StochasticGradientDescent()
    loss_fun = MeanSquaredError()
    losses = []
    batches = range(0, len(x), batch_size)

    # progress bars
    epoch_progress = tqdm(range(epochs))
    batch_progress = tqdm(batches)
    for epoch in range(epochs):
        epoch_progress.update(1)
        batch_progress.refresh()
        batch_progress.reset()

        for batch, i in enumerate(batches):
            batch_progress.update(1)
            end = min(batch + batch_size, len(x))
            x_batch = x[batch:end]
            y_batch = y_true[batch:end]

            with tf.GradientTape() as tape:
                loss, grads = optimizer(
                    x_batch,
                    y_batch,
                    lin_reg,
                    tape,
                    loss_fun,
                    lin_reg.coefficients,
                    learning_rate,
                )
                losses.append(loss)

        if epoch % log_every_n_lines == 0:
            coeffs = [c.numpy() for c in lin_reg.coefficients]
            grads = [g.numpy() for g in grads]
            tqdm.write(f"training {epoch=}: {coeffs=}, loss = {loss}, {grads=}")

    _weights = [c.numpy() for c in lin_reg.coefficients]

    tqdm.write(f"finished training: {coeffs=}, loss = {loss}")

    # lin_reg = LinearRegression()
    # losses = []
    # for epoch in range(epochs):
    #     with tf.GradientTape() as tape:
    #         y_pred = lin_reg(x)
    #         loss = mse(y_true, y_pred)
    #         dw_0, db = tape.gradient(loss, [lin_reg.w_0, lin_reg.bias])
    #         lin_reg.w_0.assign_sub(learning_rate * dw_0)
    #         lin_reg.bias.assign_sub(learning_rate * db)
    #         losses.append(loss)
    #         if epoch % log_every_n_lines == 0:
    #             print(
    #                     f"training {epoch=}: m = {lin_reg.w_0.value()}, n = {lin_reg.bias.value()}, loss = {loss}"
    #                     )
    #
    #     if (
    #             epoch > 2
    #             and early_stopping
    #             and tf.greater(losses[-1], losses[-2]).numpy()
    #             and tf.greater(losses[-1], losses[-3]).numpy()
    #             ):
    #         print("loss is increasing")
    #         print(
    #                 f"training {epoch=}: m = {lin_reg.w_0.value()}, n = {lin_reg.bias.value()}, {loss=}"
    #                 )
    #         break

    # lin_reg = LinearRegressionWithDegree(degree=len(coefficients) - 1, dtype=tf.float64)
    # losses = []
    # for epoch in range(epochs):
    #     with tf.GradientTape() as tape:
    #         y_pred = lin_reg(x)
    #         loss = mae(y_true, y_pred)
    #         dw = tape.gradient(loss, lin_reg.coefficients)
    #         for i in range(len(lin_reg.coefficients)):
    #             lin_reg.coefficients[i].assign_sub(learning_rate * dw[i])
    #         losses.append(loss)
    #         if epoch % log_every_n_lines == 0:
    #             coeffs = [c.numpy() for c in lin_reg.coefficients]
    #             grads = [g.numpy() for g in dw]
    #             print(f"training {epoch=}: {coeffs=}, loss = {loss}, {grads=}")

    fig, (data_ax, loss_ax) = plt.subplots(2, 1)
    data_ax.set_title("Data")
    data_ax.set_xlabel("x")
    data_ax.plot(x, lin_reg(x), color="orange")
    data_ax.scatter(x, y_true, marker="x", s=1, color="blue")
    loss_ax.set_title("Training loss")
    loss_ax.plot(range(len(losses)), losses)
    plt.show()


if __name__ == "__main__":
    main()
