import numpy as np


class Layer:

    def __init__(self, neuron_count, input_shape=None):
        self.neuron_count = neuron_count
        self.last_layer: Layer = None
        self.next_layer: Layer = None

        self.theta: np.ndarray = None
        self.delta: np.ndarray = np.zeros((self.neuron_count, 1))
        self.gradient: np.ndarray = None
        self.z: np.ndarray = None
        self.a: np.ndarray = None

        if input_shape is not None:
            count = 1
            for i in input_shape:
                count *= i
            self.theta = np.zeros((self.neuron_count, count + 1))
            self.gradient = np.zeros(self.theta.shape)

    def set_last_layer(self, last_layer):
        self.last_layer = last_layer
        self.theta = np.zeros((self.neuron_count, self.last_layer.neuron_count + 1))
        self.gradient = np.zeros(self.theta.shape)

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def fp(self, input_data: np.ndarray):
        input_data = input_data.reshape(self.theta.shape[1] - 1, 1)
        input_data = np.insert(input_data, input_data.shape[0], values=np.array([1, ]), axis=0)
        self.z = np.dot(self.theta, input_data)
        self.a = 1 / (1 + np.exp(0 - self.z))
        return self.a

    def bp(self, input_error):
        if self.last_layer is None:
            return
        self.delta += input_error
        a_temp = np.insert(self.last_layer.a, self.last_layer.a.shape[0], values=np.array([1, ]), axis=0)
        self.gradient = np.dot(self.delta, a_temp.transpose())
        self.theta -= 0.01 * self.gradient
        return np.dot(self.theta.transpose(), input_error)[1:self.last_layer.neuron_count + 1] * self.last_layer.a * (1 - self.last_layer.a)


class Model:

    def __init__(self):
        self.header_layer: Layer = None
        self.rear_layer: Layer = None

    def add_layer(self, layer: Layer):
        if self.header_layer is None:
            self.header_layer = layer
        if self.rear_layer is None:
            self.rear_layer = layer
        else:
            self.rear_layer.set_next_layer(layer)
            layer.set_last_layer(self.rear_layer)
            self.rear_layer = layer

    def fp_calculate(self, input_data):
        cur_layer = self.header_layer
        result = input_data
        while cur_layer is not None:
            result = cur_layer.fp(result)
            cur_layer = cur_layer.next_layer
        return result

    def bp_calculate(self, input_delta):
        cur_layer = self.rear_layer
        cur_delta = input_delta
        while cur_layer is not None:
            cur_delta = cur_layer.bp(cur_delta)
            cur_layer = cur_layer.last_layer

    def fit(self, train_x, train_y, time=1000):
        sample_count = train_x.shape[0]

        for t in range(0, time):
            loss = 0
            for i in range(0, sample_count):
                cur_x = train_x[i]
                cur_y = train_y[i]
                cur_result = self.fp_calculate(cur_x)
                self.bp_calculate(cur_result - cur_y)
                cur_loss = cur_y * np.log(cur_result) + (1 - cur_y) * np.log(1 - cur_result)
                loss += cur_loss
            loss = -np.sum(loss)
            loss /= sample_count
            print(loss)


def main():
    x = np.array([i for i in range(0, 100)])
    y = np.array([1 if x[i] > 50 else 0 for i in range(0, x.shape[0])])

    model: Model = Model()
    model.add_layer(Layer(1, input_shape=(1,)))
    model.add_layer(Layer(2))
    model.add_layer(Layer(1))
    model.fit(x, y)


if __name__ == '__main__':
    main()