from __future__ import division
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Pattern:
    def __init__(self, input, output):
        self.input = input
        self.output = output

class Data:
    def __init__(self, patterns):
        self.patterns = patterns


# تعریف Dropout
def apply_dropout(layer, dropout_rate):
    """
    Apply dropout to a layer.
    :param layer: The layer to apply dropout to.
    :param dropout_rate: The probability of dropping a neuron.
    :return: The layer with dropout applied.
    """
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be in the range [0, 1).")
    mask = np.random.binomial(1, 1 - dropout_rate, size=layer.shape)
    return layer * mask / (1 - dropout_rate)  # Scale the output to maintain the expected value


class RBFNetwork:
    def __init__(self, no_of_input, no_of_hidden, no_of_output, data, learning_rate=0.0262, sigma_range=(0.1, 1.0), 
                 weight_range=(0, 1), bias_range=(0, 1), dropout_rate=0.5):
        """
        Initialize the RBF Network with customizable hyperparameters.

        :param no_of_input: Number of input features.
        :param no_of_hidden: Number of hidden neurons.
        :param no_of_output: Number of output neurons.
        :param data: Training data.
        :param learning_rate: Learning rate for gradient descent (default: 0.0262).
        :param sigma_range: Range for initial sigma values (default: (0.1, 1.0)).
        :param weight_range: Range for initial weights (default: (0, 1)).
        :param bias_range: Range for initial biases (default: (0, 1)).
        :param dropout_rate: Dropout rate for regularization (default: 0.5).
        """
        self.no_of_input = no_of_input
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.data = data
        self.input = np.zeros(self.no_of_input)
        self.centroid = np.zeros((self.no_of_hidden, self.no_of_input))
        self.sigma = np.zeros(self.no_of_hidden)
        self.hidden_output = np.zeros(self.no_of_hidden)
        self.hidden_to_output_weight = np.zeros((self.no_of_hidden, self.no_of_output))
        self.output = np.zeros(self.no_of_output)
        self.output_bias = np.zeros(self.no_of_output)
        self.actual_target_values = []
        self.total = 0
        self.learningRate = learning_rate
        self.sigma_range = sigma_range
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.dropout_rate = dropout_rate  # Dropout rate
        self.setup_center()
        self.setup_sigma_spread_radius()
        self.set_up_hidden_to_ouput_weight()
        self.set_up_output_bias()

    def setup_center(self):
        """Setup center using clustering ,for now just randomize between 0 and 1"""
        # print("Setup center")
        for i in range(self.no_of_hidden):
            self.centroid[i] = np.random.uniform(0, 1, self.no_of_input)

    def setup_sigma_spread_radius(self):
        for i in range(self.no_of_hidden):
            self.sigma[i] = np.random.uniform(0.1, 1.0)  # مقدار اولیه sigma بین 0.1 و 1.0

    def set_up_sigma_for_center(self, center):
        # print("Get sigma for center")
        p = self.no_of_hidden / 3
        sigma = 0
        distances = [0 for i in range(self.no_of_hidden)]
        for i in range(self.no_of_hidden):
            distances[i] = self.euclidean_distance(center, self.centroid[i])
            # print("Distance ", i, distances[i])
        sum = 0
        for i in range(int(p)):
            nearest = self.get_smallest_index(distances)
            distances[nearest] = float("inf")

            neighbour_centroid = self.centroid[nearest]
            for j in range(len(neighbour_centroid)):
                sum += (center[j] - neighbour_centroid[j]) ** 2

        sigma = sum / p
        sigma = math.sqrt(sigma)
        #return random.uniform(0, 1) * 6
        return sigma

    @staticmethod
    def euclidean_distance( x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def get_smallest_index( distances):
        min_index = 0
        for i in range(len(distances)):
            if (distances[min_index] > distances[i]):
                min_index = i
        return min_index

    def set_up_hidden_to_ouput_weight(self):
        print("Setup hidden to output weight")
        self.hidden_to_output_weight = np.random.uniform(0, 1, (self.no_of_hidden, self.no_of_output))

        print("Hiden to output weight ", self.hidden_to_output_weight)

    def set_up_output_bias(self):
        print("Setup output bias")
        self.output_bias = np.random.uniform(0, 1, self.no_of_output)

    # train n iteration
    def train(self, n):
        for i in range(n):
            error = self.pass_one_epoch()
            print("Iteration ", i, " Error ", error)

        return error

    # Train an epoch and return total MSE
    def pass_one_epoch(self):
        # print("Pass one epoch")
        all_error = 0
        all_index = []
        for i in range(len(self.data.patterns)):
            all_index.append(i)
        # print("All index ",all_index)

        for i in range(len(self.data.patterns)):
            random_index = (int)(random.uniform(0, 1) * len(all_index))
            # print("Random index ",random_index, " Len ", len(all_index))
            """Get a random pattern to train"""
            pattern = self.data.patterns[random_index]
            del all_index[random_index]

            input = pattern.input
            self.actual_target_values = pattern.output
            self.pass_input_to_network(input)

            error = self.get_error_for_pattern()
            all_error += error
            self.gradient_descent()

        all_error = all_error / (len(self.data.patterns))
        return all_error

    def pass_input_to_network(self, input):
        self.input = input
        self.pass_to_hidden_node()
        self.pass_to_output_node()

    def pass_to_hidden_node(self):
        # محاسبه خروجی لایه پنهان
        self.hidden_output = np.zeros(self.no_of_hidden)
        for i in range(len(self.hidden_output)):
            euclid_distance = self.euclidean_distance(self.input, self.centroid[i]) ** 2
            self.hidden_output[i] = math.exp(- (euclid_distance / (2 * self.sigma[i] * self.sigma[i])))

        # اعمال Dropout به لایه پنهان
        if self.dropout_rate > 0:
            self.hidden_output = apply_dropout(self.hidden_output, self.dropout_rate)

    def pass_to_output_node(self):
        # محاسبه خروجی لایه خروجی
        self.output = [0 for i in range(self.no_of_output)]
        for i in range(self.no_of_output):
            output_value = 0
            for j in range(self.no_of_hidden):
                self.output[i] += self.hidden_to_output_weight[j][i] * self.hidden_output[j]
            self.output[i] += self.output_bias[i]  # اضافه کردن بایاس به خروجی

        # اعمال Dropout به لایه خروجی (اختیاری)
        if self.dropout_rate > 0:
            self.output = apply_dropout(np.array(self.output), self.dropout_rate)

    # Compute error for the pattern
    def get_error_for_pattern(self):
        error = 0
        for i in range(len(self.output)):
            error += (self.actual_target_values[i] - self.output[i]) ** 2
        return error

    # Weight update by gradient descent algorithm
    def gradient_descent(self):
        # compute the error of output layer
        self.mean_error = 0
        self.error_of_output_layer = [0 for i in range(self.no_of_output)]
        for i in range(self.no_of_output):
            self.error_of_output_layer[i] = (float)(self.actual_target_values[i] - self.output[i])
            e = (float)(self.actual_target_values[i] - self.output[i]) ** 2 * 0.5
            self.mean_error += e

        # Adjust hidden to output weight
        for o in range(self.no_of_output):
            for h in range(self.no_of_hidden):
                delta_weight = self.learningRate * self.error_of_output_layer[o] * self.hidden_output[h]
                self.hidden_to_output_weight[h][o] += delta_weight

        # For bias
        for o in range(self.no_of_output):
            delta_bias = self.learningRate * self.error_of_output_layer[o]
            self.output_bias[o] += delta_bias

        # Adjust center , input to hidden weight
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)((self.input[i] - self.centroid[j][i]) / math.pow(self.sigma[j], 2))
                delta_weight = (float)(self.learningRate * self.hidden_output[j] * second_part * summ)
                self.centroid[j][i] += delta_weight

        # Adjust sigma and spread radius
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)(
                    (math.pow((self.input[i] - self.centroid[j][i]), 2)) / math.pow(self.sigma[j], 3));
                delta_weight = (float)(0.1 * self.learningRate * self.hidden_output[j] * second_part * summ);
                self.sigma[j] += delta_weight
        return self.mean_error

    def get_accuracy_for_training(self):
        # برای رگرسیون، دقت به معنای خطای میانگین مربعات است
        total_error = 0
        for i in range(len(self.data.patterns)):
            pattern = self.data.patterns[i]
            self.pass_input_to_network(pattern.input)
            n_output = self.output
            act_output = pattern.output
            error = self.get_error_for_pattern()
            total_error += error
        mse = total_error / len(self.data.patterns)
        return mse

    def get_fired_neuron(self, output):
        max = 0
        for i in range(len(output)):
            if (output[i] > output[max]):
                max = i
        return max


# تابع محاسبه Mean Absolute Error (MAE)
def calculate_mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

# تابع محاسبه Root Mean Squared Error (RMSE)
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

# تابع محاسبه R-squared (R²)
def calculate_r2(true_values, predicted_values):
    mean_true = np.mean(true_values)
    ss_total = np.sum((true_values - mean_true) ** 2)
    ss_residual = np.sum((true_values - predicted_values) ** 2)
    return 1 - (ss_residual / ss_total)

# تابع محاسبه Mean Absolute Percentage Error (MAPE)
def calculate_mape(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

# تابع محاسبه Median Absolute Error (MedAE)
def calculate_medae(true_values, predicted_values):
    return np.median(np.abs(true_values - predicted_values))

# تابع محاسبه Mean Squared Logarithmic Error (MSLE)
def calculate_msle(true_values, predicted_values):
    return np.mean((np.log1p(true_values) - np.log1p(predicted_values)) ** 2)

# تابع محاسبه Huber Loss
def calculate_huber_loss(true_values, predicted_values, delta=1.0):
    error = np.abs(true_values - predicted_values)
    return np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))

# تابع محاسبه Mean Relative Error (MRE)
def calculate_mre(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values))


data_test = pd.read_csv('testdata.csv', header=None)
data_train = pd.read_csv('finaldata.csv', header=None)

train_patterns = []
for index, row in data_train.iterrows():
    input_value = [row[0]]  # ورودی به صورت لیست
    output_value = [row[1]]  # خروجی به صورت لیست
    train_patterns.append(Pattern(input=input_value, output=output_value))

test_patterns = []
for index, row in data_test.iterrows():
    input_value = [row[0]]  # ورودی به صورت لیست
    output_value = [row[1]]  # خروجی به صورت لیست
    test_patterns.append(Pattern(input=input_value, output=output_value))

train_data = Data(train_patterns)
test_data = Data(test_patterns)

rbf_network = RBFNetwork(no_of_input=1, no_of_hidden=5, no_of_output=1, data=train_data)

# آموزش شبکه
rbf_network.train(n=1000)

# جمع‌آوری مقادیر واقعی و پیش‌بینی‌شده
true_values = []
predicted_values = []

for pattern in test_data.patterns:
    rbf_network.pass_input_to_network(pattern.input)
    true_values.append(pattern.output[0])  # مقدار واقعی
    predicted_values.append(rbf_network.output[0])  # مقدار پیش‌بینی‌شده

# تبدیل به آرایه NumPy برای محاسبات
true_values = np.array(true_values)
predicted_values = np.array(predicted_values)

# محاسبه معیارها
mae = calculate_mae(true_values, predicted_values)
rmse = calculate_rmse(true_values, predicted_values)
r2 = calculate_r2(true_values, predicted_values)
mape = calculate_mape(true_values, predicted_values)
medae = calculate_medae(true_values, predicted_values)
msle = calculate_msle(true_values, predicted_values)
huber_loss = np.mean(calculate_huber_loss(true_values, predicted_values))
mre = calculate_mre(true_values, predicted_values)

# چاپ نتایج
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Median Absolute Error (MedAE): {medae}")
print(f"Mean Squared Logarithmic Error (MSLE): {msle}")
print(f"Huber Loss: {huber_loss}")
print(f"Mean Relative Error (MRE): {mre}")

# رسم سیگنال خروجی شبکه عصبی RNN
def plot_rnn_output(rnn, data):
    """
    Plot the output signal of the RNN model.
    :param rnn: The RNN model.
    :param data: Test data.
    """
    predicted_values_rbf = []
    for pattern in data.patterns:
        rbf_network.pass_input_to_network(pattern.input)
        predicted_values_rbf.append(rbf_network.output[0])
    
    predicted_values_rnn = []
    for pattern in data.patterns:
        x = np.array(pattern.input).reshape(1, -1)  # ورودی
        y_pred = rnn.forward(x)  # پیش‌بینی خروجی
        predicted_values_rnn.append(y_pred[0][0])
    
    # رسم نمودار سیگنال خروجی
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_values_rbf, label='RBF Signal', marker='o', linestyle='-', color='red')
    plt.plot(predicted_values_rnn, label='RNN Signal', marker='o', linestyle='-', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.title('RNN Output Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0262):
        """
        Initialize a simple RNN.
        :param input_size: Number of input features.
        :param hidden_size: Number of hidden units.
        :param output_size: Number of output units.
        :param learning_rate: Learning rate for gradient descent.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # وزن‌ها و بایاس‌ها
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # وزن برای ورودی به پنهان
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # وزن برای پنهان به پنهان
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01  # وزن برای پنهان به خروجی
        self.b_h = np.zeros((1, hidden_size))  # بایاس برای لایه پنهان
        self.b_y = np.zeros((1, output_size))  # بایاس برای لایه خروجی
        
        # حالت پنهان
        self.hidden_state = np.zeros((1, hidden_size))

    def forward(self, x):
        """
        Forward pass through the RNN.
        :param x: Input data (shape: [1, input_size]).
        :return: Output of the RNN.
        """
        # به‌روزرسانی حالت پنهان
        self.hidden_state = np.tanh(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + self.b_h)
        
        # محاسبه خروجی
        output = np.dot(self.hidden_state, self.W_hy) + self.b_y
        return output

    def backward(self, x, target):
        """
        Backward pass through the RNN (gradient descent).
        :param x: Input data (shape: [1, input_size]).
        :param target: Target output (shape: [1, output_size]).
        """
        # محاسبه خطا
        output = self.forward(x)
        error = output - target
        
        # گرادیان‌ها
        dW_hy = np.dot(self.hidden_state.T, error)
        db_y = error
        
        dh = np.dot(error, self.W_hy.T) * (1 - self.hidden_state ** 2)
        dW_xh = np.dot(x.T, dh)
        dW_hh = np.dot(self.hidden_state.T, dh)
        db_h = dh
        
        # به‌روزرسانی وزن‌ها و بایاس‌ها
        self.W_hy -= self.learning_rate * dW_hy
        self.b_y -= self.learning_rate * db_y
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.b_h -= self.learning_rate * db_h

    def reset_hidden_state(self):
        """
        Reset the hidden state of the RNN.
        """
        self.hidden_state = np.zeros((1, self.hidden_size))


rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1, learning_rate=0.0262)

# رسم سیگنال خروجی برای داده‌های تست
plot_rnn_output(rnn, test_data)