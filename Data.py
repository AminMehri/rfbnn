# train RNN model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# تعریف کلاس RNN
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

# کلاس‌های Pattern و Data بدون تغییر باقی می‌مانند.

class Pattern:
    def __init__(self, input, output):
        self.input = input
        self.output = output

class Data:
    def __init__(self, patterns):
        self.patterns = patterns

# آموزش مدل RNN
def train_rnn(rnn, data, epochs=1000):
    """
    Train the RNN model.
    :param rnn: The RNN model.
    :param data: Training data.
    :param epochs: Number of training epochs.
    """
    mse_values = []
    for epoch in range(epochs):
        total_error = 0
        for pattern in data.patterns:
            x = np.array(pattern.input).reshape(1, -1)  # ورودی
            y = np.array(pattern.output).reshape(1, -1)  # خروجی واقعی
            
            # پیش‌بینی و به‌روزرسانی وزن‌ها
            rnn.forward(x)
            rnn.backward(x, y)
            
            # محاسبه خطا
            error = np.mean((y - rnn.forward(x)) ** 2)
            total_error += error
        
        # محاسبه MSE برای این اپوک
        mse = total_error / len(data.patterns)
        mse_values.append(mse)
        
        # نمایش خطا هر ۱۰۰ اپوک
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")
    
    # رسم نمودار خطا
    plt.plot(mse_values, label='MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE over Epochs')
    plt.legend()
    plt.show()

# تست مدل RNN
def test_rnn(rnn, data):
    """
    Test the RNN model.
    :param rnn: The RNN model.
    :param data: Test data.
    """
    true_values = []
    predicted_values = []
    
    for pattern in data.patterns:
        x = np.array(pattern.input).reshape(1, -1)  # ورودی
        y = np.array(pattern.output).reshape(1, -1)  # خروجی واقعی
        
        # پیش‌بینی
        y_pred = rnn.forward(x)
        
        true_values.append(y[0][0])
        predicted_values.append(y_pred[0][0])
    
    # رسم نمودار خروجی واقعی و پیش‌بینی‌شده
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values', marker='o', linestyle='-', color='blue')
    plt.plot(predicted_values, label='Predicted Values', marker='x', linestyle='--', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rnn_output(rnn, data):
    """
    Plot the output signal of the RNN model.
    :param rnn: The RNN model.
    :param data: Test data.
    """
    predicted_values = []
    
    for pattern in data.patterns:
        x = np.array(pattern.input).reshape(1, -1)  # ورودی
        y_pred = rnn.forward(x)  # پیش‌بینی خروجی
        predicted_values.append(y_pred[0][0])
    
    # رسم نمودار سیگنال خروجی
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_values, label='Predicted Output', marker='o', linestyle='-', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.title('RNN Output Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

# خواندن داده‌ها
data_test = pd.read_csv('testdata.csv', header=None)
data_train = pd.read_csv('finaldata.csv', header=None)

# آماده‌سازی داده‌های آموزش و تست
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

# ایجاد مدل RNN
rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1, learning_rate=0.0262)

# آموزش مدل RNN
train_rnn(rnn, train_data, epochs=1000)

# تست مدل RNN
test_rnn(rnn, test_data)

# رسم سیگنال خروجی برای داده‌های تست
plot_rnn_output(rnn, test_data)