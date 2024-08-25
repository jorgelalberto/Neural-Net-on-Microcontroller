from ulab import numpy as np
import time
import random
from consts import Waves
from drivers.LCD import LCD
from machine import SoftSPI, Pin
import machine

lcd = LCD(enable_pin=14,           # Enable Pin, int
         reg_select_pin=15,        # Register Select, int
         data_pins=[9,8,7,6]   # Data Pin numbers for the upper nibble. list[int]
         )
spi_0 = machine.SoftSPI(baudrate=500000,
        polarity=0,
        phase=0,
        bits=8,
        firstbit=SoftSPI.MSB,
        sck=Pin(1),
        mosi=Pin(0),
        miso=machine.Pin(3))
cs = machine.Pin(2, mode=Pin.OUT, value=1)
waves = Waves()


class ANN:

    def __init__(self, data, num_labels) -> None:
        self.data = data
        self.m, self.n = data.shape
        self.num_labels = num_labels

        data_test = self.data[0:int(self.m*.2)].T
        self.Y_test = data_test[self.n-1]   # target vals, last column
        self.one_hot_Y_test = self.one_hot(self.Y_test)
        self.X_test = data_test[0:self.n-1] / 255. # pixel vals normalized

        data_train = self.data[int(self.m*.2):self.m].T
        self.Y_train = data_train[self.n-1]   # target vals, last column
        self.one_hot_Y_train = self.one_hot(self.Y_train)
        self.X_train = data_train[0:self.n-1] / 255. # pixel vals normalized

    def init_params(self):
        W1 = [ [.5] * (self.n-1) for digit in range(self.num_labels)]
        b1 = [ [.21] * 1 for digit in range(self.num_labels)]
        W2 = [ [.32] * self.num_labels for digit in range(self.num_labels)]
        b2 = [ [0.0] * 1 for digit in range(self.num_labels)]
    
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        
        return W1, b1, W2, b2

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        exp = np.exp(Z - np.max(Z)) 
        return exp / np.sum(exp, axis=0)
        
    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = np.dot(W1, X) + b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self.num_labels))
        for i in range(Y.size):
            one_hot_Y[i][Y[i]] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        dZ2 = A2 - self.one_hot_Y_train
        dW2 = 1 / self.m * np.dot(dZ2, A1.T)
        db2 = 1 / self.m * np.sum(dZ2)
        dZ1 = np.dot(W2.T, dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / self.m * np.dot(dZ1, X.T)
        db1 = 1 / self.m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 -= alpha * dW1
        b1 -= alpha * db1    
        W2 -= alpha * dW2  
        b2 -= alpha * db2    
        return W1, b1, W2, b2

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        # ANN inits
        W1, b1, W2, b2 = self.init_params()
        # DAC inits
        waveform = waves.get("square")
        control_code = 0b1010 << 12 # 1010 Load DAC B
        j=0
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                predictions = self.get_predictions(A2)
                acc = self.get_accuracy(predictions, Y)
                # LCD #
                lcd.clear()
                lcd.print(f'Epoch {i}')
                lcd.go_to(0,1)
                lcd.print(f'{(acc*100):.{2}f}% accuracy')
                # DAC #
                x = 250
                input_code = x << 2
                data = control_code | input_code
                try:
                    cs(0)
                    buf = data.to_bytes(2,'big')
                    spi_0.write(buf)
                finally:
                        cs(1)
                j+=1

        return W1, b1, W2, b2

    def make_predictions(self, W1, b1, W2, b2, X):
        _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
        return self.get_predictions(A2)

    def test_prediction(self, index, W1, b1, W2, b2):
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

    def rand_test_prediction(self, W1, b1, W2, b2):
        index = random.randrange(0, len(self.X_train.T), 1)
        current_image = np.array([self.X_train.T[index]]).T
        prediction = self.make_predictions(W1, b1, W2, b2, current_image)
        label = self.Y_train[index]
        # LCD #
        lcd.clear()
        lcd.print(f'Expected {prediction}')
        lcd.go_to(0,1)
        lcd.print(f'Predicted {label}     ')
        return prediction == label

if __name__ == '__main__':
    data_list = []
    # Start timing
    start_time = time.time()

    data = np.load('./data/zerosNones_200.npy')
    
    # Stop timing
    end_time = time.time()

    print(f"Processed Data in {end_time - start_time} seconds")

    ann = ANN(data=data, num_labels=2)

    W1, b1, W2, b2 = ann.gradient_descent(ann.X_train, ann.Y_train, .4, 1_000)