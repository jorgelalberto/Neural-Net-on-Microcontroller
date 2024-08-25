from drivers.LCD import LCD
from machine import SoftSPI, Pin, ADC
from ulab import numpy as np
from consts import Waves
from ANN import ANN
import utime

"""instantiations"""
lcd = LCD(enable_pin=14,           # Enable Pin, int
         reg_select_pin=15,        # Register Select, int
         data_pins=[9,8,7,6]   # Data Pin numbers for the upper nibble. list[int]
         )

spi_0 = SoftSPI(baudrate=500000,
        polarity=0,
        phase=0,
        bits=8,
        firstbit=SoftSPI.MSB,
        sck=Pin(1),
        mosi=Pin(0),
        miso=machine.Pin(3))
cs = Pin(2, mode=Pin.OUT, value=1)

analog_pin = Pin(27, machine.Pin.IN)
analog_pin = ADC(27)

btn1 = Pin(22, machine.Pin.IN, machine.Pin.PULL_UP) # 0 pressed, 1 not pressed
btn2 = Pin(21, machine.Pin.IN, machine.Pin.PULL_UP) # 0 pressed, 1 not pressed

red_led = Pin(18, machine.Pin.OUT)
green_led = Pin(17, machine.Pin.OUT)
yellow_led = Pin(16, machine.Pin.OUT)

waves = Waves()

start_time = utime.time_ns()
data = []
data = np.load('./data/zerosNones_200.npy')
end_time = utime.time_ns()
data_processing_time = int(end_time - start_time)/1_000_000
ann = ANN(data=data, num_labels=2)

"""initializations"""
lcd.init()
lcd.clear()

conversion_factor = 1 / 65535.

green_led.value(0)
red_led.value(0)
yellow_led.value(0)

control_code = 0b1010 << 12 # 1010 Load DAC B
waveform = waves.get("sin")

lcd.print(f'Loaded Data in    ')
lcd.go_to(0,1)
lcd.print(f'{data_processing_time} us     ')

"""ML vars"""
W1,b1,W2,b2 = ann.init_params()

def output_wave(waveform_name, freq=500, signal_periods=1_000):
    waveform = waves.get(waveform_name)
    for t in range(signal_periods):
        for i in range(50): # 50 is len of array
            analog_input = analog_pin.read_u16()
            volt_input = analog_input * conversion_factor
            input_code = int(waveform[i]*volt_input) << 2
            data = control_code | input_code
            try:
                cs(0)
                buf = data.to_bytes(2,'big')
                spi_0.write(buf)
            finally:
                cs(1)
            utime.sleep_us(int(1000000/(50*freq))) # 10Hz @ 100ms - 100Hz @ 10ms

while True:
    train = not btn1.value()
    test = not btn2.value()

    ### Train ANN ###
    if train:
        yellow_led.value(1)
        lcd.clear()
        lcd.print(f'ANN Training    ')
        lcd.go_to(0,1)
        lcd.blink()
        utime.sleep(2)
        W1, b1, W2, b2 = ann.gradient_descent(ann.X_train, ann.Y_train, .8, 250)
        output_wave("sin")
        yellow_led.value(0)
    
    ### Test ANN ###
    elif test:
        yellow_led.value(1)
        lcd.clear()
        lcd.print(f'ANN Testing     ')
        lcd.go_to(0,1)
        lcd.blink()
        utime.sleep(2)
        yellow_led.value(0)
        test_correct = ann.rand_test_prediction(W1, b1, W2, b2)
        
        if test_correct:
            green_led.value(1)
            output_wave("sin")
            green_led.value(0)
        elif not test_correct:
            red_led.value(1)
            output_wave("square")
            red_led.value(0)
