# -----------------------------------------
#                 NOTES
# -----------------------------------------
"""
Dieter Steinhauser
5/2023
Parallel LCD class, Configured for 4-bit communication.
"""

# -----------------------------------------
#               IMPORTS
# -----------------------------------------

from machine import Pin
import utime

# -----------------------------------------
#            LCD Instantiation
# -----------------------------------------

# add this code to the beginning of your main file to instantiate the LCD

# GPIO 
# lcd = LCD(enable_pin=0,           # Enable Pin, int
         # reg_select_pin=1,        # Register Select, int
         # data_pins=[2, 3, 4, 5]   # Data Pin numbers for the upper nibble. list[int]
         # )

# lcd.init()
# lcd.clear()
# lcd.cursor_on()
# lcd.blink()


# -----------------------------------------
#                 LCD Class:
# -----------------------------------------


class LCD:
    """The LCD class is meant to abstract the LCD driver further and streamline development."""


    def __init__(self, enable_pin: int, reg_select_pin: int, data_pins: list) -> None:
        """Object initialization"""

        self.enable_pin = Pin(enable_pin, Pin.OUT)
        self.reg_select_pin = Pin(reg_select_pin, Pin.OUT)
        self._data_pins = data_pins
        self.data_bus = []

        # Configure the pins of the device.
        self._configure()
        utime.sleep_ms(120)

    # -----------------------------------------

    def _configure(self):
        """Creates the data bus object from the pin list. """

        # Configure the pins of the device.
        for element in self._data_pins:
            self.data_bus.append(Pin(element, Pin.OUT))

    # -----------------------------------------

    def init(self):
        """Initializes the LCD for communication."""

        # clear values on data bus.
        for index in range(4):
            self.data_bus[index].value(0)
        utime.sleep_ms(50)

        # initialization sequence.
        self.data_bus[0].value(1)
        self.data_bus[1].value(1)
        self.strobe()
        utime.sleep_ms(10)

        self.strobe()
        utime.sleep_ms(10)

        self.strobe()
        utime.sleep_ms(10)

        self.data_bus[0].value(0)
        self.strobe()
        utime.sleep_ms(5)

        self.write(0x28, 0)
        utime.sleep_ms(1)

        self.write(0x08, 0)
        utime.sleep_ms(1)

        self.write(0x01, 0)
        utime.sleep_ms(10)

        self.write(0x06, 0)
        utime.sleep_ms(5)

        self.write(0x0C, 0)
        utime.sleep_ms(10)

    # -----------------------------------------

    def strobe(self):
        """Flashes the enable line and provides wait period."""

        self.enable_pin.value(1)
        utime.sleep_ms(1)

        self.enable_pin.value(0)
        utime.sleep_ms(1)

    # -----------------------------------------

    def write(self, command, mode):
        """Sends data to the LCD module. """

        # determine if writing a command or data
        data = command if mode == 0 else ord(command)

        # need upper nibble for first loop. lower nibble can use data directly.
        upper = data >> 4

        # write the upper nibble
        for index in range(4):
            bit = upper & 1
            self.data_bus[index].value(bit)
            upper = upper >> 1

        # strobe the LCD, sending the nibble
        self.reg_select_pin.value(mode)
        self.strobe()

        # write the lower nibble
        for index in range(4):
            bit = data & 1
            self.data_bus[index].value(bit)
            data = data >> 1

        # Strobe the LCD, sending the nibble
        self.reg_select_pin.value(mode)
        self.strobe()
        utime.sleep_ms(1)
        self.reg_select_pin.value(1)

    # -----------------------------------------

    def clear(self):
        """Clear the LCD Screen."""

        self.write(0x01, 0)
        utime.sleep_ms(5)

    # -----------------------------------------

    def home(self):
        """Return the Cursor to the starting position."""

        self.write(0x02, 0)
        utime.sleep_ms(5)

    # -----------------------------------------


    def blink(self):
        """Have the cursor start blinking."""

        self.write(0x0D, 0)
        utime.sleep_ms(1)

    # -----------------------------------------

    def cursor_on(self):
        """Have the cursor on, Good for debugging."""

        self.write(0x0E, 0)
        utime.sleep_ms(1)

    # -----------------------------------------

    def cursor_off(self):
        """Turn the cursor off."""

        self.write(0x0C, 0)
        utime.sleep_ms(1)

    # -----------------------------------------

    def print(self, string):
        """Write a string on to the LCD."""

        for element in string:
            self._putch(element)

    # -----------------------------------------

    def _putch(self, c):
        """Write a character on to the LCD."""
        self.write(c, 1)

    # -----------------------------------------

    def _puts(self, string):
        """Write a string on to the LCD."""

        for element in string:
            self._putch(element)


    # -----------------------------------------
    def go_to(self, column, row):


        if row == 0:
            address = 0

        if row == 1:
            address = 0x40

        if row == 2:
            address = 0x14

        if row == 3:
            address = 0x54

        address = address + column
        self.write(0x80 | address, 0)


# -----------------------------------------
#              END OF FILE
# -----------------------------------------