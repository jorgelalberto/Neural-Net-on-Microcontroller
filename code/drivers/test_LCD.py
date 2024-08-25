# -----------------------------------------
#                 NOTES
# -----------------------------------------
"""
Dieter Steinhauser
9/2023
Parallel LCD class test.
"""
from time import sleep
from drivers.LCD import LCD

# -----------------------------------------
#           LCD Instantiation
# -----------------------------------------

# add this code to the beginning of your main file to instantiate the LCD

# GPIO 
lcd = LCD(enable_pin=14,           # Enable Pin, int
         reg_select_pin=15,        # Register Select, int
         data_pins=[9,8,7,6]   # Data Pin numbers for the upper nibble. list[int]
         )

lcd.init()

lcd.clear()
lcd.cursor_on()
lcd.blink()

lcd.print(f'LCD Test          ')
lcd.go_to(0,1)
lcd.print(f'Dieter S.         ')
sleep(2)

while True:
    sleep(1)
    pass
    