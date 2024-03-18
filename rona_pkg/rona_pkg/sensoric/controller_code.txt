import machine
import select
import utime
import sys

# Setup pins and variables
shake_pin = machine.Pin(1, machine.Pin.IN, machine.Pin.PULL_UP)
vibration = False

# Show that the controller has initialized via onboard LED
LED = machine.Pin(25, machine.Pin.OUT)
LED.value(1)
utime.sleep(1)
LED.value(0)

# Set up the poll object
poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

# Read functions
def read_shake_sensor():
    return shake_pin.value()

# Main loop
while True:
    # Wait for input on stdin
    poll_results = poll_obj.poll(1) # the '1' is how long it will wait for message before looping again (in microseconds)
    if poll_results:
        # read a command from the host
        command = sys.stdin.readline().strip()

        # if command is read, start measure
        if command.lower() == "sensor":
            LED.value(1)
            # Reset states for this measurement
            motion = False
                    
            # Read Shake sensor second for 5 seconds        
            start = utime.time()
            while ((utime.time()-start) <= 5):
                
                #print("Reading shake")
                shake_value = read_shake_sensor()
                utime.sleep(0.05)
                
                if shake_value == 1:
                    print(f"shake:{shake_value}")
                    vibration = True
                    break
            
            #if (vibration):
            if (vibration):
                print("success")
            else:
                print("fail")
            
            LED.value(0)
            
    utime.sleep(0.05)