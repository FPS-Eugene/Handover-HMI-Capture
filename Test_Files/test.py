import keyboard

def my_keyboard_hook(keyboard_event):
    print("Name:", keyboard_event.name)
    print("Scan code:", keyboard_event.scan_code)
    print("Time:", keyboard_event.time)
    print("Event type", keyboard_event.event_type)

keyboard.hook(my_keyboard_hook)

# Block forever, so that the program won't automatically finish,
# preventing you from typing and seeing the printed output
#keyboard.wait()

input()

# print(keyboard.read_key())

# keyboard.wait('enter')

# print(f"released {keyboard.read_key()}")
























