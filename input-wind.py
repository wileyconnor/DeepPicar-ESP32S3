import sys
import msvcrt

def read_single_keypress():
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):  
            ch = msvcrt.getwch()    
        return ch
    return ' '

def read_single_event():
    return ord(read_single_keypress())

if __name__ == "__main__":
    print("Press keys (q to quit):")
    while True:
        key = read_single_event()
        if key == ord('q'):
            break
        elif key != ord(' '):
            print("%s pressed\r" % key)