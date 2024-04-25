import pyautogui
import time

def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click_mouse(button='left'):
    if button == 'left':
        pyautogui.click()
    elif button == 'right':
        pyautogui.rightClick()

def drag_mouse(x, y, duration=1):
    pyautogui.dragTo(x, y, duration)

def scroll_mouse(amount):
    pyautogui.scroll(amount)

# Example usage:
if __name__ == "__main__":
    # Move the mouse to coordinates (100, 100) and click the left button
    move_mouse(100, 100)
    click_mouse('left')
    # Wait for a second
    time.sleep(1)
    # Drag the mouse to coordinates (200, 200) over a duration of 2 seconds
    drag_mouse(200, 200, 2)
    # Scroll the mouse up by 10 "clicks"
    scroll_mouse(10)
