#!/usr/bin/env python3
"""
menu_ui.py - A curses-based UI for running the ngram calculator or unit tests.
Provides a clean, colorful interface with plugin-decorated layers, layer and scene managers,
and a configuration for commands. Navigation uses 'w' (up), 's' (down), space/enter or 1/2/3 to select.
After running a command, its output is displayed (fitting the screen) before returning to the menu.
Version: 1.0.1
"""

import curses
import shutil
import subprocess
from abc import ABC, abstractmethod

# --------------------------
# Config Class
# --------------------------
class Config:
    def __init__(self):
        self.background_color = curses.COLOR_BLACK
        self.border_color = curses.COLOR_GREEN
        self.menu_color = curses.COLOR_WHITE
        self.title_color = curses.COLOR_RED
        self.ngram_command = "python src/ngram_calculator.py data/english.csv data/sample_test_data/english_test_data.csv src/output.csv"
        self.test_command = "python manage.py test"
        self.menu_items = ["1. Ngram Calculator", "2. Unit Tests", "3. Quit"]

# --------------------------
# Plugin Class and Decorator
# --------------------------
class Plugin:
    @staticmethod
    def layer_decorator(func):
        def wrapper(*args, **kwargs):
            # Plugin hook: additional processing or logging could go here.
            return func(*args, **kwargs)
        return wrapper

# --------------------------
# Layer Classes
# --------------------------
class Layer:
    def __init__(self, config):
        self.config = config

    def draw(self, win):
        pass

class BackgroundLayer(Layer):
    @Plugin.layer_decorator
    def draw(self, win):
        win.bkgd(' ', curses.color_pair(1))
        win.erase()

class BorderLayer(Layer):
    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(2))
        win.box()
        win.attroff(curses.color_pair(2))

class TitleLayer(Layer):
    @Plugin.layer_decorator
    def draw(self, win):
        title = "Ngram Calculator UI"
        height, width = win.getmaxyx()
        # Place the title in the middle-top (e.g., row = height//4)
        y = max(height // 4, 0)
        x = max((width - len(title)) // 2, 0)
        win.attron(curses.color_pair(3) | curses.A_BOLD)
        win.addstr(y, x, title)
        win.attroff(curses.color_pair(3) | curses.A_BOLD)

class MenuLayer(Layer):
    def __init__(self, config, menu_items, current_index=0):
        super().__init__(config)
        self.menu_items = menu_items
        self.current_index = current_index

    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(4))
        height, width = win.getmaxyx()
        # Vertically center the menu block
        start_y = height // 2 - len(self.menu_items) // 2
        # Left-align the menu items at a fixed margin (e.g., x = 5)
        x = 5
        for idx, item in enumerate(self.menu_items):
            if idx == self.current_index:
                win.attron(curses.A_REVERSE)
                win.addstr(start_y + idx, x, item)
                win.attroff(curses.A_REVERSE)
            else:
                win.addstr(start_y + idx, x, item)
        win.attroff(curses.color_pair(4))

# --------------------------
# Layer Manager
# --------------------------
class LayerManager:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def draw_all(self, win):
        for layer in self.layers:
            layer.draw(win)

# --------------------------
# Scene Classes
# --------------------------
class Scene(ABC):
    def __init__(self, config):
        self.config = config
        self.layer_manager = LayerManager()

    @abstractmethod
    def handle_input(self, key):
        pass

    def render(self, win):
        self.layer_manager.draw_all(win)
        win.refresh()

class MenuScene(Scene):
    def __init__(self, config):
        super().__init__(config)
        self.menu_index = 0
        self.menu_items = config.menu_items
        self.layer_manager.add_layer(BackgroundLayer(config))
        self.layer_manager.add_layer(BorderLayer(config))
        self.layer_manager.add_layer(TitleLayer(config))
        self.menu_layer = MenuLayer(config, self.menu_items, self.menu_index)
        self.layer_manager.add_layer(self.menu_layer)
        self.selected = False
        self.selection = None

    def handle_input(self, key):
        if key in [ord('w'), curses.KEY_UP]:
            self.menu_index = (self.menu_index - 1) % len(self.menu_items)
        elif key in [ord('s'), curses.KEY_DOWN]:
            self.menu_index = (self.menu_index + 1) % len(self.menu_items)
        elif key in [ord(' '), curses.KEY_ENTER, 10, 13]:
            self.selected = True
            self.selection = self.menu_index
        elif key in [ord('1')]:
            self.selected = True
            self.selection = 0
        elif key in [ord('2')]:
            self.selected = True
            self.selection = 1
        elif key in [ord('3')]:
            self.selected = True
            self.selection = 2
        self.menu_layer.current_index = self.menu_index

# --------------------------
# Scene Manager
# --------------------------
class SceneManager:
    def __init__(self, initial_scene):
        self.current_scene = initial_scene

    def run(self, win):
        term_size = shutil.get_terminal_size()
        curses.resize_term(term_size.lines, term_size.columns)
        win.clear()
        win.nodelay(False)
        while True:
            self.current_scene.render(win)
            key = win.getch()
            self.current_scene.handle_input(key)
            if self.current_scene.selected:
                return self.current_scene.selection

# --------------------------
# Main Function
# --------------------------
def main(stdscr):
    # Initialize curses settings
    curses.noecho()           # Hide key echo
    curses.cbreak()           # Immediate key responses
    curses.curs_set(0)        # Hide cursor
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Background
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Border
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Title
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Menu

    config = Config()
    while True:
        scene = MenuScene(config)
        scene_manager = SceneManager(scene)
        selection = scene_manager.run(stdscr)
        output = ""
        try:
            if selection == 0:
                proc = subprocess.run(config.ngram_command, shell=True, capture_output=True, text=True)
            elif selection == 1:
                proc = subprocess.run(config.test_command, shell=True, capture_output=True, text=True)
            elif selection == 2:
                stdscr.clear()
                stdscr.addstr(0, 0, "Exiting. Press any key.")
                stdscr.refresh()
                stdscr.getch()
                break
            output = proc.stdout + "\n" + proc.stderr
        except Exception as e:
            output = "Error executing command:\n" + str(e)
        # Display the command output in an output screen that fits the window.
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(4))
        lines = output.splitlines()
        # Reserve last 2 lines for prompt.
        for idx, line in enumerate(lines[:height - 2]):
            stdscr.addnstr(idx, 0, line, width - 1)
        stdscr.attroff(curses.color_pair(4))
        stdscr.addnstr(height - 1, 0, "Press any key to return to the menu.", width - 1)
        stdscr.refresh()
        stdscr.getch()
        # Loop back to the menu

if __name__ == "__main__":
    import shutil
    import subprocess
    curses.wrapper(main)