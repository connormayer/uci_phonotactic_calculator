#!/usr/bin/env python3
"""
menu_ui.py - A curses-based UI for running the ngram calculator or unit tests,
with separate scenes for command output display and a loading screen.
Version: 1.3.0
"""

import curses
import shutil
import subprocess
import time
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
        start_y = height // 2 - len(self.menu_items) // 2
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
# New Layers for Output Scene
# --------------------------
class CustomTitleLayer(Layer):
    @Plugin.layer_decorator
    def draw(self, win):
        title = "Command Output"
        height, width = win.getmaxyx()
        y = 1  # Fixed row near the top
        x = max((width - len(title)) // 2, 0)
        win.attron(curses.color_pair(3) | curses.A_BOLD)
        win.addstr(y, x, title)
        win.attroff(curses.color_pair(3) | curses.A_BOLD)

class OutputLayer(Layer):
    def __init__(self, config, output, margin_x=2, margin_top=3):
        super().__init__(config)
        self.output = output
        self.margin_x = margin_x
        self.margin_top = margin_top

    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(4))
        height, width = win.getmaxyx()
        lines = self.output.splitlines()
        max_lines = height - self.margin_top - 1  # Reserve last line for prompt
        for idx, line in enumerate(lines[:max_lines]):
            win.addnstr(self.margin_top + idx, self.margin_x, line, width - self.margin_x - 1)
        win.attroff(curses.color_pair(4))

# --------------------------
# New Layer for Loading Scene
# --------------------------
class LoadingLayer(Layer):
    def __init__(self, config, loading_scene):
        super().__init__(config)
        self.loading_scene = loading_scene

    @Plugin.layer_decorator
    def draw(self, win):
        height, width = win.getmaxyx()
        spinner = self.loading_scene.spinner[self.loading_scene.spinner_index]
        message = "Loading... Please wait " + spinner
        x = max((width - len(message)) // 2, 0)
        y = height // 2
        win.attron(curses.color_pair(3) | curses.A_BOLD)
        win.addstr(y, x, message)
        win.attroff(curses.color_pair(3) | curses.A_BOLD)

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
        self.selected = False

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

class OutputScene(Scene):
    def __init__(self, config, output):
        super().__init__(config)
        self.output = output
        self.layer_manager.add_layer(BackgroundLayer(config))
        self.layer_manager.add_layer(BorderLayer(config))
        self.layer_manager.add_layer(CustomTitleLayer(config))
        self.layer_manager.add_layer(OutputLayer(config, output))
    
    def handle_input(self, key):
        if key != -1:
            self.selected = True

    def render(self, win):
        self.layer_manager.draw_all(win)
        height, width = win.getmaxyx()
        win.attron(curses.color_pair(4))
        prompt = "Press any key to return to the menu."
        win.addnstr(height - 1, 0, prompt, width - 1)
        win.attroff(curses.color_pair(4))
        win.refresh()

class LoadingScene(Scene):
    def __init__(self, config, process):
        super().__init__(config)
        self.process = process
        self.spinner = ['|', '/', '-', '\\']
        self.spinner_index = 0
        self.last_update = time.time()
        # Add background and border layers along with our custom LoadingLayer.
        self.layer_manager.add_layer(BackgroundLayer(config))
        self.layer_manager.add_layer(BorderLayer(config))
        self.layer_manager.add_layer(LoadingLayer(config, self))
    
    def handle_input(self, key):
        if self.process.poll() is not None:
            self.selected = True
        now = time.time()
        if now - self.last_update > 0.2:
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner)
            self.last_update = now

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
        while True:
            if isinstance(self.current_scene, LoadingScene):
                win.timeout(100)
            else:
                win.timeout(-1)
            self.current_scene.render(win)
            key = win.getch()
            self.current_scene.handle_input(key)
            if self.current_scene.selected:
                return getattr(self.current_scene, 'selection', None)

# --------------------------
# Main Function
# --------------------------
def main(stdscr):
    curses.noecho()           # Hide key echo
    curses.cbreak()           # Immediate key responses
    curses.curs_set(0)        # Hide cursor
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Background
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Border
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Title/Loading
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Menu/Output

    config = Config()
    while True:
        scene = MenuScene(config)
        scene_manager = SceneManager(scene)
        selection = scene_manager.run(stdscr)
        # Quit option
        if selection == 2:
            stdscr.clear()
            stdscr.addstr(0, 0, "Exiting. Press any key.")
            stdscr.refresh()
            stdscr.getch()
            break

        # Determine command based on selection
        command = config.ngram_command if selection == 0 else config.test_command
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Show loading scene until process completes
            loading_scene = LoadingScene(config, process)
            scene_manager = SceneManager(loading_scene)
            scene_manager.run(stdscr)
            stdout, stderr = process.communicate()
            output = stdout + "\n" + stderr
            # For ngram calculator, if no output is captured, try reading the CSV file.
            if selection == 0 and not output.strip():
                try:
                    with open("src/output.csv", "r") as f:
                        output = f.read()
                except Exception as e:
                    output = "No output available. Error reading output file: " + str(e)
        except Exception as e:
            output = "Error executing command:\n" + str(e)
        # Display the command output using the new OutputScene.
        output_scene = OutputScene(config, output)
        scene_manager = SceneManager(output_scene)
        scene_manager.run(stdscr)
        # Loop back to the menu

if __name__ == "__main__":
    curses.wrapper(main)