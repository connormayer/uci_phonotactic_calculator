#!/usr/bin/env python3
"""
menu_ui.py - A curses-based UI for running the ngram calculator or unit tests.
This module provides a multi-scene curses UI to select between running the ngram calculator
or executing the unit tests.
Version: 1.3.0
"""

import curses
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
import functools

# --------------------------
# Config Class
# --------------------------
class Config:
    """
    Configuration class that holds UI settings such as colors, commands, menu items, and layout constants.
    """
    def __init__(self):
        self.background_color = curses.COLOR_BLACK
        self.border_color = curses.COLOR_GREEN
        self.menu_color = curses.COLOR_WHITE
        self.title_color = curses.COLOR_RED
        # Updated commands to run the new system
        self.ngram_command = "python -m src.ngram_calculator data/english.csv data/sample_test_data/english_test_data.csv src/output.csv"
        self.test_command = "python webcalc/tests_updated.py"
        self.menu_items = ["1. Ngram Calculator", "2. Unit Tests", "3. Quit"]
        # Layout constants
        self.menu_x_offset = 5
        self.output_margin_x = 2
        self.output_margin_top = 3

# --------------------------
# Plugin Class and Decorator
# --------------------------
class Plugin:
    """
    Plugin class providing a decorator to wrap layer draw methods for potential logging or additional functionality.
    """
    @staticmethod
    def layer_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Additional processing or logging can be added here.
            return func(*args, **kwargs)
        return wrapper

# --------------------------
# Layer Classes
# --------------------------
class Layer(ABC):
    """
    Abstract base class for all UI layers. All subclasses must implement the draw() method.
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def draw(self, win):
        """
        Draw the layer on the given curses window.
        """
        pass

class BackgroundLayer(Layer):
    """
    Layer that draws the UI background.
    """
    @Plugin.layer_decorator
    def draw(self, win):
        win.bkgd(' ', curses.color_pair(1))
        win.erase()

class BorderLayer(Layer):
    """
    Layer that draws a border around the UI.
    """
    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(2))
        win.box()
        win.attroff(curses.color_pair(2))

class TitleLayer(Layer):
    """
    Layer that displays the main title of the UI.
    """
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
    """
    Layer that draws the interactive menu items and highlights the current selection.
    """
    def __init__(self, config, menu_items, current_index=0):
        super().__init__(config)
        self.menu_items = menu_items
        self.current_index = current_index

    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(4))
        height, width = win.getmaxyx()
        start_y = height // 2 - len(self.menu_items) // 2
        x = self.config.menu_x_offset  # Use configurable x-offset
        for idx, item in enumerate(self.menu_items):
            if idx == self.current_index:
                win.attron(curses.A_REVERSE)
                win.addstr(start_y + idx, x, item)
                win.attroff(curses.A_REVERSE)
            else:
                win.addstr(start_y + idx, x, item)
        win.attroff(curses.color_pair(4))

class CustomTitleLayer(Layer):
    """
    Layer that displays a custom title for output scenes.
    """
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
    """
    Layer that displays text output with configurable margins.
    """
    def __init__(self, config, output):
        super().__init__(config)
        self.output = output
        # Get margins from configuration
        self.margin_x = config.output_margin_x
        self.margin_top = config.output_margin_top

    @Plugin.layer_decorator
    def draw(self, win):
        win.attron(curses.color_pair(4))
        height, width = win.getmaxyx()
        lines = self.output.splitlines()
        max_lines = height - self.margin_top - 1  # Reserve last line for prompt
        for idx, line in enumerate(lines[:max_lines]):
            win.addnstr(self.margin_top + idx, self.margin_x, line, width - self.margin_x - 1)
        win.attroff(curses.color_pair(4))

class LoadingLayer(Layer):
    """
    Layer that displays a loading spinner and message.
    """
    def __init__(self, config, loading_scene):
        super().__init__(config)
        self.loading_scene = loading_scene

    @Plugin.layer_decorator
    def draw(self, win):
        height, width = win.getmaxyx()
        spinner = self.loading_scene.spinner[self.loading_scene.spinner_index]
        message = f"Loading... Please wait {spinner}"
        x = max((width - len(message)) // 2, 0)
        y = height // 2
        win.attron(curses.color_pair(3) | curses.A_BOLD)
        win.addstr(y, x, message)
        win.attroff(curses.color_pair(3) | curses.A_BOLD)

# --------------------------
# Layer Manager
# --------------------------
class LayerManager:
    """
    Manages a list of layers and renders them in order.
    """
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
    """
    Abstract base class for UI scenes. Each scene is composed of layers and handles its own input.
    """
    def __init__(self, config):
        self.config = config
        self.layer_manager = LayerManager()
        self.selected = False

    @abstractmethod
    def handle_input(self, key):
        """
        Handle input key events for the scene.
        """
        pass

    def render(self, win):
        """
        Render all layers in the scene.
        """
        self.layer_manager.draw_all(win)
        win.refresh()

class BaseScene(Scene):
    """
    Base scene that automatically adds common layers (background and border) to every scene.
    """
    def __init__(self, config):
        super().__init__(config)
        self.layer_manager.add_layer(BackgroundLayer(config))
        self.layer_manager.add_layer(BorderLayer(config))

class MenuScene(BaseScene):
    """
    Scene that presents the main menu with selectable items.
    """
    def __init__(self, config):
        super().__init__(config)
        self.current_index = 0  # Standardized variable for the current selection index
        self.menu_items = config.menu_items
        self.layer_manager.add_layer(TitleLayer(config))
        self.menu_layer = MenuLayer(config, self.menu_items, self.current_index)
        self.layer_manager.add_layer(self.menu_layer)
        self.selection = None

    def handle_input(self, key):
        """
        Handle navigation and selection input for the menu.
        """
        if key in [ord('w'), curses.KEY_UP]:
            self.current_index = (self.current_index - 1) % len(self.menu_items)
        elif key in [ord('s'), curses.KEY_DOWN]:
            self.current_index = (self.current_index + 1) % len(self.menu_items)
        elif key in [ord(' '), curses.KEY_ENTER, 10, 13]:
            self.selected = True
            self.selection = self.current_index
        elif key in [ord('1')]:
            self.selected = True
            self.selection = 0
        elif key in [ord('2')]:
            self.selected = True
            self.selection = 1
        elif key in [ord('3')]:
            self.selected = True
            self.selection = 2
        self.menu_layer.current_index = self.current_index

class OutputScene(BaseScene):
    """
    Scene that displays the output from a command execution.
    """
    def __init__(self, config, output):
        super().__init__(config)
        self.output = output
        self.layer_manager.add_layer(CustomTitleLayer(config))
        self.layer_manager.add_layer(OutputLayer(config, output))
    
    def handle_input(self, key):
        """
        Exit the output scene on any key press.
        """
        if key != -1:
            self.selected = True

    def render(self, win):
        """
        Render the output scene and display a prompt at the bottom.
        """
        self.layer_manager.draw_all(win)
        height, width = win.getmaxyx()
        win.attron(curses.color_pair(4))
        prompt = "Press any key to return to the menu."
        win.addnstr(height - 1, 0, prompt, width - 1)
        win.attroff(curses.color_pair(4))
        win.refresh()

class LoadingScene(BaseScene):
    """
    Scene that displays a loading animation while a subprocess command is running.
    """
    def __init__(self, config, process):
        super().__init__(config)
        self.process = process
        self.spinner = ['|', '/', '-', '\\']
        self.spinner_index = 0
        self.last_update = time.time()
        self.layer_manager.add_layer(LoadingLayer(config, self))
    
    def handle_input(self, key):
        """
        Update the spinner animation and check if the subprocess has finished.
        """
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
    """
    Manages the active scene, handles input, and triggers rendering.
    """
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
# Command Execution Helper
# --------------------------
def execute_command(command, selection, config, stdscr):
    """
    Execute a command using subprocess, display a loading scene,
    capture output, and fallback to reading CSV output if needed.
    """
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Display loading scene while command executes
        loading_scene = LoadingScene(config, process)
        scene_manager = SceneManager(loading_scene)
        scene_manager.run(stdscr)
        stdout, stderr = process.communicate()
        output = f"{stdout}\n{stderr}"
        # For ngram calculator, if no output is captured, fallback to CSV file output.
        if selection == 0 and not output.strip():
            try:
                with open("src/output.csv", "r") as f:
                    output = f.read()
            except Exception as e:
                output = f"No output available. Error reading output file: {e}"
    except Exception as e:
        output = f"Error executing command:\n{e}"
    return output

# --------------------------
# Main Function
# --------------------------
def main(stdscr):
    """
    Main function to initialize curses and start the UI loop.
    """
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
        # Quit option selected
        if selection == 2:
            stdscr.clear()
            stdscr.addstr(0, 0, "Exiting. Press any key.")
            stdscr.refresh()
            stdscr.getch()
            break

        # Determine command based on the user's selection
        command = config.ngram_command if selection == 0 else config.test_command
        output = execute_command(command, selection, config, stdscr)
        # Display the command output
        output_scene = OutputScene(config, output)
        scene_manager = SceneManager(output_scene)
        scene_manager.run(stdscr)
        # Loop back to the menu

if __name__ == "__main__":
    curses.wrapper(main)