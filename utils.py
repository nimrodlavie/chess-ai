from IPython.display import SVG, display, clear_output
import chess.svg
from lxml import etree
from PIL import Image, ImageTk
from svglib.svglib import SvgRenderer
from reportlab.graphics import renderPM
from io import BytesIO
import tkinter as tk
import re
import os
import sys
import threading
import logging
import time


NOTEBOOK = False


class Visualizer(tk.Frame):

    canvas = None
    visualizer = None
    image = None

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()


def open_tk_visualizer_in_background(initial_board):
    assert Visualizer.visualizer is None

    vis = Visualizer()
    Visualizer.visualizer = vis
    vis.master.title('Visualizer')
    canvas = tk.Canvas(vis, width=530, height=530)
    canvas.pack()
    Visualizer.canvas = canvas

    display_board_on_tk_visualizer(initial_board)

    Visualizer.visualizer.mainloop()


def display_board_in_notebook(board):
    # First clear the previous output:
    clear_output()
    svg = re.sub("viewBox=\"0 0 (.*?)\"", "viewbox=\"0 0 1600 1600\"", chess.svg.board(board, orientation=board.turn))
    display(SVG(svg))


def display_board_on_tk_visualizer(board):
    # assert Visualizer.visualizer is not None

    svg_root = etree.fromstring(chess.svg.board(board, orientation=board.turn, size=1100))
    svg_renderer = SvgRenderer("DummyPath")
    svgfile = svg_renderer.render(svg_root)
    bytespng = BytesIO()
    # TODO - this line prints/logs some unwanted stuff:
    renderPM.drawToFile(svgfile, bytespng, bg=0x393A4C, fmt="PNG")
    img = Image.open(bytespng)
    img = img.resize((500, 500))
    Visualizer.canvas.pack()
    img = ImageTk.PhotoImage(img)
    Visualizer.canvas.delete()
    Visualizer.canvas.create_image(20, 20, anchor=tk.NW, image=img)
    Visualizer.image = img


def display_board(board):

    if NOTEBOOK:
        display_board_in_notebook(board)
        return

    display_board_on_tk_visualizer(board)


def outcome_to_numeric(outcome):
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner else -1


