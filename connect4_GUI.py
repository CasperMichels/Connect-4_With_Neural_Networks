import tkinter as tk
import glob
from connect4 import *

beige_hex = "#fff1d0"
blue_hex = "#086788"
dark_blue_hex = "#06517d"
disabled_blue_hex = "#64848f"
red_hex = "#cc0000"
dark_red_hex = "#ba002f"
yellow_hex = "#f0c908"
dark_yellow_hex = "#e39e09"

root = tk.Tk()
root.title("Connect-4")
root.geometry("500x300+50+50")
canvas = tk.Canvas(root, height=300, width=500, bg=beige_hex)
canvas.pack()

player_options = ["human", "random", "minmax"]
game_started = False

heuristics_options = {
    "random": "random"}
for file in glob.glob("*.npy"):
    heuristics_options[file] = numpy.load(file, allow_pickle=True).tolist()


class PlayerSelection:
    def __init__(self, x, y, colour, active_colour, name):
        self.x = x
        self.y = y

        self.display = tk.Label(root)
        self.display.config(text=name, bg=colour)
        self.display.place(relx=self.x, rely=self.y - 0.1, anchor="se")

        self.player = tk.StringVar(root)
        self.player.set("human")
        self.player.trace("w", self.additional_options)
        self.player_menu = tk.OptionMenu(root, self.player, *player_options)
        self.player_menu.config(bg=colour, activebackground=active_colour, highlightthickness=0)
        self.player_menu.place(relx=self.x, rely=self.y, anchor="se")

        self.heuristic = tk.StringVar(root)
        self.heuristic.set("random")
        self.heuristic_menu = tk.OptionMenu(root, self.heuristic, *heuristics_options.keys())
        self.heuristic_menu.config(bg=colour, activebackground=active_colour, highlightthickness=0)

        self.depth = tk.Spinbox(root, from_=1, to=42, width=5)
        self.depth.config(bg=colour, buttonbackground=colour, activebackground=active_colour)

    def additional_options(self, *args):
        if self.player.get() == "minmax":
            self.heuristic_menu.place(relx=self.x, rely=self.y, anchor="ne")
            self.depth.place(relx=self.x, rely=self.y, anchor="sw")
        else:
            self.heuristic_menu.place_forget()
            self.depth.place_forget()


# create two menus to select the players
player_1 = PlayerSelection(0.35, 0.4, yellow_hex, dark_yellow_hex, "PLAYER 1")

player_2 = PlayerSelection(0.8, 0.4, red_hex, dark_red_hex, "PLAYER 2")


# button that starts the game
def on_start_button():
    start_button.config(bg=disabled_blue_hex, state="disabled")
    start_button.update()
    heuristic_1 = heuristics_options[player_1.heuristic.get()]
    heuristic_2 = heuristics_options[player_2.heuristic.get()]
    depth_1 = int(player_1.depth.get())
    depth_2 = int(player_2.depth.get())
    play_game(main_window, main_board.copy(),
              player_1.player.get(), player_2.player.get(),
              heuristic_1=heuristic_1, heuristic_2=heuristic_2,
              nn1=heuristic_1, nn2=heuristic_2,
              depth_1=depth_1, depth_2=depth_2)
    start_button.update()
    start_button.config(bg=blue_hex, state="normal")


def on_start_button_hover(button):
    start_button.config(bg=dark_blue_hex)


def on_start_button_leave(button):
    start_button.config(bg=blue_hex)


title = tk.Label(root)
title.config(text="CONNECT-4", bg=beige_hex, font=("Arial", 30))
title.place(relx=0.5, rely=0.1, anchor="center")

start_button = tk.Button(root, text="START GAME",
                         bg=blue_hex, activebackground=dark_blue_hex,
                         command=on_start_button)
start_button.bind("<Enter>", on_start_button_hover)
start_button.bind("<Leave>", on_start_button_leave)
start_button.place(relx=0.5, rely=0.6, anchor="center")

root.mainloop()