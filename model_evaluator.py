import json
import os
import time
from matplotlib import pyplot as plt

class Evaluator:

    def __init__(self, dumpfolder: str):
        self.player_wins: dict[str, list[int]] = {}
        self.game_number = 0
        self.dumpfolder = dumpfolder
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.file = f"{self.dumpfolder}/adversarial_win_ratio-{ts}.json"

    def register_player(self, name):
        self.player_wins[name] = []
        print(self.player_wins)

    def register_win(self, name):
        self.game_number += 1
        self.player_wins[name].append(self.game_number)
        self.dump()
        print(self.player_wins)

    def dump(self):
        os.makedirs(self.dumpfolder, exist_ok=True)
        with open(self.file,"w") as f:
            json.dump(self.player_wins, f)

    def create_plots(self, show = True):
        Evaluator._create_plots(self.player_wins, show)

    @staticmethod
    def _create_plots(player_wins, show):
        game_number = -1

        for k,v in player_wins.items():
            game_number = max(game_number, max(v))
        
        games = range(1,game_number+1)

        p1_wins = []
        p2_wins = []
        p1_name = list(player_wins.keys())[0]
        p2_name = list(player_wins.keys())[1]

        for g in games:
            p1_wins.append(
                len(list(filter(lambda x: x < g, player_wins[p1_name])))
            )
            p2_wins.append(
                len(list(filter(lambda x: x < g, player_wins[p2_name])))
            )

        plt.title("Wins vs. Games Played")
        plt.plot(games, p1_wins, label=p1_name)
        plt.plot(games, p2_wins, label=p2_name)
        plt.ylabel("Wins")
        plt.xlabel("Games Played")
        plt.legend()

        if(show): plt.show()

    @staticmethod
    def plot_file(filepath, show=True):
        with open(filepath,"r") as f:
            player_wins = json.load(f)
        Evaluator._create_plots(player_wins, show)