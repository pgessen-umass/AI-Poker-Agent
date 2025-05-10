from functools import reduce
import os
import pprint

from matplotlib import pyplot as plt
import torch
from model_evaluator import Evaluator
from pypokerengine.api.game import setup_config, start_poker
from q.game_encoder import PokerGameEncoder
from q.model import DeepQModelWrapper, QModel
from q.training_player import QPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer
from custom_player import setup_ai

def getLatestModel(folder, hidden_layer_dimensions=256, num_hidden_layers=3):
    if(os.path.exists(folder)):
        with open(f"{folder}/metadata.txt", "r") as f:
            hidden_layer_dimensions = int(f.read())
        
        model = QModel(hidden_layer_dimensions,num_hidden_layers)
        available_models = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
        if(len(available_models) > 0):
            latest_model = available_models[-1]
            latest_state_dict = torch.load(f"{folder}/{latest_model}")
            model.load_state_dict(latest_state_dict)
    else:
        os.makedirs(folder)
        with open(f"{folder}/metadata.txt", "w") as f:
            f.write(str(hidden_layer_dimensions))
        model = QModel(hidden_layer_dimensions,num_hidden_layers)

    return model

def cleanUpFolder(folder):
    if(os.path.exists(folder)):
        available_models = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
        if(len(available_models) > 5):
            keep = available_models[-5:]
            for f in [f for f in os.listdir(folder) if f.endswith('.pt')]:
                if(f not in keep): os.remove(f"{folder}/{f}")

eval = Evaluator("q_against_q")

PLAYER_ONE_NAME = "ONE"
PLAYER_TWO_NAME = "TWO"

eval.register_player(PLAYER_ONE_NAME)
eval.register_player(PLAYER_TWO_NAME)

def get_player_who_won(players):
    p1 = next(filter(lambda x: x['name'] == PLAYER_ONE_NAME, players))
    p2 = next(filter(lambda x: x['name'] == PLAYER_TWO_NAME, players))

    if(p1["stack"] > p2["stack"]): return PLAYER_ONE_NAME
    return PLAYER_TWO_NAME

g = 1000
i=0
m1History = None
m2History = None
history1saved = False
history2saved = False
while i<g:
    i+=1
    config = setup_config(max_round=200, initial_stack=1000, small_blind_amount=10)

    model1 = getLatestModel("ub_model_1", 256,2)
    wrapper1 = DeepQModelWrapper(model1, 0.9, 0.2, "ub_model_1", 100, 100000, learning_rate=1e-3, history=m1History)
    p1 = QPlayer(wrapper1, training=False, verbose=False)
    config.register_player(name=PLAYER_ONE_NAME, algorithm=p1)

    # model2 = getLatestModel("ub_model_2", 256,3)
    # wrapper2 = DeepQModelWrapper(model2, 0.9, 0.2, "ub_model_2", 100, 100000, learning_rate=1e-7, history=m1History)
    # p2 = QPlayer(wrapper2, training=False, verbose=False)
    # config.register_player(name=PLAYER_TWO_NAME, algorithm=p2)

    # p3 = RaisedPlayer()
    # config.register_player(name=PLAYER_ONE_NAME, algorithm=p3)
    # p4 = RaisedPlayer()
    # config.register_player(name=PLAYER_TWO_NAME, algorithm=p4)

    p5 = setup_ai()
    config.register_player(name=PLAYER_TWO_NAME, algorithm=p5)

    game_result, rounds_played = start_poker(config, verbose=0)

    winner = get_player_who_won(game_result['players'])
    eval.register_win(winner, toConsole=True)

    # m1History = wrapper1.history
    # m2History = wrapper2.history

    # if(not history1saved and len(m1History) == 20000):
    #     wrapper1.dump(history=True)
    #     history1saved = True
    # else:
    #     wrapper1.dump()
    
    # if(not history2saved and len(m2History) == 20000):
    #     wrapper2.dump(history=True)
    #     history2saved = True
    # else:
    #     wrapper2.dump()

    # cleanUpFolder("model_1_clean")
    # cleanUpFolder("model_2_clean")
    # cleanUpFolder("model_3")

    # pprint.pprint(game_result)


# PLAYER_ONE_NAME = "ONE"
# PLAYER_TWO_NAME = "TWO"

# # plt.subplot(1,2,1)
# plt.title("Loss vs. Number of Samples Trained On")
# model1 = getLatestModel("model_1", 256,1)
# wrapper1 = DeepQModelWrapper(model1, dump_folderpath='model_1')
# wrapper1.plot_loss( show=False, label="1e-5 Learning Rate, 1 hidden layer")

# # plt.subplot(1,2,2)
# # plt.title("1e-3 Learning Rate")
# model2 = getLatestModel("model_2", 256,2)
# wrapper2 = DeepQModelWrapper(model2, dump_folderpath='model_2')
# wrapper2.plot_loss(show=False, label="1e-3 Learning Rate, 2 hidden layers")
# plt.xlabel("Number of Samples Trained On")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# model2 = getLatestModel("model_3", 256,1)
# wrapper2 = DeepQModelWrapper(model2, dump_folderpath='model_3')
# wrapper2.plot_loss(exponential_decay_fit=True, show=False,label=PLAYER_TWO_NAME)
# plt.show()

# plt.subplot(1,2,2)
Evaluator.plot_file('q_against_q')