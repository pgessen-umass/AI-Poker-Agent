import os
import pprint

import torch
from pypokerengine.api.game import setup_config, start_poker
from q.game_encoder import PokerGameEncoder
from q.model import DeepQModelWrapper, QModel
from q.training_player import QPlayer

def getLatestModel(folder, hidden_layer_dimensions=256):
    if(os.path.exists(folder)):
        with open(f"{folder}/metadata.txt", "r") as f:
            hidden_layer_dimensions = int(f.read())
        
        model = QModel(hidden_layer_dimensions)
        available_models = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
        if(len(available_models) > 0):
            latest_model = available_models[-1]
            latest_state_dict = torch.load(f"{folder}/{latest_model}")
            model.load_state_dict(latest_state_dict)
    else:
        os.makedirs(folder)
        with open(f"{folder}/metadata.txt", "w") as f:
            f.write(str(hidden_layer_dimensions))
        model = QModel(hidden_layer_dimensions)

    return model

def cleanUpFolder(folder):
    if(os.path.exists(folder)):
        available_models = sorted([f for f in os.listdir(folder) if f.endswith('.pt')])
        if(len(available_models) > 5):
            keep = available_models[-5:]
            for f in [f for f in os.listdir(folder) if f.endswith('.pt')]:
                if(f not in keep): os.remove(f"{folder}/{f}")

swap = False
while True:
    config = setup_config(max_round=1000, initial_stack=10000, small_blind_amount=10)

    model1 = getLatestModel("model_1", 256)
    wrapper1 = DeepQModelWrapper(model1, 0.9, 0.2, "model_1", 100, 1000)

    model2 = getLatestModel("model_2", 128)
    wrapper2 = DeepQModelWrapper(model2, 0.9, 0.2, "model_2", 100, 1000)

    first = wrapper1 if swap else wrapper2
    second = wrapper2 if swap else wrapper1

    p1 = QPlayer(first, training=True, verbose=False)
    p2 = QPlayer(second, training=True, verbose=False)

    # p2 = DeepQPlayer(counter, PlayerType.RESETTER)

    # config.register_player(name="f1", algorithm=RandomPlayer())
    # config.register_player(name="FT2", algorithm=RaisedPlayer())
    config.register_player(name="DeepQ Player1", algorithm=p1)
    config.register_player(name="DeepQ Player2", algorithm=p2)

    game_result = start_poker(config, verbose=0)

    cleanUpFolder("model_1")
    cleanUpFolder("model_2")

    swap = not swap
    pprint.pprint(game_result)