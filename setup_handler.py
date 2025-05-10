import os
import torch
from q.game_encoder import PokerGameEncoder
from q.model import DeepQModelWrapper, QModel

int_to_action = {
    0: "raise",
    1: "call",
    2: "fold"
}

action_to_int = {
    "raise": 0,
    "call": 1,
    "fold": 2
}

def get_valid_actions_mask(valid_actions):
    out = torch.tensor([False, False, False])
    for a in valid_actions:
        out[action_to_int[a['action']]] = True
    return out

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def loadModel(folder):

    with open(f"{folder}/metadata.txt","r") as f:
        hidden_layer_size, number_of_hidden_layers = list(map(int,f.read().split(",")))
    
    model = QModel(hidden_layer_size, number_of_hidden_layers)
    models_available = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if(len(models_available) != 1): raise ValueError(f"Bad folder structure: {folder}")
    state_dict = torch.load(f"{folder}/{models_available[0]}")
    model.load_state_dict(state_dict)

    return model

@singleton
class PlayerTools:

    def __init__(self):
        self.model = loadModel("./best_model")
        self.model_wrapper = DeepQModelWrapper(self.model)
        self.classencoder = None

    def chooseAction(self, valid_actions, hole_card, round_state, algorithm):
        if(self.classencoder is None or self.classencoder.our_uuid != algorithm.uuid):
            seats = round_state['seats']
            self_uuid = algorithm.uuid
            op_uuid = [p['uuid'] for p in seats if p['uuid'] != self_uuid][0]
            self_stack = next(filter(lambda s: s['uuid'] == self_uuid, seats))['stack']
            op_stack = next(filter(lambda s: s['uuid'] == op_uuid, seats))['stack']
            self.classencoder = PokerGameEncoder(self_uuid, self_stack, op_uuid, op_stack, seats)
        
        valid_actions_mask = get_valid_actions_mask(valid_actions)
        self.classencoder.update(round_state, hole_card)
        s, op_histories = self.classencoder.get_features_as_tensor()
        s1, _ = self.classencoder.get_features()
        return int_to_action[self.model_wrapper.deployed_choose_action(s, 0, op_histories, s1, round_state, hole_card, return_action=True, valid_actions=valid_actions_mask)]