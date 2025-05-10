from time import perf_counter
from q.game_encoder import PokerGameEncoder
from q.model import DeepQModelWrapper, QModel
from pypokerengine.players import BasePokerPlayer
from q.transforms import encode
import torch

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

class QPlayer(BasePokerPlayer):

    def __init__(self, model_wrapper: DeepQModelWrapper, training=False, verbose = False):
        super().__init__()
        
        self.classencoder = None
        self.model_wrapper = model_wrapper
        self.training = training
        self.verbose = verbose
        self.round_count = 0
        
    def declare_action(self, valid_actions, hole_card, round_state):
        # start = perf_counter()
        valid_actions_mask = get_valid_actions_mask(valid_actions)
        self.classencoder.update(round_state, hole_card)

        s, op_histories = self.classencoder.get_features_as_tensor()
        s1, _ = self.classencoder.get_features()

        if(self.training): 
            next_action = self.model_wrapper.register_current_state(s, 0, op_histories, s1, round_state, hole_card, return_action=True, valid_actions=valid_actions_mask)
            self.model_wrapper.optimize()
        else:
            next_action = self.model_wrapper.deployed_choose_action(s, 0, op_histories, s1, round_state, hole_card, return_action=True, valid_actions=valid_actions_mask)

        for i in valid_actions:
            if i["action"] == int_to_action[next_action]:
                action = i["action"]
                # end = perf_counter()
                # print(end-start)
                return action  # action returned here is sent to the poker engine
        
        assert False, f"Action chosen is not a valid action! Chosen: {next_action}. Available: {valid_actions}. Mask: {valid_actions_mask}"

    def receive_game_start_message(self, game_info):
        seats = game_info.get('seats')
        one = False
        two = False
        for seat in seats:
            if seat.get('uuid') == self.uuid:
                self_uuid = seat.get('uuid')
                self_stack = seat.get('stack')
                one = True
            else:
                op_uuid = seat.get('uuid')
                op_stack = seat.get('stack')
                two = True

        assert(one and two, "Invalid initialization of PokerGameEncoder. Check receive_game_start_message function") # Temporary - checking assignment of both

        self.classencoder = PokerGameEncoder(self_uuid, self_stack, op_uuid, op_stack, game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count += 1
        if(self.verbose): print(f"Round {self.round_count} started")
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if(self.verbose): print(f"Round {self.round_count} ended")
        # print("hand info:", hand_info)
        self.classencoder.update(round_state)
        s, op_histories = self.classencoder.get_features_as_tensor()
        s1, _ = self.classencoder.get_features()
        
        if(len(winners) != 1):
            # Tie occured
            reward = 0
        else:
            # Otherwise zero-sum game since 
            won = winners[0]['uuid'] == self.uuid
            sign = 1 if won else -1
            reward = sign*round_state['pot']['main']['amount']
        
        if(self.training): self.model_wrapper.register_current_state(s,reward/1000,op_histories,s1,round_state,None)