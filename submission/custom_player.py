from pypokerengine.players import BasePokerPlayer
from setup_handler import PlayerTools

class CustomPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()

    def declare_action(self, valid_actions, hole_card, round_state):
        return PlayerTools().chooseAction(valid_actions, hole_card, round_state, self)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CustomPlayer()