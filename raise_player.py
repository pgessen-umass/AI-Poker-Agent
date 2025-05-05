from pypokerengine.players import BasePokerPlayer
from time import sleep
from q.game_encoder import PokerGameEncoder
import pprint

class RaisedPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    pp = pprint.PrettyPrinter(indent=2)
    self.classencoder.update(round_state, hole_card)
    print("------------ROUND_STATE(RANDOM)--------")
    pp.pprint(round_state)
    print("------------HOLE_CARD----------")
    pp.pprint(hole_card)
    print("------------VALID_ACTIONS----------")
    pp.pprint(valid_actions)
    print("--------------ENCODER-----------------")
    pp.pprint(self.classencoder.get_features()[0])
    pp.pprint(self.classencoder.get_features()[1])
    for i in valid_actions:
        if i["action"] == "raise":
            action = i["action"]
            return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

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

    self.classencoder = PokerGameEncoder(self_uuid, self_stack, op_uuid, op_stack, game_info)

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return RandomPlayer()
