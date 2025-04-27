from pypokerengine.players import BasePokerPlayer
from time import sleep
from game_encoder.game_encoder import PokerGameEncoder
import pprint

class RaisedPlayer(BasePokerPlayer):

  classencoder = None

  def declare_action(self, valid_actions, hole_card, round_state):
    self.classencoder.update(round_state, hole_card)
    pp = pprint.PrettyPrinter(indent=2)
    print("------------ROUND_STATE(RAISE)--------")
    pp.pprint(round_state)
    print("------------HOLE_CARD----------")
    pp.pprint(hole_card)
    print("------------VALID_ACTIONS----------")
    pp.pprint(valid_actions)
    print("------------FEATURES----------")
    pp.pprint(self.classencoder.get_features())
    print("-------------------------------")
    for i in valid_actions:
        if i["action"] == "raise":
            action = i["action"]
            return action
    action = valid_actions[1]["action"]
    return action

  def receive_game_start_message(self, game_info):
    seats = game_info.get('seats')
    for seat in seats:
      if seat.get('name') == 'RaisedPlayer':
        self_uuid = seat.get('uuid')
        self_stack = seat.get('stack')
      else:
        op_uuid = seat.get('uuid')
        op_stack = seat.get('stack')
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
