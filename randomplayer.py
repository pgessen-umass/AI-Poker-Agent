from pypokerengine.players import BasePokerPlayer
import random as rand
from game_encoder.game_encoder import PokerGameEncoder
import pprint

class RandomPlayer(BasePokerPlayer):

  classencoder = None

  def declare_action(self, valid_actions, hole_card, round_state):
    self.classencoder.update(round_state, hole_card)
    pp = pprint.PrettyPrinter(indent=2)
    print("------------ROUND_STATE(RANDOM)--------")
    pp.pprint(round_state)
    print("------------HOLE_CARD----------")
    pp.pprint(hole_card)
    print("------------VALID_ACTIONS----------")
    pp.pprint(valid_actions)
    print("------------FEATURES----------")
    pp.pprint(self.classencoder.get_features())
    print("-------------------------------")
    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    return action  # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    seats = game_info.get('seats')
    for seat in seats:
      if seat.get('name') == 'Random Warrior 1':
        self_uuid = seat.get('uuid')
        self_stack = seat.get('stack')
      else:
        op_uuid = seat.get('uuid')
        op_stack = seat.get('stack')
    self.classencoder = PokerGameEncoder(self_uuid, self_stack, op_uuid, op_stack, game_info)

  def receive_round_start_message(self, round_count, hole_card, seats):
    pp = pprint.PrettyPrinter(indent=2)
    print("------------NEW_ROUND_STATE(RANDOM)--------")
    pp.pprint(seats)

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return RandomPlayer()
