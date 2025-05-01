import pprint
import torch

class PokerGameEncoder:
    _suit_offset = {'C': 0, 'D': 13, 'H': 26, 'S': 39}
    _rank_value = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
    _pre_rounds = {'river': 'turn', 'turn': 'flop', 'flop': 'preflop'}

    def __init__(self, our_uuid, our_stack, op_uuid, op_stack, game_info):
        self.our_uuid = our_uuid
        self.opponent_uuid = op_uuid
        self.hole_cards_vector = torch.zeros(52, dtype=torch.float32)
        self.community_cards_vector = torch.zeros(52, dtype=torch.float32)
        self.our_total_money = our_stack
        self.opponent_total_money = op_stack
        self.our_investment = 0
        self.opponent_investment = 0
        self.round_count = 1
        self.seats = game_info.get('seats')
        self.prev_our_pot = our_stack
        self.prev_op_pot = op_stack

    @classmethod
    def card_to_index(cls, card):
        return cls._suit_offset[card[0]] + cls._rank_value[card[1]]

    @classmethod
    def encode_cards(cls, cards):
        vec = torch.zeros(52, dtype=torch.float32)
        for card in cards:
            vec[cls.card_to_index(card)] = 1.0
        return vec

    def get_stack(self, uuid, new_game_data):
        for seat in new_game_data.get('seats'):
            if seat['uuid'] == uuid:
                return seat['stack']
        raise ValueError(f"UUID {uuid} not found in seats")

    def update(self, new_game_data, new_hole_cards=None):
        if new_hole_cards:
            self.hole_cards_vector = self.encode_cards(new_hole_cards)
        self.community_cards_vector = self.encode_cards(new_game_data.get('community_card', []))
        
        if new_game_data['round_count'] != self.round_count:
            self.round_count = new_game_data['round_count']
            small_blind_pos = new_game_data['small_blind_pos']
            small_blind_amount = new_game_data['small_blind_amount']

            small_blind_uuid = new_game_data['seats'][small_blind_pos]['uuid']

            if new_game_data.get('action_histories').get('preflop')[-1].get('action') == 'BIGBLIND':
                if self.our_uuid == small_blind_uuid:
                    self.our_investment = small_blind_amount
                    self.opponent_investment = 2 * small_blind_amount
                    self.prev_our_pot = self.get_stack(self.our_uuid, new_game_data) + small_blind_amount
                    self.prev_op_pot = self.get_stack(self.opponent_uuid, new_game_data) + 2 * small_blind_amount
                    self.our_total_money = self.get_stack(self.our_uuid, new_game_data)
                    self.opponent_total_money = self.get_stack(self.opponent_uuid, new_game_data)
                else:
                    self.our_investment = 2 * small_blind_amount
                    self.opponent_investment = small_blind_amount
                    self.prev_our_pot = self.get_stack(self.our_uuid, new_game_data) + 2 * small_blind_amount
                    self.prev_op_pot = self.get_stack(self.opponent_uuid, new_game_data) + small_blind_amount
                    self.our_total_money = self.get_stack(self.our_uuid, new_game_data)
                    self.opponent_total_money = self.get_stack(self.opponent_uuid, new_game_data)
            else:
                last_opponent_paid = new_game_data.get('action_histories').get('preflop')[-1].get('paid')

                # TODO: CHECK THIS PLEASE. Does this logic make sense. Was getting error. Seems like would occur when opponent would
                # Fold in preflop round 
                if(last_opponent_paid is None): last_opponent_paid = 0

                if self.our_uuid == small_blind_uuid:
                    self.our_investment = small_blind_amount
                    self.opponent_investment = 2 * small_blind_amount + last_opponent_paid
                    self.prev_our_pot = self.get_stack(self.our_uuid, new_game_data) + small_blind_amount
                    self.prev_op_pot = self.get_stack(self.opponent_uuid, new_game_data) + 2 * small_blind_amount + last_opponent_paid
                    self.our_total_money = self.get_stack(self.our_uuid, new_game_data)
                    self.opponent_total_money = self.get_stack(self.opponent_uuid, new_game_data)
                else:
                    self.our_investment = 2 * small_blind_amount
                    self.opponent_investment = small_blind_amount + last_opponent_paid
                    self.prev_our_pot = self.get_stack(self.our_uuid, new_game_data) + 2 * small_blind_amount 
                    self.prev_op_pot = self.get_stack(self.opponent_uuid, new_game_data) + small_blind_amount + last_opponent_paid
                    self.our_total_money = self.get_stack(self.our_uuid, new_game_data)
                    self.opponent_total_money = self.get_stack(self.opponent_uuid, new_game_data)
        else:
            self.round_count = new_game_data['round_count']
            self.our_total_money = self.get_stack(self.our_uuid, new_game_data)
            self.opponent_total_money = self.get_stack(self.opponent_uuid, new_game_data)
            self.our_investment = self.prev_our_pot - self.our_total_money
            self.opponent_investment = self.prev_op_pot - self.opponent_total_money

    def get_features(self):
        return {
            'hole_cards_vector': self.hole_cards_vector.to(torch.float32),
            'community_cards_vector': self.community_cards_vector.to(torch.float32),
            'our_total_money': torch.tensor([self.our_total_money]).to(torch.float32),
            'opponent_total_money': torch.tensor([self.opponent_total_money]).to(torch.float32),
            'our_investment_this_round': torch.tensor([self.our_investment]).to(torch.float32),
            'opponent_investment_this_round': torch.tensor([self.opponent_investment]).to(torch.float32),
        }
    
    def get_features_as_tensor(self):
        return torch.concat(tuple(self.get_features().values()))