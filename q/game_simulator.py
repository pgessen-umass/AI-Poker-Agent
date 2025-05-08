import numpy as np
import copy
from pypokerengine.utils import card_utils
import torch
from pypokerengine.engine.card import Card

class GameSimulator:

    def simulate_call(state, QModel, nonTensorState, round_state, hole_cards, op_histories=None):
        if hole_cards is not None:
            hole_cards = [Card.from_str(card) for card in hole_cards]
        community_cards = [Card.from_str(card) for card in round_state.get("community_card")]
        state_copy = copy.deepcopy(nonTensorState)
        # print("state:", round_state)
        simulated_rewards = 0
        if op_histories is None:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        op_histories_copy = np.array(copy.deepcopy(op_histories))
        if op_histories_copy.sum() < 5:
            op_histories_copy = np.array([1,1,1])
        our_amount_1 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round')
        state_copy['our_investment_this_round'] += our_amount_1
        state_copy['our_total_money'] -= our_amount_1
        simulated_rewards += our_amount_1
        if our_amount_1 == 0:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        op_action_1 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
        if op_action_1 == 0:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        elif op_action_1 == 1:
            op_amount_1 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
            simulated_rewards += op_amount_1
            state_copy['opponent_investment_this_round'] += op_amount_1
            state_copy['opponent_total_money'] -= op_amount_1
            if op_amount_1 == 0:
                return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
            else:
                s = torch.concat(tuple(state_copy.values()))
                our_action_2 = QModel(s).argmax().item()
                if our_action_2 == 0:
                    return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                elif our_action_2 == 1:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round')
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
                else:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round') + 10
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
        else:
            op_amount_1 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round') + 10
            simulated_rewards += op_amount_1
            state_copy['opponent_investment_this_round'] += op_amount_1
            state_copy['opponent_total_money'] -= op_amount_1
            if op_amount_1 == 0:
                return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
            else:
                s = torch.concat(tuple(state_copy.values()))
                our_action_2 = QModel(s).argmax().item()
                if our_action_2 == 0:
                    return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                elif our_action_2 == 1:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round')
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
                else:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round') + 10
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
    
    def simulate_raise(state, QModel, nonTensorState, round_state, hole_cards, op_histories=None):
        if hole_cards is not None:
            hole_cards = [Card.from_str(card) for card in hole_cards]
        community_cards = [Card.from_str(card) for card in round_state.get("community_card")]
        state_copy = copy.deepcopy(nonTensorState)
        simulated_rewards = 0
        if op_histories is None:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        op_histories_copy = np.array(copy.deepcopy(op_histories))
        if op_histories_copy.sum() < 5:
            op_histories_copy = np.array([1,1,1])
        our_amount_1 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round') + 10
        state_copy['our_investment_this_round'] += our_amount_1
        state_copy['our_total_money'] -= our_amount_1
        simulated_rewards += our_amount_1
        if our_amount_1 == 0:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        op_action_1 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
        if op_action_1 == 0:
            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
        elif op_action_1 == 1:
            op_amount_1 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
            simulated_rewards += op_amount_1
            state_copy['opponent_investment_this_round'] += op_amount_1
            state_copy['opponent_total_money'] -= op_amount_1
            if op_amount_1 == 0:
                return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
            else:
                s = torch.concat(tuple(state_copy.values()))
                our_action_2 = QModel(s).argmax().item()
                if our_action_2 == 0:
                    return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                elif our_action_2 == 1:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round')
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
                else:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round') + 10
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
        else:
            op_amount_1 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round') + 10
            simulated_rewards += op_amount_1
            state_copy['opponent_investment_this_round'] += op_amount_1
            state_copy['opponent_total_money'] -= op_amount_1
            if op_amount_1 == 0:
                return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
            else:
                s = torch.concat(tuple(state_copy.values()))
                our_action_2 = QModel(s).argmax().item()
                if our_action_2 == 0:
                    return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                elif our_action_2 == 1:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round')
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10
                else:
                    our_amount_2 = state_copy.get('opponent_investment_this_round') - state_copy.get('our_investment_this_round') + 10
                    state_copy['our_investment_this_round'] += our_amount_2
                    state_copy['our_total_money'] -= our_amount_2
                    simulated_rewards += our_amount_2
                    if our_amount_2 == 0:
                        return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                    else:
                        op_action_2 = np.random.choice(len(op_histories_copy), p=op_histories_copy/op_histories_copy.sum())
                        op_amount_2 = state_copy.get('our_investment_this_round') - state_copy.get('opponent_investment_this_round')
                        if op_action_2 == 0:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5)
                        elif op_action_1 == 1:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2
                        else:
                            return simulated_rewards * (card_utils.estimate_hole_card_win_rate(10,2, hole_cards, community_cards) if hole_cards is not None else 0.5) + op_amount_2 + 10