import copy
import os
import pickle
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from q.game_simulator import GameSimulator
from collections import deque, namedtuple

class QModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(108, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        """
        Takes in round_state as tensor. Outputs a vector with 3 components that map to a specific action
        """

        return self.network(x)

class StateTransition:

    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
    
class DeepQModelWrapper:
    
    def __init__(self, model: QModel, discount_factor: float, exploration_factor: float, dump_folderpath: str, target_update_frequency: int, dump_freq: int = None):
        assert exploration_factor >= 0 and exploration_factor <= 1, "Exploration factor must be between 0 and 1"

        self.activated = False
        self.history = deque(maxlen=10000)
        self.model = model
        self.target_model = copy.deepcopy(self.model)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.dump_folderpath = dump_folderpath
        self.dump_freq = dump_freq
        self.target_update_frequency = target_update_frequency
        
        self.states_encountered = 0

        self.last_state = None
        self.last_action = None

    def new_round(self):
        self.last_state = None
        self.last_action = None

    def _choose_action(self, state, valid_actions: torch.Tensor, nonTensorState, round_state, hole_card, op_histories=None):
        """
        Either choose max Q value to pursure or, if exploration is triggered, explore random action
        """
        if(random.random() < self.exploration_factor):
            action_distribution = torch.rand(3)
        else:
            action_distribution = self.model(state)
        call_reward = GameSimulator.simulate_call(state, self.model, nonTensorState, round_state, hole_card, op_histories)
        print("call rewards:", call_reward)
        raise_reward = GameSimulator.simulate_raise(state, self.model, nonTensorState, round_state, hole_card, op_histories)
        print("rais rewards:", raise_reward)
        valid_indices = valid_actions.nonzero(as_tuple=False).squeeze()
        return valid_indices[action_distribution[valid_actions].argmax()].item()

    def register_current_state(self, curr_state, curr_reward, op_histories, nonTensorState, round_state, hole_card, return_action = False, valid_actions: torch.Tensor = None):
        """
        Saves the current state to history and associates it with the last state and action taken.
        All logic that deals with history occurs in this function
        """
        
        p1 = return_action and valid_actions is not None
        p2 = not return_action and valid_actions is None
        assert p1 or p2, "Invalid call to register_current_state"
        if(valid_actions is None): valid_actions = torch.tensor([True, True, True])

        self.states_encountered += 1

        if self.last_state is None or self.last_action is None:
            self.last_state = curr_state
            self.last_action = self._choose_action(curr_state, valid_actions, nonTensorState, round_state, hole_card, op_histories)
            return self.last_action if return_action else None
        
        self.history.append(StateTransition(self.last_state, self.last_action, curr_state, curr_reward))

        if(self.dump_freq is not None and self.states_encountered % self.dump_freq == 0):
            self.dump_history_and_model()

        self.last_state = curr_state
        self.last_action = self._choose_action(curr_state, valid_actions, nonTensorState, round_state, hole_card, op_histories)
        return self.last_action if return_action else None

    def dump_history_and_model(self):
        """
        Saves current history and model state to file for replay later
        """
        os.makedirs(self.dump_folderpath, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        hist_path = os.path.join(self.dump_folderpath, f"history-{ts}.pkl")
        with open(hist_path, "wb") as f:
            pickle.dump(list(self.history), f)
        model_path = os.path.join(self.dump_folderpath, f"model-{ts}.pt")
        torch.save(self.model.state_dict(), model_path)

    def optimize(self):
        """
        Optimizes Q based on history if there is enough registered
        """

        if len(self.history) < 1000:
            return

        batch = random.sample(self.history, 100)

        states = torch.stack([t.state for t in batch])
        actions = torch.tensor([t.action for t in batch])
        next_states = torch.stack([t.next_state for t in batch])
        rewards = torch.tensor([t.reward for t in batch])

        Q_left: torch.Tensor = torch.gather(self.model(states), 1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Given that only terminal nodes have non-zero rewards, we can set the target as follows
            Q_next: torch.Tensor = self.target_model(next_states).max(dim=1).values
            terminal_mask = rewards != 0
            Q_target = torch.where(
                terminal_mask,
                rewards,
                Q_next*self.discount_factor
            )

        loss = F.mse_loss(Q_left, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.states_encountered % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())