import copy
from scipy.optimize import curve_fit
import json
import os
import pickle
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from q.game_simulator import GameSimulator
from collections import deque, namedtuple

class QModel(nn.Module):

    def __init__(self, hidden_layer_size, num_hidden_layers):
        super().__init__()
        h = []
        for i in range(num_hidden_layers):
            h.append(
                nn.Linear(hidden_layer_size, hidden_layer_size)
            )
            h.append(nn.ReLU())

        self.network = nn.Sequential(
            nn.Linear(108, hidden_layer_size),
            nn.ReLU(),
            *h,
            nn.Linear(hidden_layer_size, 3)
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
    
    def __init__(self, model: QModel, discount_factor: float = None, exploration_factor: float = None, dump_folderpath: str = None, target_update_frequency: int = None, dump_freq: int = None, learning_rate: float = 1e-5, history = None):

        self.activated = False
        self.model = model

        if(os.path.exists(f"{dump_folderpath}/training_info.json")):
            with open(f"{dump_folderpath}/training_info.json","r") as f:
                self.training_info = json.load(f)
        else:
            self.training_info = {
                'loss': [],
                'training_points': 0
            }

        # The following is only relevant for the training stage
        if(discount_factor is not None):
            assert exploration_factor >= 0 and exploration_factor <= 1, "Exploration factor must be between 0 and 1"
            if(history is not None): self.history = history
            elif(os.path.exists(f"{dump_folderpath}/history.pkl")):
                with open(f"{dump_folderpath}/history.pkl","rb") as f:
                    self.history = pickle.load(f)
            else:
                self.history = deque(maxlen=20000)
            self.target_model = copy.deepcopy(self.model)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

    def deployed_choose_action(self, curr_state, curr_reward, op_histories, nonTensorState, round_state, hole_card, return_action = False, valid_actions: torch.Tensor = None):
        call_reward = GameSimulator.simulate_call(curr_state, self.model, nonTensorState, round_state, hole_card, op_histories)
        raise_reward = GameSimulator.simulate_raise(curr_state, self.model, nonTensorState, round_state, hole_card, op_histories)
        action_distribution = self.model(curr_state)
        action_distribution[0] += raise_reward.item()
        action_distribution[1] += call_reward.item()
        valid_indices = valid_actions.nonzero(as_tuple=False).squeeze()
        return valid_indices[action_distribution[valid_actions].argmax()].item()

    def _choose_action(self, state, valid_actions: torch.Tensor, nonTensorState, round_state, hole_card, op_histories=None):
        """
        Either choose max Q value to pursure or, if exploration is triggered, explore random action
        """
        if(random.random() < self.exploration_factor):
            action_distribution = torch.rand(3)
        else:
            action_distribution = self.model(state)
        call_reward = GameSimulator.simulate_call(state, self.model, nonTensorState, round_state, hole_card, op_histories)
        raise_reward = GameSimulator.simulate_raise(state, self.model, nonTensorState, round_state, hole_card, op_histories)
        action_distribution[0] += raise_reward.item()
        action_distribution[1] += call_reward.item()
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
            self.dump()

        self.last_state = curr_state
        self.last_action = self._choose_action(curr_state, valid_actions, nonTensorState, round_state, hole_card, op_histories)
        return self.last_action if return_action else None

    def dump(self, history=False):
        """
        Saves current history and model state to file for replay later
        """
        os.makedirs(self.dump_folderpath, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(self.dump_folderpath, f"model-{ts}.pt")
        torch.save(self.model.state_dict(), model_path)
        with open(f"{self.dump_folderpath}/training_info.json","w") as f:
            json.dump(self.training_info,f)
        if(history):
            with open(f"{self.dump_folderpath}/history.pkl","wb") as f:
                pickle.dump(self.history,f)

    def plot_loss(self, exponential_decay_fit = False, show=True,**kwargs):
        number_of_training_points = range(0,len(self.training_info['loss'])*100,100)
        plt.plot(number_of_training_points, self.training_info['loss'], **kwargs)
        
        if(exponential_decay_fit):
            e = lambda x, a, b, c: a*np.exp(-b*x) + c
            x = np.array(number_of_training_points)
            y = np.array(self.training_info['loss'])
            out, _ = curve_fit(e, x, y)
            plt.plot(x,e(x,*out))
        if(show): plt.show()

    def optimize(self):
        """
        Optimizes Q based on history if there is enough registered
        """

        if len(self.history) < 10000:
            return
        
        batch = random.sample(self.history, 1000)

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

        self.training_info['loss'].append(loss.clone().detach().item())
        self.training_info['training_points'] += 1000

        if(self.training_info['training_points'] > 3e7 and self.exploration_factor > 0.05):
                self.exploration_factor = max(0.05, self.exploration_factor * 0.99)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.states_encountered % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())