from __future__ import annotations
import numpy as np
import torch
from q_model import QModel

# Utility function. Outside of QSession because requires knowledge of actions that would take place
def get_maximal_Q(session: QSession, s_prime):

    # Get all of the actions that can be taken from s_prime
    actions = []

    action_Qs = np.zeros(len(actions))

    for i, a in enumerate(actions):
        action_Qs[i] = session.get_Q(session.encode(s_prime, a))

    return action_Qs.max()

class QModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data

class QSession:

    def __init__(self):

        # Initialized in either new_model or load_model
        self.model: torch.nn.Module = None
        self.optimizer = None

        # This will likely just hold a list of tensors
        self.replay_buffer = []

        # The weight of future rewards. Set this to something reasonable
        self.gamma = 0.9

    # Somehow encodes a state and action into a tensor
    def encode(self, state, action, reward, s_prime) -> torch.Tensor:
        pass

    # Set this to a reasonable loss function
    def loss_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x-y).square().mean()

    # Will store in replay_buffer 
    def register_in_buffer(self, state, action, reward, s_prime):
        self.replay_buffer(
            [state, action, reward, s_prime]
        )

    # This trains network on values seen in the past to regularize 
    def train_on_past_values(self, batch_size):

        # Sample random values from replay_buffer

        samples = np.random.randint(0,len(self.replay_buffer), size=batch_size)

        for i in samples:
            self.train(*self.replay_buffer[i], new=False)

    # Updates network based on single step. 
    def train(self, state, action, reward, s_prime, new = True) -> torch.Tensor :

        if(new): self.register_in_buffer(state, action, reward, s_prime)

        input_encoded = session.encode(state, action, reward, s_prime)

        # Forward pass of the network
        pred_Q = self.model(input_encoded)

        # Get Q target of network
        target_Q = reward + get_maximal_Q(self, s_prime)*self.gamma

        # Calculate loss
        loss = self.loss_function(pred_Q, target_Q)

        # Update weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if(new): self.train_on_past_values()

        # Return loss
        return loss

    def get_Q(self, state, action):

        with torch.no_grad():
            pred_Q = self.model(self.encode(state, action))

        return pred_Q

    def new_model(self):
        self.model = QModel()
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=0.001,
        )

    def save_model(self, save_path):
        torch.save(self.model, save_path)

    def load_model(self, model_filepath):
        self.model = torch.load(model_filepath)
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=0.001,
        )

if __name__ == "__main__":

    # Example instance
    session = QSession()
    session.new_model()

    # Agent will traverse the state space. As it traverses, gets states and actions, and rewards from environement.
    # Run this as many times as you want, exploring different parts of state space
    state, action, reward, s_prime = None, None, None, None
    session.train(state, action, reward, s_prime)

    # Save model trained somewhere
    save_path = ""
    session.save_model(save_path)

    # Load later
    session.load_model(save_path)

    # Keep training, or go ahead and actually use by calling forward_no_grad
    pred_Q = session.get_Q(state, action)