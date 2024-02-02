import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define Hyperparameters
HISTORY_LENGTH = 5  # Length of history to consider
GAMMA = 0.9          # Discount factor
LEARNING_RATE = 0.01
VALID_ACTIONS = 2 # Two possible actions: Cooperate or Defect
INITIAL_BIAS_FOR_C = 0.1 # we want our network to have a preference for C
# without a bias for C, initially the model could use D and trigger strategies 
# such as grudger, tit-for-tat, and so on. This greatly reduces the overall result.
INITIAL_BIAS_FOR_D = 0.0
VALUE_FOR_C = 0
VALUE_FOR_D = 1
INITIAL_EPSILON = 1 # Start with 100% exploration
MIN_EPSILON = 0.01 # Minimum value of epsilon (% of exploration)


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, history_length, valid_actions, initial_bias_for_C, initial_bias_for_D):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(history_length * 2, 50)  # History from both players
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, valid_actions)  # Two possible actions: Cooperate or Defect
        self.fc3.bias.data = torch.tensor([initial_bias_for_C, initial_bias_for_D])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def reset_weights(self):
        initial_bias_for_C = INITIAL_BIAS_FOR_C
        initial_bias_for_D = INITIAL_BIAS_FOR_D

        for layer in self.children():
            if isinstance(layer, nn.Linear):
                # Use Xavier uniform initialization for weights
                torch.nn.init.xavier_uniform_(layer.weight)

                # Reset biases
                if layer == self.fc3:
                    # Set custom biases for fc3
                    layer.bias.data = torch.tensor([initial_bias_for_C, initial_bias_for_D])
                else:
                    # Set biases to zero for other layers
                    layer.bias.data.fill_(0.0)

# Memory Class for Storing Experiences
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Main Class for Prisoners Dilemma DQN
class PrisonersDilemmaDQN:
    def __init__(self):
        self.model = DQN(HISTORY_LENGTH, VALID_ACTIONS, INITIAL_BIAS_FOR_C, INITIAL_BIAS_FOR_D)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.epsilon = INITIAL_EPSILON

    def reset_model_weights(self):
        # Call initialize_weights on the DQN instance
        self.model.reset_weights()

    def _preprocess_history(self, my_history, opponent_history, very_verbose):
        if very_verbose: print(f"my_history: {''.join(my_history[-10:])} opponent_history: {''.join(opponent_history[-10:])}")
        # Preprocess history to be fed into the DQN
        # Pad the history with 'C' if it's shorter than HISTORY_LENGTH
        padded_my_history = (['C'] * (HISTORY_LENGTH - len(my_history))) + my_history
        padded_opponent_history = (['C'] * (HISTORY_LENGTH - len(opponent_history))) + opponent_history

        # Combine the last HISTORY_LENGTH actions from each player
        history = padded_my_history[-HISTORY_LENGTH:] + padded_opponent_history[-HISTORY_LENGTH:]
        history_numerical = [VALUE_FOR_C if h == 'C' else VALUE_FOR_D for h in history]  # Convert to numerical format (0 for 'C', 1 for 'D')

        if very_verbose: print(f"history_numerical: {''.join(str(num) for num in history_numerical)}")

        return torch.tensor([history_numerical], dtype=torch.float32)

    def _get_reward(self, my_action, opponent_action):
        # Define the reward function for the prisoner's dilemma
        # Note that this is DIFFERENT from the values given by the game
        # and are meant to help train te model in the "direction" we are interested into
        if my_action == 'C' and opponent_action == 'C':
            return 1  # Both cooperate
        elif my_action == 'C' and opponent_action == 'D':
            return -1  # Player cooperates, opponent defects
        elif my_action == 'D' and opponent_action == 'C':
            return 2  # Player defects, opponent cooperates
        else:  # my_action == 'D' and opponent_action == 'D'
            return 0  # Both defect

    def train_model(self, my_history, opponent_history, my_action, opponent_action, very_verbose):
        if very_verbose: print(f"my_action: {my_action} opponent_action: {opponent_action}")
                       

        # Convert histories and actions to tensors
        state = self._preprocess_history(my_history, opponent_history, very_verbose)
        next_state = self._preprocess_history(my_history + [my_action], opponent_history + [opponent_action], very_verbose)

        my_actionNumber = VALUE_FOR_C if my_action == 'C' else VALUE_FOR_D
        action_tensor = torch.tensor([my_actionNumber], dtype=torch.int64)
        reward = self._get_reward(my_action, opponent_action)

        # Store in memory
        self.memory.push(state, action_tensor, next_state, reward)

        if len(self.memory) < HISTORY_LENGTH:
            if very_verbose: print(f"Not enough memory to train")
            return  # Not enough memory to train

        # Train on a single memory sample
        transitions = self.memory.sample(1)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        # Compute Q values
        current_q_values = self.model(torch.cat(batch_state)).gather(1, torch.tensor(batch_action).unsqueeze(1))
        max_next_q_values = self.model(torch.cat(batch_next_state)).max(1)[0].detach()
        expected_q_values = torch.tensor(batch_reward) + (GAMMA * max_next_q_values)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(1), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if very_verbose:
            print(f"Training loss: {loss.item()}")

    def make_decision(self, explore_flag, my_history, opponent_history, very_verbose):

        # Check if my_history is empty, indicating the start of a new game
        if not my_history:
            # Call reset_weights to reinitialize the model
            self.reset_model_weights()
            if very_verbose: print("Weights and biases have been reset.")
            self.epsilon = INITIAL_EPSILON
            # initially gpt did not offer this code and vocally advocated against it
            # saying that a model should be generic enough to face multiple adversary
            # but in reality each adversary plays so differently that resetting weights 
            # is key to the performance of the model

        state = self._preprocess_history(my_history, opponent_history,very_verbose)
        with torch.no_grad():
            q_values = self.model(state)
            if very_verbose: print(f"q_values: {q_values}")
        action = q_values.max(1)[1].item()

        action_char = 'C' if action == 0 else 'D'

        if very_verbose:
            print(f"Decided action: {action_char}")

        # I implemented epsilon greedy strategy in a different way
        # I first calculate the decided action and if epsilon triggers exploration
        # I do the opposite
        # if opponent has used 'D' at least once, we can contemplate exploration
        if len(opponent_history) > 0 and 'D' in opponent_history:
            if very_verbose:
                print(f"we could explore - epsilon: {self.epsilon}")
            if explore_flag and random.random() < self.epsilon:
                # so we know we have the "license" to explore, and we randomly decided to explore
                # so we will return the opposite value to the prediction
                if action_char == 'C':
                    action_char = 'D'
                else:
                    action_char = 'C'
                if very_verbose:
                    print(f"epsilon: {self.epsilon} random action: {action_char}")
                # Update epsilon
                if self.epsilon > MIN_EPSILON:
                    self.epsilon = max(MIN_EPSILON, self.epsilon / 2)


        return action_char



