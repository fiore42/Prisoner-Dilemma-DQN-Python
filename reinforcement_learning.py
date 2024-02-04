import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# device = torch.device("mps")

# Define Hyperparameters
HISTORY_LENGTH = 5  # Length of history to consider
# GAMMA = 0.79          # this value has the lowest mean and stdev with heatmap.py
GAMMA = 0.9          # Discount factor - this also has low mean and stdev with heatmap.py
LEARNING_RATE = 0.01 # this value has the lowest mean and stdev with heatmap.py
# LEARNING_RATE = 0.0001
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
        self.layer1 = nn.Linear(history_length * 2, 50)  # History from both players
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, valid_actions)  # Two possible actions: Cooperate or Defect
        self.layer3.bias.data = torch.tensor([initial_bias_for_C, initial_bias_for_D])

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        return self.layer3(x)

    def reset_weights(self, bias_for_c):
        if bias_for_c:
            initial_bias_for_C = INITIAL_BIAS_FOR_C
            initial_bias_for_D = INITIAL_BIAS_FOR_D
        else:
            # our opponent stroke the first blow, now we prefer D!
            initial_bias_for_C = 0.0
            initial_bias_for_D = 0.0
            # initial_bias_for_C = INITIAL_BIAS_FOR_D
            # initial_bias_for_D = INITIAL_BIAS_FOR_C

        for layer in self.children():
            if isinstance(layer, nn.Linear):
                # Use Xavier uniform initialization for weights
                torch.nn.init.xavier_uniform_(layer.weight)

                # Reset biases
                if layer == self.layer3:
                    # Set custom biases for fc3
                    # print (f"initial_bias_for_C: {initial_bias_for_C} initial_bias_for_D: {initial_bias_for_D}")
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

    def clear(self):
        """Clears all the experiences in the memory."""
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

# Main Class for Prisoners Dilemma DQN
class PrisonersDilemmaDQN:
    def __init__(self,hp):
        self.model = DQN(HISTORY_LENGTH, VALID_ACTIONS, INITIAL_BIAS_FOR_C, INITIAL_BIAS_FOR_D) # Move model to device
        self.memory = ReplayMemory(10000)
        if hp.lr is not None:
            # we are in explore mode, let's use provided learning rate
            # print (f"we are in explore mode, let's use provided learning rate: {hp.lr}")
            self.optimizer = optim.Adam(self.model.parameters(), lr=hp.lr)
        else:
            # print (f"we are NOT in explore mode, let's use standard learning rate: {LEARNING_RATE}")
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.epsilon = INITIAL_EPSILON
        self.flag = 'white_flag' #if an opponent strikes first, we switch to pirate_flag

    def reset_model_weights(self, bias_for_c):
        # Call initialize_weights on the DQN instance
        self.model.reset_weights(bias_for_c)

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
            return 1  # Both cooperate (3 points)
        elif my_action == 'C' and opponent_action == 'D':
            return -1  # Player cooperates, opponent defects (0 points)
        elif my_action == 'D' and opponent_action == 'C':
            return 2  # Player defects, opponent cooperates (5 points)
        else:  # my_action == 'D' and opponent_action == 'D'
            return 0  # Both defect (1 point)

    def train_model(self, my_history, opponent_history, my_action, opponent_action, hp, very_verbose):
        if very_verbose: print(f"\n[train_model] my_action: {my_action} opponent_action: {opponent_action}")

        # Convert histories and actions to tensors
        state = self._preprocess_history(my_history, opponent_history, very_verbose)
        next_state = self._preprocess_history(my_history + [my_action], opponent_history + [opponent_action], very_verbose)

        my_action_number = VALUE_FOR_C if my_action == 'C' else VALUE_FOR_D
        action_tensor = torch.tensor([my_action_number], dtype=torch.int64)
        reward = self._get_reward(my_action, opponent_action)

        if very_verbose: print (f"reward: {reward}")

        # Store in memory
        self.memory.push(state, action_tensor, next_state, reward)

        # if very_verbose: print(f"state: {state} action_tensor: {action_tensor} next_state: {next_state} reward: {reward}")

        if len(self.memory) < HISTORY_LENGTH:
            if very_verbose: print(f"Not enough memory to train")
            return  # Not enough memory to train
        # else:
        #     if very_verbose: print (f"len(self.memory): {len(self.memory)}")


        # # Train on a single memory sample
        # transitions = self.memory.sample(1)
        # batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        # Train on the most recent experience
        most_recent_transition = [self.memory.memory[-1]]  # Assuming self.memory.memory is a deque
        batch_state, batch_action, batch_next_state, batch_reward = zip(*most_recent_transition)

        # if very_verbose: print(f"batch_state: {batch_state} batch_action: {batch_action} batch_next_state: {batch_next_state} batch_reward: {batch_reward}")

        # # what happens if I train ONLY BASED ON MY EXPERIENCE not on random element of self.memory?
        # batch_state = state
        # batch_action = action_tensor
        # batch_next_state = next_state
        # batch_reward = reward

        # Compute Q values
        current_q_values = self.model(torch.cat(batch_state)).gather(1, torch.tensor(batch_action).unsqueeze(1))
        # if very_verbose: print (f"state: {state}, current_q_values: {current_q_values}")
        model_next_state = self.model(torch.cat(batch_next_state))
        # if very_verbose: print (f"model_next_state: {model_next_state}")
        max_next_q_values = model_next_state.max(1)[0].detach()
        # if very_verbose: print (f"next_state: {next_state}, max_next_q_values: {max_next_q_values}")
        if hp.gamma:
            # we are in explore loop - let's use the provided gamma
            #print (f"we are in explore loop - let's use the provided gamma: {hp.gamma}")
            expected_q_values = torch.tensor(batch_reward) + (hp.gamma * max_next_q_values)
        else:
            expected_q_values = torch.tensor(batch_reward) + (GAMMA * max_next_q_values)
        # if very_verbose: print (f"expected_q_values: {expected_q_values}")

        # if very_verbose: print(f"current_q_values: {current_q_values.squeeze(1)} expected_q_values: {expected_q_values}")

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(1), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if very_verbose: print(f"Training loss: {loss.item()}")

    def make_decision(self, greedy_flag, my_history, opponent_history, very_verbose):

        if very_verbose: print(f"\n[make_decision] decision: {len(my_history)+1}")

        # Check if my_history is empty, indicating the start of a new game
        if not my_history:
            # Call reset_weights to reinitialize the model
            bias_for_c = True
            self.reset_model_weights(bias_for_c)
            if very_verbose: print("Weights have been reset (with bias).")
            # clean all memory, otherwise you train with behaviour of other adversaries
            self.memory.clear()
            if very_verbose: print("Memory has been cleaned.")
            self.epsilon = INITIAL_EPSILON
            self.flag = 'white_flag' # let's reset our white flag with a new opponent
            # initially gpt did not offer this code and vocally advocated against it
            # saying that a model should be generic enough to face multiple adversary
            # but in reality each adversary plays so differently that resetting weights 
            # is key to the performance of the model

        if self.flag == 'white_flag' and 'D' not in my_history and 'D' in opponent_history:
            if very_verbose: print("Opponent stroke first. Let's change our flag to the pirate flag!")
            bias_for_c = False # this is our pirate flag, now we don't give any bias to C anymore.
            self.reset_model_weights(bias_for_c)
            if very_verbose: print("Weights have been reset (with bias).")
            # clean all memory, otherwise you train with behaviour of other adversaries
            self.memory.clear()
            if very_verbose: print("Memory has been cleaned.")
            self.epsilon = INITIAL_EPSILON
            self.flag = 'pirate_flag' #we found a sneaky opponent, let's switch to pirate flag!

        state = self._preprocess_history(my_history, opponent_history, very_verbose)
        with torch.no_grad():
            q_values = self.model(state)
            if very_verbose: print(f"q_values: {q_values}")

        action = q_values.max(1)[1].item()

        # # temp delete me
        # temp_state = torch.tensor([0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        # with torch.no_grad():
        #     temp_q_values = self.model(temp_state)
        #     if very_verbose: print(f"temp_q_values: {temp_q_values} state [0,0,0,0,0,0,0,0,0,0]")


        action_char = 'C' if action == 0 else 'D'

        if very_verbose:
            print(f"Decided action: {action_char}")

        # I implemented epsilon greedy strategy in a different way
        # I first calculate the decided action and if epsilon triggers exploration
        # I do the opposite
        # if opponent has used 'D' at least once, we can contemplate exploration
        if opponent_history and 'D' in opponent_history:
            if very_verbose:
                print(f"we could explore - epsilon: {self.epsilon}")
                print(f"opponent_history: {''.join(opponent_history[-10:])}")

            if greedy_flag and random.random() < self.epsilon:
                # so we know we have the "license" to explore, and we randomly decided to explore
                # so we will return the opposite value to the prediction
                if action_char == 'C':
                    action_char = 'D'
                else:
                    action_char = 'C'
                if very_verbose:
                    print(f"epsilon: {self.epsilon} random action: {action_char} decision: {len(my_history)+1}")
                # Update epsilon
                if self.epsilon > MIN_EPSILON:
                    self.epsilon = max(MIN_EPSILON, self.epsilon / 2)


        return action_char



