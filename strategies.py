
def rl_strategy(dqn_agent, explore_flag, myHistory, opponentHistory, very_verbose):
    action = dqn_agent.make_decision(explore_flag, myHistory, opponentHistory, very_verbose)
    return action

def always_cooperate(myHistory, opponentHistory, very_verbose):
    return 'C'


def always_defect(myHistory, opponentHistory, very_verbose):
    return 'D'


def tit_for_tat_trustful(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'C'
    return opponentHistory[-1]


def grudger(myHistory, opponentHistory, very_verbose):
    if 'D' in opponentHistory:
        return 'D'
    else:
        return 'C'


def random_strategy(myHistory, opponentHistory, very_verbose):
    import random
    return random.choice(['C','D'])


def win_stay_lose_shift(myHistory, opponentHistory, very_verbose):
    if not myHistory:
        return 'C'
    last_my_move = myHistory[-1]
    last_opponent_move = opponentHistory[-1]

    # Collate last moves and check if they are 'CC' or 'DC'
    last_moves = last_my_move + last_opponent_move
    if last_moves in ['CC', 'DC']:
        return last_my_move
    else:
        # Return the opposite of last_my_move
        if last_my_move == 'C':
            return 'D'
        else:
            return 'C'


def grudger_recovery(myHistory, opponentHistory, very_verbose):
    import random
    if not opponentHistory:
        return 'C'  # Default to cooperate if history is undefined or empty

    # Check if the last move I made was to cooperate and if I have ever defected
    if myHistory[-1] == 'C' and 'D' in myHistory:
        # We are in a recovery scenario

        # Check if the last move I made was cooperate and the one before was defect
        if len(myHistory) > 2 and myHistory[-2] == 'D':
            # Need to cooperate again since I couldn't see the opponent's reaction yet
            return 'C'

        # Find the last position where I defected
        myLastDefectPosition = len(myHistory) - 1 - myHistory[::-1].index('D')

        # Get opponent's history since two moves after my last defect
        opponentHistorySinceImGood = opponentHistory[myLastDefectPosition + 2:]

        # Decide next move based on opponent's history since then
        return 'D' if 'D' in opponentHistorySinceImGood else 'C'

    # Generate a random number between 5 and 10
    import random
    randomNumber = random.random() * 5 + 5

    # Check if both players have been defecting for a while
    if len(myHistory) > randomNumber:
        recent_my_history = myHistory[-int(randomNumber):]
        recent_opponent_history = opponentHistory[-int(randomNumber) + 1:]

        # Check if both players have been defecting in the recent history
        if all(move == 'D' for move in recent_my_history) and all(move == 'D' for move in recent_opponent_history):
            # We are in a D/D deadlock, try to break it unless facing an always-defect opponent
            if all(move == 'D' for move in opponentHistory):
                return 'D'
            else:
                return 'C'

    # Default action based on opponent's history
    return 'D' if 'D' in opponentHistory else 'C'

def provocateur(myHistory, opponentHistory, very_verbose):
    # if it cooperated for the last 2 rounds, defect
    if len(myHistory) >= 2 and myHistory[-1] == 'C' and myHistory[-2] == 'C':
        return 'D'
    else:
        return 'C'


def tit_for_tat_opposite_coop(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'C'
    if opponentHistory[-1] == 'C':
        return 'D'
    else:
        return 'C'


def tit_for_tat_opposite_def(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'D'
    if opponentHistory[-1] == 'C':
        return 'D'
    else:
        return 'C'



def tit_for_tat_suspicious(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'D'
    return opponentHistory[-1]


def tit_for_tat_generous_30(myHistory, opponentHistory, very_verbose):
    import random
    if not opponentHistory:
        return 'C'
    if opponentHistory[-1] == 'C' or random.random() < 0.3:
        return 'C'
    else:
        return 'D'


def tit_for_tat_generous_10(myHistory, opponentHistory, very_verbose):
    import random
    if not opponentHistory:
        return 'C'
    if opponentHistory[-1] == 'C' or random.random() < 0.1:
        return 'C'
    else:
        return 'D'


def two_tits_for_tat(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'C'
    if len(opponentHistory) == 1:
        return opponentHistory[-1]
    if opponentHistory[-1] == 'D' or opponentHistory[-2] == 'D':
        return 'D'
    else:
        return 'C'


def tit_for_two_tats(myHistory, opponentHistory, very_verbose):
    if opponentHistory or len(opponentHistory) < 2:
        return 'C'
    if opponentHistory[-1] == 'D' and opponentHistory[-2] == 'D':
        return 'D'
    else:
        return 'C'

def tit_for_three_tats(myHistory, opponentHistory, very_verbose):
    if opponentHistory or len(opponentHistory) < 3:
        return 'C'
    if opponentHistory[-1] == 'D' and opponentHistory[-2] == 'D' and opponentHistory[-3] == 'D':
        return 'D'
    else:
        return 'C'

def tit_for_tat_gradual(myHistory, opponentHistory, very_verbose):
    # https://plato.stanford.edu/entries/prisoner-dilemma/strategy-table.html
    # TFT with two differences: (1) it increases the string of punishing defection responses 
    # with each additional defection by its opponent 
    # (2) it apologizes for each string 
    # of defections by cooperating in the subsequent two rounds.

    # Start with cooperation if no history
    if not opponentHistory:
        return 'C'

    # Count the number of defections by the opponent
    opponentDefections = opponentHistory.count('D')

    # Count the number of consecutive defections at the end of my history
    myRecentDefections = 0
    for move in reversed(myHistory):
        if move == 'D':
            myRecentDefections += 1
        else:
            break

    # If opponent has defected more times than my recent consecutive defections, retaliate with 'D'
    if opponentDefections > myRecentDefections:
        return 'D'

    # If my number of recent defections equals opponent's total defections, switch to 'C'
    if opponentDefections == myRecentDefections:
        return 'C'

    # Apologize with two 'C' in a row after a string of defections
    if len(myHistory) >= 2 and myHistory[-1] == 'C' and myHistory[-2] == 'D':
        return 'C'

    # Default to classic TFT behavior
    return opponentHistory[-1] 


def tit_for_tat_imperfect(myHistory, opponentHistory, very_verbose):
    import random
    imitation_probability = 0.9
    if not opponentHistory:
        return 'C'
    response = opponentHistory[-1]

    if random.random() < imitation_probability:
        return response
    else:
        if response == 'C':
            return 'D'
        else:
            return 'C'

def return_avg_last_5(myHistory, opponentHistory, very_verbose):
    if not opponentHistory:
        return 'C'
    cutOff = min (5,len(opponentHistory))
    last_moves_max_5 = opponentHistory[-cutOff:]
    cooperate_count = last_moves_max_5.count('C')
    defect_count = last_moves_max_5.count('D')
    # print (f"opponentHistory: {opponentHistory[-10:]} cutOff: {cutOff} last_moves_max_5: {last_moves_max_5} cooperate_count: {cooperate_count} defect_count: {defect_count}")
    if cooperate_count > defect_count:
        return 'C'
    else: 
        return 'D'


def return_avg_all_def(opponentHistory, myHistory, very_verbose):
    # Returns the most common action of the opponent so far
    # When equal, return DEFECT
    if len(myHistory) < 1:
        return 'C'
    cooperate_count = opponentHistory.count('C')
    defect_count = opponentHistory.count('D')
    if cooperate_count > defect_count:
        return 'C'
    else: 
        return 'D'

def return_avg_all_coop(myHistory, opponentHistory, very_verbose):
    # Returns the most common action of the opponent so far
    # When equal, return COOPERATE
    if len(myHistory) < 1:
        return 'C'
    cooperate_count = opponentHistory.count('C')
    defect_count = opponentHistory.count('D')
    if cooperate_count >= defect_count:
        return 'C'
    else: 
        return 'D'

def random_70_cooperation(myHistory, opponentHistory, very_verbose):
    import random
    if random.random() > 0.3:
        return 'C'
    else: 
        return 'D'

def random_80_cooperation(myHistory, opponentHistory, very_verbose):
    import random
    if random.random() > 0.2:
        return 'C'
    else: 
        return 'D'

def random_90_cooperation(myHistory, opponentHistory, very_verbose):
    import random
    if random.random() > 0.1:
        return 'C'
    else: 
        return 'D'

def alternate3and3(myHistory, opponentHistory, very_verbose):
    cycle_of_moves = ['C','C','C','D','D','D']
    current_position = len(myHistory) % len(cycle_of_moves)
    selected_move = cycle_of_moves[current_position]
    return selected_move


def alternate_coop(myHistory, opponentHistory, very_verbose):
    cycle_of_moves = ['C','D']
    current_position = len(myHistory) % len(cycle_of_moves)
    selected_move = cycle_of_moves[current_position]
    return selected_move


def alternate_def(myHistory, opponentHistory, very_verbose):
    cycle_of_moves = ['D','C']
    current_position = len(myHistory) % len(cycle_of_moves)
    selected_move = cycle_of_moves[current_position]
    return selected_move

