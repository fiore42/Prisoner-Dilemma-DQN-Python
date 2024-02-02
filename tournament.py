# tournament.py

import random
import inspect
import strategies
import sys
from importlib import import_module
# from reinforcement_learning import train_model, print_q_table
from reinforcement_learning import PrisonersDilemmaDQN
from config import POINTS_SYSTEM

# Create a global instance of the DQN agent
dqn_agent = PrisonersDilemmaDQN()

def play_round(strategy1, name1, strategy2, name2, history1, history2, very_verbose):

    if name1 == 'rl_strategy' and name2 == 'rl_strategy':
        move1 = strategy1(dqn_agent, history1, history2, very_verbose)  # Pass both histories to strategy1
        move2 = strategy2(dqn_agent, history2, history1, very_verbose)  # Pass both histories to strategy2
        dqn_agent.train_model(history1, history2, move1, move2, very_verbose) # when rl_strategy face itself we train only once
    elif name1 == 'rl_strategy':
        move1 = strategy1(dqn_agent, history1, history2, very_verbose)  # Pass both histories to strategy1
        move2 = strategy2(history2, history1, very_verbose)  # Pass both histories to strategy2
        dqn_agent.train_model(history1, history2, move1, move2, very_verbose) # when rl_strategy face itself we train only once
    elif name2 == 'rl_strategy':
        move1 = strategy1(history1, history2, very_verbose)  # Pass both histories to strategy1
        move2 = strategy2(dqn_agent, history2, history1, very_verbose)  # Pass both histories to strategy2
        dqn_agent.train_model(history2, history1, move2, move1, very_verbose)
    else:
        move1 = strategy1(history1, history2, very_verbose)  # Pass both histories to strategy1
        move2 = strategy2(history2, history1, very_verbose)  # Pass both histories to strategy2
        
    # print(f"name1: {name1}", file=sys.stderr)
    # print(f"name2: {name2}", file=sys.stderr)

    # Update history
    history1.append(move1)
    history2.append(move2)

    # Calculate scores based on the points system
    score1, score2 = POINTS_SYSTEM[move1+move2]

    return score1, score2

    # tournament_result = tournament(num_rounds, max_strategies, opponent_strategies, verbose, very_verbose)


def select_strategies(all_strategies, max_strategies, opponent_strategies):
    selected_strategies = ['rl_strategy']

    # Remove 'rl_strategy' from the pool to avoid duplicating it
    available_strategies = [name for name in all_strategies.keys() if name != 'rl_strategy']

    # Check if all opponent strategies are available
    if all(strategy in available_strategies for strategy in opponent_strategies):
        # Make selected_strategies = 'rl_strategy' and opponent strategies
        selected_strategies += opponent_strategies
    else:
        # Print an error and exit the function
        print("\nSelect valid opponent strategies. Some of the specified strategies are not available.")
        print("\nAvailable strategies are:")
        for strategy in available_strategies:
            print(f"- {strategy}")
        return None

    # Fill the rest of selected_strategies with random choices from available strategies
    # until reaching the limit of min(all_strategies, max_strategies)
    while len(selected_strategies) < min(len(all_strategies), max_strategies):
        # Randomly choose a strategy from available_strategies
        choice = random.choice(available_strategies)
        # Ensure we don't add duplicates
        if choice not in selected_strategies:
            selected_strategies.append(choice)
    
    return selected_strategies

def tournament(num_rounds, max_strategies, opponent_strategies, verbose, very_verbose):

    results_temp = {}
    hands_played = {}

    # Automatically build the strategies dictionary
    all_strategies = build_strategies_dict()

    selected_strategies = select_strategies (all_strategies, max_strategies, opponent_strategies)

    if selected_strategies is None:
        # print(f"\nselected_strategies is None, return")

        return None, None

    strategy_points = {name: (0, 0) for name in selected_strategies} # total points, number of games

    # # Print the selected strategies
    # print("Selected strategies for this tournament:", selected_strategies)

    # Play each pair of strategies against each other
    for name1 in selected_strategies:
        for name2 in selected_strategies:
            history1, history2 = [], []
            total_score1, total_score2 = 0, 0
            match_hands = []

            # print (f"\n\n{name1} vs {name2} num_rounds: {num_rounds}")
            if verbose: print (f"\n\n{name1} vs {name2} num_rounds: {num_rounds}")

            for _ in range(num_rounds):
                score1, score2 = play_round(all_strategies[name1], name1, all_strategies[name2], name2, history1, history2, very_verbose)
                total_score1 += score1
                total_score2 += score2
                match_hands.append((history1[-1], history2[-1]))

            if name1 == 'rl_strategy' or name2 == 'rl_strategy':
                # print(f"test")
                if very_verbose: 
                    print (f"{name1} vs {name2} num_rounds: {num_rounds} score1: {total_score1} score2: {total_score2}")

            if verbose: 
                print(f"{name1} vs {name2} num_rounds: {num_rounds} score1: {total_score1} score2: {total_score2}")
                for i in range(len(history1)):
                    pair = history1[i] + history2[i]
                    # if pair in ["CC", "DC"]:
                    if pair in ["CD"]:
                        print(f"\033[1m{pair}\033[0m", end=" ")
                    else:
                        print(pair, end=" ")
                    if (i + 1) % 20 == 0:
                        print("")  # Newline character after every 20 pairs

            # percent_diff = (total_score1 - total_score2) / max(total_score1, total_score2) * 100 if max(total_score1, total_score2) > 0 else 0
            # percent_diff = (total_score1 - total_score2) / total_score2 * 100 if total_score2 > 0 else 0
            percent_diff = int((total_score1 - total_score2) / total_score2 * 100) if total_score2 > 0 else 0

            if percent_diff >= 100:
                percent_diff_string = "Diff >100"
            elif percent_diff == -100:
                percent_diff_string = "YOU LOST"
            elif percent_diff == 0:
                percent_diff_string = ""
            else:
                percent_diff_string = f"Diff: {percent_diff}%"

            results_temp[(name1, name2)] = (total_score1, total_score2, total_score1/num_rounds, total_score2/num_rounds, percent_diff_string)
            hands_played[(name1, name2)] = match_hands
            # Update total points and number of games
            total_points1, games_played1 = strategy_points[name1]
            strategy_points[name1] = (total_points1 + total_score1, games_played1 + num_rounds)
            total_points2, games_played2 = strategy_points[name2]
            strategy_points[name2] = (total_points2 + total_score2, games_played2 + num_rounds)

            # strategy_points[name1] += total_score1
            # strategy_points[name2] += total_score2

    # for name, (total_points, games_played) in strategy_points.items():
    #     print(f"name: {name}: total_points: {total_points} games_played: {games_played}", file=sys.stderr)

    # Calculate the average points per game for each strategy
    avg_points_per_game = {}
    for name, (total_points, games_played) in strategy_points.items():
        avg_points_per_game[name] = total_points / games_played

    # # Calculate the average points per game for each strategy
    # avg_points_per_game = {
    #     name: total_points / games_played
    #     for name, (total_points, games_played) in strategy_points.items()
    # }

    # Combine total points and average points per game into a single value for sorting
    combined_points = {}
    for name, (total_points, _) in strategy_points.items():
        avg_points = avg_points_per_game[name]
        combined_points[name] = (total_points, avg_points)

    # Find the strategy with the highest average points per game (best_strategy)
    best_strategy = max(avg_points_per_game, key=avg_points_per_game.get)

    # Calculate delta_avg_points for each strategy
    for name, (total_points, avg_points) in combined_points.items():
        delta_avg_points = (avg_points_per_game[name] - avg_points_per_game[best_strategy]) / avg_points_per_game[best_strategy] * 100
        combined_points[name] = (total_points, avg_points, delta_avg_points)  # Add delta_avg_points to combined_points

    # # Combine total points and average points per game into a single value for sorting
    # combined_points = {
    #     name: (total_points, avg_points)
    #     for name, (total_points, _) in strategy_points.items() for avg_points in [avg_points_per_game[name]]
    # }

    # Sort strategies by total points and include average points per game
    sorted_strategies = sorted(combined_points.items(), key=lambda x: x[1][0], reverse=True)

    # sort results_temp in the same order from the top strategy down
    results = {}
    for key, _ in sorted_strategies:
        # print(key)
        filtered_results = {(name1, name2): values for (name1, name2), values in results_temp.items() if name1 == key}
        for (name1, name2), values in filtered_results.items():
            results[(name1, name2)] = results_temp[(name1, name2)]


    # # Sort strategies by total points
    # sorted_strategies = sorted(strategy_points.items(), key=lambda x: x[1], reverse=True)

    return results, sorted_strategies

def print_results (results, sorted_strategies):
    # Always print results 
    print("\nTournament Results:")
    # Find the longest strategy name
    max_length = max(len(strategy) for match in results.keys() for strategy in match)
    last_name1 = None  # Initialize a variable to keep track of the last 'name1'

    # Format and print the results
    for match, score in results.items():
        name1, name2 = match
        score1, score2, avg_score1, avg_score2, percent_diff = score
        # percent_diff = (score1 - score2) / max(score1, score2) * 100 if max(score1, score2) > 0 else 0
        # Check if 'name1' has changed since the last iteration
        if last_name1 and name1 != last_name1:
            print()  # Print an extra empty line
        # note percent_diff is a string, not a number!
        # print(f"{name1:{max_length}} vs {name2:{max_length}}: {score1:5} - {score2:5} (Avg: {avg_score1:.2f} - {avg_score2:.2f}) Diff: {percent_diff:.2f}%")
        print(f"{name1:{max_length}} vs {name2:{max_length}}: {score1:5} - {score2:5} (Avg: {avg_score1:.2f} - {avg_score2:.2f}) {percent_diff}")
        last_name1 = name1  # Update 'last_name1' for the next iteration

    # Find the maximum points for formatting
    max_points = max(total_points for _, (total_points, _, _) in sorted_strategies)

    # Length for points formatting
    max_points_length = len(str(max_points))

    # Always print sorted strategies
    print("\nSorted Strategies:")
    for strategy, (total_points, avg_points, delta_avg_points) in sorted_strategies:
        formatted_delta_avg_points = f", Delta Avg = {'{:=3d}'.format(int(delta_avg_points))}%" if not delta_avg_points == 0 else ""
        # formatted_delta_avg_points = f", Delta Avg = {delta_avg_points: .0f}%" if not delta_avg_points == 0 else ""
        string = f"{strategy:{max_length}}: Total Points = {total_points:{max_points_length}}, Avg Points/Game = {avg_points:.2f}{formatted_delta_avg_points}"
        # Check if the strategy is "rl_strategy" and apply ANSI bold if true
        if strategy == "rl_strategy":
            formatted_string = f"\033[1m{string}\033[0m"
        else:
            formatted_string = string       
        print (formatted_string) 

    print(f"")
    # for strategy, score in sorted_strategies:
    #     print(f"{strategy}: {score}")

def build_strategies_dict():
    # Initialize an empty strategies_dict
    # I REFUSE to list manually strategies defined in strategies.py 
    # so this complicated code analyses callable functions defined in strategies.py
    strategies_dict = {}

    # Import the strategies module
    strategies_module = import_module('strategies')

    # Get the module name of the strategies module
    strategies_module_name = strategies_module.__name__

    # Iterate through the members of the strategies module
    for member_name, member_obj in inspect.getmembers(strategies_module):
        if (
            callable(member_obj) and 
#            member_name != 'train_model' and
            getattr(member_obj, '__module__', '') == strategies_module_name
        ):
            # Check if it's a callable function (strategy), exclude 'train_model', 
            # and ensure it's defined within the strategies module
            strategies_dict[member_name] = member_obj
            # print(f"Strategy: {member_name}", file=sys.stderr)

    return strategies_dict

# Strategies dictionary
# Automatically build the strategies dictionary
# strategies_dict = {name: func for name, func in inspect.getmembers(strategies, inspect.isfunction)}
# strategies_dict = build_strategies_dict()

# # Points system for the Prisoner's Dilemma
# points_system = {'CC': (3, 3), 'CD': (0, 5), 'DC': (5, 0), 'DD': (1, 1)}

# # Run a single tournament as an example
# if __name__ == "__main__":
#     tournament_results = tournament(strategies_dict, points_system)
#     for match, score in tournament_results.items():
#         print(f"{match}: {score}")
