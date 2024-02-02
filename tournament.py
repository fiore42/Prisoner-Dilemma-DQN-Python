# tournament.py

import random
import inspect
import strategies
import sys
from importlib import import_module
# from reinforcement_learning import train_model, print_q_table
from reinforcement_learning import PrisonersDilemmaDQN

# Create a global instance of the DQN agent
dqn_agent = PrisonersDilemmaDQN()

def play_round(strategy1, name1, strategy2, name2, history1, history2, points_system, very_verbose):

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
    score1, score2 = points_system[move1+move2]

    return score1, score2

def tournament(strategies, points_system, num_rounds, verbose, very_verbose, opponent_strategies):
    results_temp = {}
    hands_played = {}

    # Remove 'rl_strategy' from the pool to avoid duplicating it
    available_strategies = [name for name in strategies.keys() if name != 'rl_strategy']

    if not opponent_strategies:

        # Randomly select 8 other strategies
        randomly_selected_strategies = random.sample(available_strategies, min(8, len(available_strategies)))

        # Add 'rl_strategy' to the list of selected strategies
        selected_strategies = ['rl_strategy'] + randomly_selected_strategies
    else:
        # Check if all opponent strategies are available
        if all(strategy in available_strategies for strategy in opponent_strategies):
            # Make selected_strategies = 'rl_strategy' and opponent strategies
            selected_strategies = ['rl_strategy'] + opponent_strategies
        else:
            # Print an error and exit the function
            print("\nSelect valid opponent strategies. Some of the specified strategies are not available.")
            print("\nAvailable strategies are:")
            for strategy in available_strategies:
                print(f"- {strategy}")
            return None

    # Randomly select 9 strategies for the tournament
    # selected_strategies = random.sample(list(strategies.keys()), min(9, len(strategies)))

    strategy_points = {name: (0, 0) for name in selected_strategies} # total points, number of games

    # Print the selected strategies
    selected_strategies_str = '\n'.join(selected_strategies)
    print(f"\nSelected strategies for this tournament: \n{selected_strategies_str}")

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
                score1, score2 = play_round(strategies[name1], name1, strategies[name2], name2, history1, history2, points_system, very_verbose)
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

            percent_diff = (total_score1 - total_score2) / max(total_score1, total_score2) * 100 if max(total_score1, total_score2) > 0 else 0

            results_temp[(name1, name2)] = (total_score1, total_score2, total_score1/num_rounds, total_score2/num_rounds, percent_diff)
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
strategies_dict = build_strategies_dict()

# # Points system for the Prisoner's Dilemma
# points_system = {'CC': (3, 3), 'CD': (0, 5), 'DC': (5, 0), 'DD': (1, 1)}

# # Run a single tournament as an example
# if __name__ == "__main__":
#     tournament_results = tournament(strategies_dict, points_system)
#     for match, score in tournament_results.items():
#         print(f"{match}: {score}")
