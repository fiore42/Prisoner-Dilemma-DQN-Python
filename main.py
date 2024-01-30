# main.py

import argparse
import sys
from tournament import tournament, strategies_dict
from config import POINTS_SYSTEM

num_rounds = 1_000

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        if 'expected one argument' in message and '-a' in message or '--against' in message:
            self.print_help()
            print(f"\nError: {message}. Please specify a specific strategy to compete against with -a or --against.")
        else:
            self.print_help()
            print(f"\nError: {message}")
        sys.exit(2)

def main(verbose, very_verbose, opponent_strategies):
    if very_verbose: verbose = True #so I don't need to check for both each time
    # Run the tournament and unpack the results
    tournament_result = tournament(strategies_dict, POINTS_SYSTEM, num_rounds, verbose, very_verbose, opponent_strategies)

    if tournament_result is None:
        print("\nTournament did not produce any results.\n")
        # Handle the None case, e.g., exit the function or perform alternative actions
        return
    else:
        results, sorted_strategies = tournament_result
        # Continue with processing results and sorted_strategies    
    
    # # Print the selected strategies
    # selected_strategies_str = '\n'.join(selected_strategies)
    # print(f"\nSelected strategies for this tournament: \n{selected_strategies_str}")

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
        print(f"{name1:{max_length}} vs {name2:{max_length}}: {score1:5} - {score2:5} (Avg: {avg_score1:.2f} - {avg_score2:.2f}) Diff: {percent_diff:.2f}%")
        last_name1 = name1  # Update 'last_name1' for the next iteration

    # Find the maximum length of strategy names
    max_name_length = max(len(strategy) for strategy in strategies_dict.keys())

    # Find the maximum points for formatting
    max_points = max(total_points for _, (total_points, _, _) in sorted_strategies)

    # Length for points formatting
    max_points_length = len(str(max_points))

    # Always print sorted strategies
    print("\nSorted Strategies:")
    for strategy, (total_points, avg_points, delta_avg_points) in sorted_strategies:
        formatted_delta_avg_points = f", Delta Avg = {'{:=3d}'.format(int(delta_avg_points))}%" if not delta_avg_points == 0 else ""
        # formatted_delta_avg_points = f", Delta Avg = {delta_avg_points: .0f}%" if not delta_avg_points == 0 else ""
        string = f"{strategy:{max_name_length}}: Total Points = {total_points:{max_points_length}}, Avg Points/Game = {avg_points:.2f}{formatted_delta_avg_points}"
        # Check if the strategy is "rl_strategy" and apply ANSI bold if true
        if strategy == "rl_strategy":
            formatted_string = f"\033[1m{string}\033[0m"
        else:
            formatted_string = string       
        print (formatted_string) 

    print(f"")
    # for strategy, score in sorted_strategies:
    #     print(f"{strategy}: {score}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = CustomArgumentParser(description="Run a Prisoner's Dilemma tournament.")

    # parser = argparse.ArgumentParser(description="Run a Prisoner's Dilemma tournament.")
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('-vv', '--very-verbose', dest='very_verbose', action='store_true', help='Print very verbose output')
    #parser.add_argument('-a', '--against', type=str, default='', help='Specify the strategy to play against')
    # Use nargs to accept one or two strings for -a
    parser.add_argument('-a', '--against', nargs='+', default='', help='Specify the strategy/strategies to play against')

    args = parser.parse_args()

    # Run the main function with the specified verbosity level
    main(args.verbose, args.very_verbose, args.against)

