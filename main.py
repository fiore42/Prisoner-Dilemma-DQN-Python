# main.py

import argparse
import sys
from tournament import tournament, print_results
from config import DEBUG


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

    if DEBUG:
        num_rounds = 200
        max_strategies = 3
    else:
        num_rounds = 1_000
        max_strategies = 9

    if very_verbose: verbose = True #so I don't need to check for both each time
    # Run the tournament and unpack the results
    results, sorted_strategies = tournament(num_rounds, max_strategies, opponent_strategies, verbose, very_verbose)

    # tournament_result = tournament(num_rounds, max_strategies, opponent_strategies, verbose, very_verbose)

    # Check if the variables are not empty
    if results and sorted_strategies:
        # Do something with the non-empty variables
        print_results (results, sorted_strategies)
        return
    else:
        print("\nTournament did not produce any results.\n")
        return


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

