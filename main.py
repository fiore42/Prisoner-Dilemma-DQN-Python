# main.py

import argparse
import sys
from tournament import tournament, print_results, print_strategies
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


def main(opponent_strategies, debug, verbose, very_verbose):

    if debug or DEBUG:
        num_rounds = 200
        max_strategies = 3
    else:
        num_rounds = 1_000
        max_strategies = 9

    if very_verbose: verbose = True #so I don't need to check for both each time
    # Run the tournament and unpack the results
    results, sorted_strategies = tournament(num_rounds, max_strategies, opponent_strategies, verbose, very_verbose)

    # Check if the variables are not empty
    if results and sorted_strategies:
        # Do something with the non-empty variables
        print_results (results, sorted_strategies)
        return
    else:
        print("\nTournament did not produce any results.\n")
        return


import argparse
import sys

def handle_arguments():
    parser = argparse.ArgumentParser(description='Handle arguments for the application')
    
    parser.add_argument('-a', '--against', nargs='*', default='', help='Specify the strategy/strategies to play against.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Turn on debug. Less strategies, less rounds.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output.')
    parser.add_argument('-vv', '--very-verbose', action='store_true', default=False, help='Very verbose output. Includes ML output.')
    parser.add_argument('-p', '--print', action='store_true', default=False, help='Print all available strategies.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.very_verbose:
        args.verbose = True

    # Uncomment to print the values for debugging
    # print(f"Verbose: {args.verbose}")
    # print(f"Very Verbose: {args.very_verbose}")
    # print(f"Print strategies: {args.print}")
    # print(f"Against: {args.against}")

    # Return the parsed arguments as needed
    return args.print, args.against, args.debug, args.verbose, args.very_verbose

# Example of how to use this function
if __name__ == "__main__":

    print_strategies_flag, opponent_strategies, debug, verbose, very_verbose = handle_arguments()

    if print_strategies_flag :
        print_strategies()
        sys.exit()
    else:
        # Run the main function with the specified verbosity level
        # sys.exit()
        main(opponent_strategies, debug, verbose, very_verbose)
        sys.exit()
