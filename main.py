# main.py

import argparse
import sys
from tournament import tournament, print_results, print_strategies
from config import DEBUG
import time
import numpy as np
from collections import namedtuple

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        if 'expected one argument' in message and '-a' in message or '--against' in message:
            self.print_help()
            print(f"\nError: {message}. Please specify a specific strategy to compete against with -a or --against.")
        else:
            self.print_help()
            print(f"\nError: {message}")
        sys.exit(2)


# def main(opponent_strategies, debug, infinite_loop, no_bold, verbose, very_verbose):
def main(opponent_strategies, debug, infinite_loop, no_bold, verbose, very_verbose, learning_rate=None, gamma=None):

    if debug or DEBUG:
        num_rounds = 200
        max_strategies = 3
    else:
        num_rounds = 1_000
        max_strategies = 9

    Flags = namedtuple('Flags', ['infinite_loop', 'no_bold', 'explore'])
    if learning_rate is None and gamma is None:
        flags = Flags(infinite_loop=infinite_loop, no_bold=no_bold, explore=False)
    else:
        flags = Flags(infinite_loop=infinite_loop, no_bold=no_bold, explore=True)

    # print (flags)

    HP = namedtuple('HP', ['lr', 'gamma'])


    if flags.explore: # set Hyperparameters
        hp = HP(lr=learning_rate,gamma=gamma)
        # print (hp)
    else:
        hp = HP(lr=None,gamma=None)

    results, sorted_strategies = tournament(num_rounds, max_strategies, opponent_strategies, flags, verbose, very_verbose, hp)

    # Check if the variables are not empty
    if results and sorted_strategies:
        # Do something with the non-empty variables
        print_results (flags, results, sorted_strategies)
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
    parser.add_argument('-l', '--loop', action='store_true', default=False, help='Starts an infinite loop and prints ML performance to help debugging.')
    parser.add_argument('-b', '--no-bold', action='store_true', default=False, help='Avoid printing in bold. Useful when you redirect output to file.')
    parser.add_argument('-e', '--explore', action='store_true', default=False, help='Explore the impact of changing ML hyperparameters.')
    
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
    return args.print, args.against, args.debug, args.loop, args.no_bold, args.explore, args.verbose, args.very_verbose

# Example of how to use this function
if __name__ == "__main__":

    print_strategies_flag, opponent_strategies, debug, infinite_loop, no_bold, explore, verbose, very_verbose = handle_arguments()

    if print_strategies_flag :
        print_strategies()
        sys.exit()
    else:
        # Run the main function with the specified verbosity level
        # sys.exit()
        if not infinite_loop and not explore:
            main(opponent_strategies, debug, infinite_loop, no_bold, verbose, very_verbose)
            sys.exit()
        else: 
            if infinite_loop:
                while True:
                    # infinite loop doesn't allow any argument expect -d
                    verbose = False 
                    very_verbose = False
                    opponent_strategies = []
                    main(opponent_strategies, debug, infinite_loop, no_bold, verbose, very_verbose)
                    time.sleep (1)
            if explore:
                opponent_strategies = ['grudger_recovery','always_defect','tit_for_tat_trustful']
                opponent_strategies.extend(['random_strategy','provocateur','tit_for_tat_opposite_def'])
                opponent_strategies.extend(['tit_for_tat_suspicious','alternate3and3','alternate_coop','alternate_def'])
                for lr in np.arange(0, 1.01, 0.05):  # Goes from 0 to 1 inclusive, in steps of 0.01
                    for gamma in np.arange(0, 1.01, 0.05):
                        print(f"LR: {lr:.2f}, GAMMA: {gamma:.2f}")
                        main(opponent_strategies, debug, infinite_loop, no_bold, verbose, very_verbose, lr, gamma)
