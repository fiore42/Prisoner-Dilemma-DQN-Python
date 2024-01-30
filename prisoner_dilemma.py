
def prisoners_dilemma(player1_choice, player2_choice, points):
    """
    Determines the outcome of a single round of the Prisoner's Dilemma game.

    Args:
    player1_choice (str): The choice of player 1, 'C' for Cooperate or 'D' for Defect.
    player2_choice (str): The choice of player 2, 'C' for Cooperate or 'D' for Defect.
    points (dict): A dictionary with key as a string concatenation of player choices
                   and value as a tuple (player1_points, player2_points).

    Returns:
    tuple: Points earned by player 1 and player 2 in this round.
    """
    return points[player1_choice + player2_choice]

if __name__ == '__main__':
    points = {
        'CC': (3, 3),
        'CD': (0, 5),
        'DC': (5, 0),
        'DD': (1, 1) 
        }
    p1_choice, p2_choice = ('C', 'D')
    result = prisoners_dilemma(p1_choice, p2_choice, points)
    print(f'''Outcome of Player 1 choosing {p1_choice} and Player 2 choosing {p2_choice}: {result}''')

