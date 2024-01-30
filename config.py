# config.py

# Points system for the Prisoner's Dilemma
POINTS_SYSTEM = {'CC': (3, 3), 'CD': (0, 5), 'DC': (5, 0), 'DD': (1, 1)}

    # T (for temptation to betray a cooperating opponent)
    # R (for reward when both players cooperate with each other).
    # S (for sucker’s payoff when being betrayed while cooperating oneself), 
    # P (for punishment when both players betray each other), and 
    # S < P < R < T and 2R > T + S
