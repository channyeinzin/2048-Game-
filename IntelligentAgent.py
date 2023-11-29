import random
import time
import math
from BaseAI import BaseAI


def simulate_move(grid, move):
    new_grid = grid.clone()
    new_grid.move(move)
    return new_grid


def simulate_tile_insertion(grid, cell, value):
    new_grid = grid.clone()
    new_grid.insertTile(cell, value)
    return new_grid


def free_spaces(grid):
    return len(grid.getAvailableCells())


def weight(grid):
    weight_matrix = [
        [math.pow(4, 15), math.pow(4, 14), math.pow(4, 13), math.pow(4, 12)],
        [math.pow(4, 8), math.pow(4, 9), math.pow(4, 10), math.pow(4, 11)],
        [math.pow(4, 7), math.pow(4, 6), math.pow(4, 5), math.pow(4, 4)],
        [math.pow(4, 0), math.pow(4, 1), math.pow(4, 2), math.pow(4, 3)]
    ]

    score = 0
    for i in range(4):
        for j in range(4):
            score += weight_matrix[i][j] * grid.map[i][j]
    return score


def evaluate(grid):
    return weight(grid) + (5 * free_spaces(grid))


def get_chance_nodes(grid):
    chance_nodes = []
    for cell in grid.getAvailableCells():
        new_grid2 = simulate_tile_insertion(grid, cell, 2)
        new_grid4 = simulate_tile_insertion(grid, cell, 4)
        chance_nodes.append((new_grid2, 0.9))  # 90% chance for a 2 tile
        chance_nodes.append((new_grid4, 0.1))  # 10% chance for a 4 tile
    return chance_nodes


class IntelligentAgent(BaseAI):
    def __init__(self):
        self.max_time = 0.01  # Time limit for decision-making

    def getMove(self, grid):
        start_time = time.perf_counter()
        best_move, _ = self.iterative_deepen(grid, start_time)
        return best_move if best_move is not None else random.choice(grid.getAvailableMoves())[0]

    def iterative_deepen(self, grid, start_time):
        depth = 0
        best_move = None
        best_utility = -math.inf

        # Iterative deepening search
        while time.perf_counter() - start_time < self.max_time:
            depth += 1
            utility, move = self.maximize(grid, depth, -math.inf, math.inf, start_time)

            if utility > best_utility:
                best_utility = utility
                best_move = move

        return best_move, best_utility

    def maximize(self, grid, depth, alpha, beta, start_time):
        if not grid.canMove() or depth == 0 or time.perf_counter() - start_time > self.max_time:
            return evaluate(grid), None

        max_utility = -math.inf
        best_move = None
        moves = sorted(grid.getAvailableMoves(), key=lambda move: evaluate(simulate_move(grid, move[0])),
                       reverse=True)

        for move in moves:
            utility = self.minimize(simulate_move(grid, move[0]), depth - 1, alpha, beta, start_time)
            if utility > max_utility:
                max_utility = utility
                best_move = move[0]

            if max_utility >= beta:
                break

            alpha = max(alpha, max_utility)
        return max_utility, best_move

    def minimize(self, grid, depth, alpha, beta, start_time):
        if not grid.canMove() or depth == 0 or time.perf_counter() - start_time > self.max_time:
            return evaluate(grid)

        min_utility = math.inf
        chance_nodes = get_chance_nodes(grid)

        for child, probability in chance_nodes:
            utility = self.maximize(child, depth - 1, alpha, beta, start_time)[0]
            min_utility = min(min_utility, utility * probability)

            if min_utility <= alpha:
                break

            beta = min(beta, min_utility)
        return min_utility

    def Expectiminimax(grid, depth, t1):
        global it
        modes = [1, 0, -1]
        state = modes[it]

        if state > 0:
            val = grid.maximize(depth, t1, -math.inf, math.inf)
        elif state == 0:
            val = get_chance_nodes(grid)
        else:
            val = grid.minimize(depth, t1, -math.inf, math.inf)

        it = (it + 1) % 3  # Update the iteration counter in a more concise way

        return val