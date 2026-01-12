#!/usr/bin/env python3
"""
Analyze why the AI chose a particular move
"""

import numpy as np
from src.game import Game2048
from src.expectimax import expectimax_best_action_tunable, evaluate_board_tunable, preview_after_move, _format_board

# Current board state from move 304
board_state = np.array([
    [ 512,   16,    2,    4],
    [  16,   64,    4,    2],
    [   8,   16,    8,    2],
    [   4,    4,    2,    0]
])

# Weights from 'no merge potential' config
weights = {
    'empty_spaces': 2.75,
    'corner_bonus': 8.0,
    'corner_stability': 2.75,
    'snake_pattern': 0.0,
    'monotonicity': 1.0,
    'smoothness': 0.1,
    'merge_potential': 0.0,  # No bonus for merge opportunities!
    'max_tile_bonus': 0.0,
    'edge_bonus': 0.0
}

print("=" * 70)
print("ANALYZING MOVE DECISION")
print("=" * 70)
print("\nCurrent Board State:")
print(_format_board(board_state))
print()

# Simulate LEFT move
game_left = Game2048()
game_left.board = board_state.copy()
left_success = game_left.move('left')
board_after_left = game_left.board.copy()

print("=" * 70)
print("LEFT MOVE")
print("=" * 70)
print("Board after LEFT:")
print(_format_board(board_after_left))
print(f"Move successful: {left_success}")
print()

# Simulate DOWN move
game_down = Game2048()
game_down.board = board_state.copy()
down_success = game_down.move('down')
board_after_down = game_down.board.copy()

print("=" * 70)
print("DOWN MOVE")
print("=" * 70)
print("Board after DOWN:")
print(_format_board(board_after_down))
print(f"Move successful: {down_success}")
print()

# Evaluate both board states (terminal evaluation, depth 0)
print("=" * 70)
print("EVALUATION (Terminal, depth 0)")
print("=" * 70)
eval_left = evaluate_board_tunable(board_after_left, weights=weights, debug=False)
eval_down = evaluate_board_tunable(board_after_down, weights=weights, debug=False)

print(f"\nLEFT move evaluation:  {eval_left:.2f}")
print(f"DOWN move evaluation:  {eval_down:.2f}")
print(f"Difference (DOWN - LEFT): {eval_down - eval_left:.2f}")
print()

if eval_down > eval_left:
    print(f"✅ AI chose DOWN (eval: {eval_down:.2f} > {eval_left:.2f})")
else:
    print(f"❌ AI chose DOWN but LEFT has higher eval ({eval_left:.2f} > {eval_down:.2f})")
    print("   This suggests expectimax looked ahead deeper than depth 0")

# Now let's see what expectimax actually chooses
print("\n" + "=" * 70)
print("EXPECTIMAX DECISION (depth=2, chance_samples=8)")
print("=" * 70)
chosen_action = expectimax_best_action_tunable(
    board_state,
    depth=2,
    chance_sample_k=8,
    weights=weights,
    debug=False
)

directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
print(f"\nExpectimax chooses: {directions[chosen_action]}")
print()

# Count empty spaces after each move
empties_left = np.count_nonzero(board_after_left == 0)
empties_down = np.count_nonzero(board_after_down == 0)

print("=" * 70)
print("DETAILED COMPARISON")
print("=" * 70)
print(f"\nEmpty spaces after LEFT:  {empties_left}")
print(f"Empty spaces after DOWN:  {empties_down}")
print(f"Difference: {empties_down - empties_left}")

# Check if max tile stays in corner
max_tile = np.max(board_state)
corners_left = [board_after_left[0, 0], board_after_left[0, 3], board_after_left[3, 0], board_after_left[3, 3]]
corners_down = [board_after_down[0, 0], board_after_down[0, 3], board_after_down[3, 0], board_after_down[3, 3]]

print(f"\nMax tile ({max_tile}) in corner after LEFT:  {max_tile in corners_left}")
print(f"Max tile ({max_tile}) in corner after DOWN:  {max_tile in corners_down}")

# Check merge opportunities
print("\nMerge opportunities:")
print("LEFT: ", end="")
merges_left = 0
for r in range(4):
    for c in range(3):
        if board_after_left[r, c] > 0 and board_after_left[r, c] == board_after_left[r, c + 1]:
            merges_left += 1
            print(f"[{r},{c}]-[{r},{c+1}]={board_after_left[r,c]} ", end="")
for c in range(4):
    for r in range(3):
        if board_after_left[r, c] > 0 and board_after_left[r, c] == board_after_left[r + 1, c]:
            merges_left += 1
            print(f"[{r},{c}]-[{r+1},{c}]={board_after_left[r,c]} ", end="")
if merges_left == 0:
    print("none")
else:
    print(f"({merges_left} total)")

print("DOWN: ", end="")
merges_down = 0
for r in range(4):
    for c in range(3):
        if board_after_down[r, c] > 0 and board_after_down[r, c] == board_after_down[r, c + 1]:
            merges_down += 1
            print(f"[{r},{c}]-[{r},{c+1}]={board_after_down[r,c]} ", end="")
for c in range(4):
    for r in range(3):
        if board_after_down[r, c] > 0 and board_after_down[r, c] == board_after_down[r + 1, c]:
            merges_down += 1
            print(f"[{r},{c}]-[{r+1},{c}]={board_after_down[r,c]} ", end="")
if merges_down == 0:
    print("none")
else:
    print(f"({merges_down} total)")

