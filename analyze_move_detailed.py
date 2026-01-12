#!/usr/bin/env python3
"""
Detailed analysis of move evaluation components
"""

import numpy as np
from src.game import Game2048
from src.expectimax import preview_after_move, _format_board

# Current board state
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

print(_format_board(board_state))

def analyze_board(board, weights, label):
    """Break down evaluation components - calculates all weight components"""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    
    empties = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    empty_score = empties * weights['empty_spaces']
    
    # Corner bonus
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    corner_bonus = 0.0
    if max_tile > 0 and max_tile in corners:
        corner_bonus = weights['corner_bonus'] * (max_tile / 4.0)
    
    # Corner stability penalty
    corner_stability_penalty = 0.0
    if weights['corner_stability'] > 0 and max_tile > 0:
        if max_tile not in corners:
            corner_stability_penalty = max_tile * weights['corner_stability']
    
    # Snake pattern bonus
    snake_bonus = 0.0
    snake_pattern = [
        board[0, 0], board[0, 1], board[0, 2], board[0, 3],
        board[1, 3], board[1, 2], board[1, 1], board[1, 0],
        board[2, 0], board[2, 1], board[2, 2], board[2, 3],
        board[3, 3], board[3, 2], board[3, 1], board[3, 0]
    ]
    non_zero = [x for x in snake_pattern if x > 0]
    if len(non_zero) >= 2:
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                snake_bonus += weights['snake_pattern']
    
    # Monotonicity
    def mono_score(arr):
        s = 0.0
        for r in range(4):
            row = arr[r, :]
            seq_length = 0
            for i in range(3):
                if row[i] >= row[i + 1] and row[i] > 0:
                    seq_length += 1
                    s += seq_length * seq_length
                else:
                    seq_length = 0
        for c in range(4):
            col = arr[:, c]
            seq_length = 0
            for i in range(3):
                if col[i] >= col[i + 1] and col[i] > 0:
                    seq_length += 1
                    s += seq_length * seq_length
                else:
                    seq_length = 0
        return s * weights['monotonicity']
    
    monotonicity_score = mono_score(board)
    
    # Smoothness
    def smooth_score(arr):
        smooth = 0.0
        for r in range(4):
            for c in range(3):
                if arr[r, c] > 0 and arr[r, c + 1] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r, c + 1]))
                    smooth -= diff * weights['smoothness']
        for c in range(4):
            for r in range(3):
                if arr[r, c] > 0 and arr[r + 1, c] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r + 1, c]))
                    smooth -= diff * weights['smoothness']
        return smooth
    
    smoothness_score = smooth_score(board)
    
    # Merge potential
    merge_potential = 0.0
    merge_details = []
    for r in range(4):
        for c in range(3):
            if board[r, c] > 0 and board[r, c] == board[r, c + 1]:
                val = board[r, c] * weights['merge_potential']
                merge_potential += val
                merge_details.append(f"Row {r}: [{c}]-[{c+1}] = {board[r,c]} (bonus: {val:.2f})")
    for c in range(4):
        for r in range(3):
            if board[r, c] > 0 and board[r, c] == board[r + 1, c]:
                val = board[r, c] * weights['merge_potential']
                merge_potential += val
                merge_details.append(f"Col {c}: [{r}]-[{r+1}] = {board[r,c]} (bonus: {val:.2f})")
    
    # Merge execution bonus (hardcoded in expectimax)
    merge_execution_bonus = 0.0
    if max_tile > 0:
        max_pos = np.unravel_index(np.argmax(board), board.shape)
        max_r, max_c = max_pos
        adjacent_positions = [
            (max_r - 1, max_c), (max_r + 1, max_c),
            (max_r, max_c - 1), (max_r, max_c + 1)
        ]
        for r, c in adjacent_positions:
            if 0 <= r < 4 and 0 <= c < 4 and board[r, c] > 0:
                tile_value = board[r, c]
                bonus = tile_value * 0.1
                if (r, c) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
                    bonus *= 1.5
                merge_execution_bonus += bonus
    
    # Max tile bonus
    max_tile_bonus = 0.0
    if weights['max_tile_bonus'] > 0:
        max_tile_bonus = max_tile * weights['max_tile_bonus']
    
    # Edge bonus
    edge_bonus = 0.0
    if weights['edge_bonus'] > 0:
        edge_tiles = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] > 0 and (r == 0 or r == 3 or c == 0 or c == 3):
                    edge_tiles += board[r, c]
        edge_bonus = edge_tiles * weights['edge_bonus']
    
    # Total score
    total_score = (
        empty_score +
        corner_bonus +
        snake_bonus +
        monotonicity_score +
        smoothness_score +
        merge_potential +
        merge_execution_bonus +
        max_tile_bonus +
        edge_bonus -
        corner_stability_penalty
    )
    
    print(f"\nBoard:")
    print(_format_board(board))
    print(f"\nWeight Component Breakdown:")
    print(f"  Empty spaces ({empties} × {weights['empty_spaces']:.2f}):     {empty_score:8.2f}")
    print(f"  Corner bonus:                              {corner_bonus:8.2f}")
    print(f"  Corner stability penalty:                  {-corner_stability_penalty:8.2f}")
    print(f"  Snake pattern:                             {snake_bonus:8.2f}")
    print(f"  Monotonicity:                              {monotonicity_score:8.2f}")
    print(f"  Smoothness:                                {smoothness_score:8.2f}")
    print(f"  Merge potential:                           {merge_potential:8.2f}")
    print(f"  Merge execution bonus:                     {merge_execution_bonus:8.2f}")
    print(f"  Max tile bonus:                            {max_tile_bonus:8.2f}")
    print(f"  Edge bonus:                                {edge_bonus:8.2f}")
    print(f"  {'─'*60}")
    print(f"  TOTAL SCORE:                               {total_score:8.2f}")
    
    return {
        'empties': empties,
        'empty_score': empty_score,
        'corner_bonus': corner_bonus,
        'corner_stability_penalty': corner_stability_penalty,
        'snake_bonus': snake_bonus,
        'monotonicity_score': monotonicity_score,
        'smoothness_score': smoothness_score,
        'merge_potential': merge_potential,
        'merge_execution_bonus': merge_execution_bonus,
        'max_tile_bonus': max_tile_bonus,
        'edge_bonus': edge_bonus,
        'total_score': total_score,
    }

# Simulate LEFT
game_left = Game2048()
game_left.board = board_state.copy()
game_left.move('left')
board_left = game_left.board.copy()

# Simulate DOWN  
game_down = Game2048()
game_down.board = board_state.copy()
game_down.move('down')
board_down = game_down.board.copy()

print("\n" + "="*70)
print("WHY DID AI CHOOSE DOWN INSTEAD OF LEFT?")
print("="*70)

left_stats = analyze_board(board_left, weights, "LEFT MOVE")
down_stats = analyze_board(board_down, weights, "DOWN MOVE")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

comparison = [
    ('Empty spaces', 'empty_score', 'points'),
    ('Corner bonus', 'corner_bonus', 'points'),
    ('Corner stability penalty', 'corner_stability_penalty', 'points'),
    ('Snake pattern', 'snake_bonus', 'points'),
    ('Monotonicity', 'monotonicity_score', 'points'),
    ('Smoothness', 'smoothness_score', 'points'),
    ('Merge potential', 'merge_potential', 'points'),
    ('Merge execution bonus', 'merge_execution_bonus', 'points'),
    ('Max tile bonus', 'max_tile_bonus', 'points'),
    ('Edge bonus', 'edge_bonus', 'points'),
    ('TOTAL SCORE', 'total_score', 'points'),
]

print(f"\n{'Component':<30} {'LEFT':>12} {'DOWN':>12} {'Difference':>12}")
print(f"{'─'*30} {'─'*12} {'─'*12} {'─'*12}")

for label, key, unit in comparison:
    left_val = left_stats.get(key, 0.0)
    down_val = down_stats.get(key, 0.0)
    diff = down_val - left_val
    print(f"{label:<30} {left_val:>12.2f} {down_val:>12.2f} {diff:>12.2f}")

print(f"\n{'='*70}")
print("CHOSEN DIRECTION: DOWN")
print(f"{'='*70}")
print(f"\nAll weight component values for DOWN direction:\n")
print(f"  Empty spaces ({down_stats['empties']} × {weights['empty_spaces']:.2f}):     {down_stats['empty_score']:8.2f}")
print(f"  Corner bonus:                              {down_stats['corner_bonus']:8.2f}")
print(f"  Corner stability penalty:                  {-down_stats['corner_stability_penalty']:8.2f}")
print(f"  Snake pattern:                             {down_stats['snake_bonus']:8.2f}")
print(f"  Monotonicity:                              {down_stats['monotonicity_score']:8.2f}")
print(f"  Smoothness:                                {down_stats['smoothness_score']:8.2f}")
print(f"  Merge potential:                           {down_stats['merge_potential']:8.2f}")
print(f"  Merge execution bonus:                     {down_stats['merge_execution_bonus']:8.2f}")
print(f"  Max tile bonus:                            {down_stats['max_tile_bonus']:8.2f}")
print(f"  Edge bonus:                                {down_stats['edge_bonus']:8.2f}")
print(f"  {'─'*60}")
print(f"  TOTAL SCORE:                               {down_stats['total_score']:8.2f}")

