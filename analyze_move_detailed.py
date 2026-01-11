#!/usr/bin/env python3
"""
Detailed analysis of move evaluation components
"""

import numpy as np
from src.game import Game2048
from src.expectimax import preview_after_move, _format_board

# Current board state
board_state = np.array([
    [1024,    4,    2,    4],
    [ 512,  128,   16,    2],
    [  16,   16,    2,    0],
    [  64,    4,    0,    0]
])

# Weights from 'ES 2.75 (Best)' config
weights = {
    'empty_spaces': 2.75,
    'corner_bonus': 8.0,
    'corner_stability': 2.75,
    'snake_pattern': 2.0,
    'monotonicity': 0.5,
    'smoothness': 0.1,
    'merge_potential': 0.1,
    'max_tile_bonus': 0.0,
    'edge_bonus': 0.0
}

def analyze_board(board, weights, label):
    """Break down evaluation components"""
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
    
    print(f"\nBoard:")
    print(_format_board(board))
    print(f"\nEmpty spaces: {empties} × {weights['empty_spaces']} = {empty_score:.2f}")
    print(f"Corner bonus (max_tile={max_tile}): {corner_bonus:.2f}")
    print(f"Merge potential: {merge_potential:.2f}")
    if merge_details:
        print(f"  Details:")
        for detail in merge_details:
            print(f"    {detail}")
    else:
        print(f"  (No merge opportunities)")
    
    return {
        'empties': empties,
        'empty_score': empty_score,
        'corner_bonus': corner_bonus,
        'merge_potential': merge_potential,
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
print(f"\nEmpty spaces:")
print(f"  LEFT: {left_stats['empties']} → {left_stats['empty_score']:.2f} points")
print(f"  DOWN: {down_stats['empties']} → {down_stats['empty_score']:.2f} points")
print(f"  Difference (LEFT advantage): {left_stats['empty_score'] - down_stats['empty_score']:.2f}")

print(f"\nCorner bonus:")
print(f"  LEFT: {left_stats['corner_bonus']:.2f} points")
print(f"  DOWN: {down_stats['corner_bonus']:.2f} points")
print(f"  Difference: {down_stats['corner_bonus'] - left_stats['corner_bonus']:.2f}")

print(f"\nMerge potential:")
print(f"  LEFT: {left_stats['merge_potential']:.2f} points")
print(f"  DOWN: {down_stats['merge_potential']:.2f} points")
print(f"  Difference (DOWN advantage): {down_stats['merge_potential'] - left_stats['merge_potential']:.2f}")

net_empty = left_stats['empty_score'] - down_stats['empty_score']
net_merge = down_stats['merge_potential'] - left_stats['merge_potential']
net_total = net_merge - net_empty

print(f"\n{'='*70}")
print("KEY INSIGHT")
print(f"{'='*70}")
print(f"\nLEFT advantage (empty spaces): +{net_empty:.2f}")
print(f"DOWN advantage (merge potential): +{net_merge:.2f}")
print(f"Net (merge - empty): {net_total:.2f}")

print(f"\n⚠️  However, note that expectimax looks ahead 2 moves (depth=2).")
print(f"   The evaluation shown above is just the terminal evaluation.")
print(f"   Expectimax also considers what happens after random tile spawns,")
print(f"   which might favor DOWN's merge opportunities over LEFT's immediate merge.")

print(f"\n📝 Your intuition (LEFT to merge the two 16s) is valid - you get:")
print(f"   - Immediate 32 tile (concrete gain)")
print(f"   - +1 empty space (3 vs 2)")
print(f"   - But fewer future merge opportunities")

print(f"\n🤖 AI's reasoning (DOWN):")
print(f"   - Preserves more merge opportunities (3 vs 0)")
print(f"   - Creates potential for multiple merges next turn")
print(f"   - But loses 1 empty space")
print(f"   - The lookahead calculation suggests DOWN leads to better future states")

