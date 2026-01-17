#!/usr/bin/env python3
"""
Debug script to understand snake pattern evaluation in detail
"""

import numpy as np
from src.game import Game2048

# Current board state
board_state = np.array([
    [   2,    0,    0,   64],
    [   0,    0,   16,  128],
    [   0,    4,   32,  512],
    [   2,    2,   64, 1024]
])

# Weights
weights = {
    'snake_pattern': 1.0,
}

SNAKE_SCALE_FACTOR = 0.3

def get_snake_sequences(transformed_board):
    """Get horizontal and vertical snake sequences for a transformed board"""
    # Horizontal snake: Row 0 L→R, Row 1 R→L, Row 2 L→R, Row 3 R→L
    snake_h = [
        transformed_board[0, 0], transformed_board[0, 1], transformed_board[0, 2], transformed_board[0, 3],
        transformed_board[1, 3], transformed_board[1, 2], transformed_board[1, 1], transformed_board[1, 0],
        transformed_board[2, 0], transformed_board[2, 1], transformed_board[2, 2], transformed_board[2, 3],
        transformed_board[3, 3], transformed_board[3, 2], transformed_board[3, 1], transformed_board[3, 0]
    ]
    # Vertical snake: Col 0 T→B, Col 1 B→T, Col 2 T→B, Col 3 B→T
    snake_v = [
        transformed_board[0, 0], transformed_board[1, 0], transformed_board[2, 0], transformed_board[3, 0],
        transformed_board[3, 1], transformed_board[2, 1], transformed_board[1, 1], transformed_board[0, 1],
        transformed_board[0, 2], transformed_board[1, 2], transformed_board[2, 2], transformed_board[3, 2],
        transformed_board[3, 3], transformed_board[2, 3], transformed_board[1, 3], transformed_board[0, 3]
    ]
    return snake_h, snake_v

def evaluate_snake_pattern_detailed(snake_sequence, weight, scale, orientation_name):
    """Evaluate a snake pattern sequence with detailed output"""
    non_zero = [x for x in snake_sequence if x > 0]
    if len(non_zero) < 2:
        return 0.0, []
    
    bonus = 0.0
    details = []
    chain_length = 0
    
    for i in range(len(non_zero) - 1):
        if non_zero[i] >= non_zero[i + 1]:
            val1 = non_zero[i]
            log_val = np.log2(val1) if val1 > 0 else 0
            pair_bonus = weight * (1 + scale * log_val)
            bonus += pair_bonus
            chain_length += 1
            details.append({
                'index': i,
                'val1': val1,
                'val2': non_zero[i + 1],
                'log_val': log_val,
                'pair_bonus': pair_bonus,
                'cumulative': bonus,
                'chain_length': chain_length
            })
        else:
            details.append({
                'index': i,
                'val1': non_zero[i],
                'val2': non_zero[i + 1],
                'log_val': None,
                'pair_bonus': 0.0,
                'cumulative': bonus,
                'chain_length': chain_length,
                'broken': True
            })
            break
    
    return bonus, details, chain_length

# Simulate moves
game_up = Game2048()
game_up.board = board_state.copy()
game_up.move('up')
board_up = game_up.board.copy()

game_right = Game2048()
game_right.board = board_state.copy()
game_right.move('right')
board_right = game_right.board.copy()

print("=" * 80)
print("SNAKE PATTERN DETAILED ANALYSIS")
print("=" * 80)

orientations = [
    ("UL (Upper-Left)", lambda b: b),
    ("UR (Upper-Right)", lambda b: np.flip(b, axis=1)),
    ("LL (Lower-Left)", lambda b: np.flip(b, axis=0)),
    ("LR (Lower-Right)", lambda b: np.flip(np.flip(b, axis=0), axis=1)),
]

def analyze_board_snake(board, label):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"\nBoard:\n{board}")
    
    all_scores = {}
    
    for orient_name, transform_func in orientations:
        transformed = transform_func(board)
        snake_h, snake_v = get_snake_sequences(transformed)
        
        bonus_h, details_h, chain_h = evaluate_snake_pattern_detailed(
            snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, f"{orient_name}-H"
        )
        bonus_v, details_v, chain_v = evaluate_snake_pattern_detailed(
            snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, f"{orient_name}-V"
        )
        
        all_scores[f"{orient_name}-H"] = (bonus_h, chain_h, details_h, snake_h)
        all_scores[f"{orient_name}-V"] = (bonus_v, chain_v, details_v, snake_v)
    
    # Find best
    best_key = max(all_scores.keys(), key=lambda k: all_scores[k][0])
    best_bonus, best_chain, best_details, best_sequence = all_scores[best_key]
    
    print(f"\n📊 BEST ORIENTATION: {best_key}")
    print(f"   Score: {best_bonus:.2f}")
    print(f"   Chain Length: {best_chain} pairs")
    print(f"   Sequence (non-zero): {[x for x in best_sequence if x > 0]}")
    
    print(f"\n   Detailed chain breakdown:")
    for detail in best_details:
        if detail.get('broken'):
            print(f"      Pair {detail['index']}: {detail['val1']} < {detail['val2']} → CHAIN BROKEN")
        else:
            print(f"      Pair {detail['index']}: {detail['val1']:4d} >= {detail['val2']:4d} → {detail['pair_bonus']:.2f} (log2={detail['log_val']:.1f}, cumulative={detail['cumulative']:.2f}, chain_len={detail['chain_length']})")
    
    print(f"\n   All orientations:")
    for key, (bonus, chain, details, seq) in sorted(all_scores.items(), key=lambda x: x[1][0], reverse=True):
        non_zero_seq = [x for x in seq if x > 0]
        print(f"      {key:20s}: {bonus:6.2f} (chain={chain:2d}, seq={non_zero_seq})")
    
    return best_bonus, best_key, best_chain

print("\n" + "="*80)
print("UP MOVE ANALYSIS")
print("="*80)
up_score, up_orient, up_chain = analyze_board_snake(board_up, "UP MOVE")

print("\n" + "="*80)
print("RIGHT MOVE ANALYSIS")
print("="*80)
right_score, right_orient, right_chain = analyze_board_snake(board_right, "RIGHT MOVE")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"UP:    Score={up_score:.2f}, Orientation={up_orient}, Chain Length={up_chain}")
print(f"RIGHT: Score={right_score:.2f}, Orientation={right_orient}, Chain Length={right_chain}")
print(f"Difference: {up_score - right_score:.2f}")
print(f"\nWhy is the difference so small?")
print(f"   - Both moves create similar snake patterns in their best orientations")
print(f"   - The algorithm takes the MAX of 8 orientations (4 corners × 2 directions)")
print(f"   - Even though UP creates a longer chain in one orientation, RIGHT may have")
print(f"     a similar pattern in another orientation that scores nearly as well")

