#!/usr/bin/env python3
"""
Detailed analysis of why UP's longer chain doesn't score much better
"""

import numpy as np
from src.expectimax import preview_after_move, _format_board

# Current board state
board_state = np.array([
    [   2,    0,    0,   64],
    [   0,    0,   16,  128],
    [   0,    4,   32,  512],
    [   2,    2,   64, 1024]
])

weights = {'snake_pattern': 1.0}
SNAKE_SCALE_FACTOR = 0.3

def get_snake_sequences(transformed_board):
    """Get horizontal and vertical snake sequences"""
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

def evaluate_snake_pattern(snake_sequence, weight, scale):
    """Evaluate snake pattern - stops at first break"""
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
                'pair': (non_zero[i], non_zero[i + 1]),
                'bonus': pair_bonus,
                'chain_len': chain_length,
                'cumulative': bonus
            })
        else:
            details.append({
                'pair': (non_zero[i], non_zero[i + 1]),
                'bonus': 0.0,
                'chain_len': chain_length,
                'cumulative': bonus,
                'broken': True
            })
            break
    
    return bonus, details, chain_length

board_up = preview_after_move(board_state, 0)  # UP
board_right = preview_after_move(board_state, 3)  # RIGHT

print("=" * 80)
print("WHY IS THE SNAKE PATTERN DIFFERENCE SO SMALL?")
print("=" * 80)

print("\n📊 BOARD STATES:")
print("\nAfter UP move:")
print(_format_board(board_up))
print(f"\nTop row: {board_up[0]}")
print(f"  This creates a decreasing pattern: 64 → 16 → 4 → 4 (4 tiles in sequence)")

print("\nAfter RIGHT move:")
print(_format_board(board_right))
print(f"\nRight column: {[board_right[r, 3] for r in range(4)]}")
print(f"  This creates an increasing pattern (decreasing when read bottom-to-top): 1024 → 512 → 128 → 64")

# Analyze all orientations for both
orientations = [
    ("UL", lambda b: b),
    ("UR", lambda b: np.flip(b, axis=1)),
    ("LL", lambda b: np.flip(b, axis=0)),
    ("LR", lambda b: np.flip(np.flip(b, axis=0), axis=1)),
]

def analyze_all_orientations(board, label):
    print(f"\n{'='*80}")
    print(f"{label} - ALL ORIENTATIONS")
    print(f"{'='*80}")
    
    all_results = {}
    
    for orient_name, transform_func in orientations:
        transformed = transform_func(board)
        snake_h, snake_v = get_snake_sequences(transformed)
        
        bonus_h, details_h, chain_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR)
        bonus_v, details_v, chain_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR)
        
        all_results[f"{orient_name}-H"] = (bonus_h, chain_h, details_h, snake_h)
        all_results[f"{orient_name}-V"] = (bonus_v, chain_v, details_v, snake_v)
    
    # Show all results
    for key in sorted(all_results.keys(), key=lambda k: all_results[k][0], reverse=True):
        bonus, chain, details, seq = all_results[key]
        non_zero = [x for x in seq if x > 0]
        print(f"\n  {key}: Score={bonus:.2f}, Chain={chain} pairs")
        print(f"    Sequence: {non_zero}")
        if details:
            print(f"    Chain breakdown:")
            for d in details[:chain]:
                print(f"      {d['pair'][0]} >= {d['pair'][1]} → {d['bonus']:.2f} (chain_len={d['chain_len']}, cumulative={d['cumulative']:.2f})")
    
    best_key = max(all_results.keys(), key=lambda k: all_results[k][0])
    best_bonus, best_chain, best_details, best_seq = all_results[best_key]
    
    print(f"\n  ✅ BEST: {best_key} with score {best_bonus:.2f} (chain length: {best_chain})")
    
    return best_bonus, best_key, best_chain, all_results

up_best, up_key, up_chain, up_all = analyze_all_orientations(board_up, "UP MOVE")
right_best, right_key, right_chain, right_all = analyze_all_orientations(board_right, "RIGHT MOVE")

print(f"\n{'='*80}")
print("KEY INSIGHT")
print(f"{'='*80}")

print(f"\nUP best:    {up_key:10s} → Score: {up_best:.2f}, Chain: {up_chain} pairs")
print(f"RIGHT best: {right_key:10s} → Score: {right_best:.2f}, Chain: {right_chain} pairs")
print(f"Difference: {up_best - right_best:.2f}")

print(f"\n🔍 WHY THE DIFFERENCE IS SMALL:")
print(f"\n1. The algorithm takes the MAXIMUM score from all 8 orientations")
print(f"   (4 corner positions × 2 directions = 8 total)")
print(f"\n2. Both UP and RIGHT create strong patterns:")
print(f"   - UP's top row [4, 4, 16, 64] creates a decreasing sequence in one orientation")
print(f"   - RIGHT's right column [64, 128, 512, 1024] creates a decreasing sequence")
print(f"     (when read in the appropriate snake orientation)")
print(f"\n3. The chain evaluation stops at the first break:")
print(f"   - Even if UP creates a 7-tile chain, if it breaks early in some orientations,")
print(f"     other orientations might score better")
print(f"\n4. The scoring formula is:")
print(f"   bonus = weight * (1 + scale * log2(tile_value))")
print(f"   - With weight=1.0 and scale=0.3")
print(f"   - The bonus scales logarithmically with tile value")
print(f"   - Large tiles (1024, 512) contribute more per pair")
print(f"\n5. RIGHT's pattern [1024→512→128→64] has higher-value tiles")
print(f"   - Each pair contributes: 4.00 + 3.70 + 3.10 + 2.80 = 13.60")
print(f"   - UP's pattern might have lower-value tiles in its best orientation")
print(f"   - So even a longer chain with smaller tiles might score similarly")

# Let's specifically check what UP's top row pattern scores
print(f"\n{'='*80}")
print("CHECKING UP'S TOP ROW PATTERN SPECIFICALLY")
print(f"{'='*80}")

# The top row [4, 4, 16, 64] in UP board
# In UL-H orientation, this would be the first 4 elements
up_snake_ul_h = get_snake_sequences(board_up)[0]  # Horizontal snake, UL orientation
up_nonzero_ul_h = [x for x in up_snake_ul_h if x > 0]
print(f"\nUP's horizontal snake (UL orientation): {up_nonzero_ul_h[:10]}")
print(f"  First elements: {up_nonzero_ul_h[:4]} (top row: {board_up[0]})")

up_bonus_ul_h, up_details_ul_h, up_chain_ul_h = evaluate_snake_pattern(
    up_snake_ul_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR
)
print(f"\n  Score: {up_bonus_ul_h:.2f}, Chain: {up_chain_ul_h} pairs")
if up_details_ul_h:
    print(f"  First few pairs:")
    for i, d in enumerate(up_details_ul_h[:min(6, len(up_details_ul_h))]):
        if d.get('broken'):
            print(f"    Pair {i}: {d['pair'][0]} < {d['pair'][1]} → BREAK")
        else:
            print(f"    Pair {i}: {d['pair'][0]} >= {d['pair'][1]} → {d['bonus']:.2f}")

print(f"\n  But the BEST orientation for UP is {up_key} with {up_best:.2f}, not UL-H!")

