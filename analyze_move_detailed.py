#!/usr/bin/env python3
"""
Detailed analysis of move evaluation components
"""

import numpy as np
from src.game import Game2048
from src.expectimax import preview_after_move, _format_board

# Current board state from move 517
board_state = np.array([
    [   0,    0,    0, 1024],
    [   0,    0,    4,   64],
    [   0,    0,    0,   64],
    [   0,    2,    8,    2]
])

# Weights from 'snake_enabled' config
weights = {
    'empty_spaces': 2.75,
    'corner_bonus': 8.0,
    'corner_stability': 2.75,
    'snake_pattern': 1.0,
    'monotonicity': 0.0,
    'smoothness': 0.1,
    'merge_potential': 0.0,
    'max_tile_bonus': 0.0,
    'edge_bonus': 0.0
}

# Snake pattern scale factor (from expectimax.py - updated to 0.3)
SNAKE_SCALE_FACTOR = 0.3

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
    
    # Snake pattern bonus (nonlinear - scales with tile value)
    # Evaluate all 4 corner orientations (UL, UR, LL, LR) × 2 directions (H, V) = 8 total
    def evaluate_snake_pattern(snake_sequence, weight, scale, orientation_name):
        """Evaluate a snake pattern sequence and return (bonus, total_value, details)"""
        non_zero = [x for x in snake_sequence if x > 0]
        if len(non_zero) < 2:
            return 0.0, 0.0, []
        
        bonus = 0.0
        total_value = sum(non_zero)  # Total value of tiles in pattern
        details = []
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                val1 = non_zero[i]
                log_val = np.log2(val1) if val1 > 0 else 0
                pair_bonus = weight * (1 + scale * log_val)
                bonus += pair_bonus
                details.append(f"{orientation_name} Pair {i+1}: {val1:4d} >= {non_zero[i+1]:4d} → {pair_bonus:.2f} (log2={log_val:.1f})")
            else:
                details.append(f"{orientation_name} Pair {i+1}: {non_zero[i]:4d} <  {non_zero[i+1]:4d} → 0.00")
        return bonus, total_value, details
    
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
    
    # Evaluate all 4 corner orientations
    all_bonuses = {}
    all_details = {}
    
    # 1. Upper-left (original board)
    snake_h, snake_v = get_snake_sequences(board)
    bonus_h, value_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UL-H")
    bonus_v, value_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UL-V")
    all_bonuses['UL-H'] = (bonus_h, value_h)
    all_bonuses['UL-V'] = (bonus_v, value_v)
    all_details['UL-H'] = details_h
    all_details['UL-V'] = details_v
    
    # 2. Upper-right (flip horizontal)
    board_flip_h = np.flip(board, axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_h)
    bonus_h, value_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UR-H")
    bonus_v, value_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UR-V")
    all_bonuses['UR-H'] = (bonus_h, value_h)
    all_bonuses['UR-V'] = (bonus_v, value_v)
    all_details['UR-H'] = details_h
    all_details['UR-V'] = details_v
    
    # 3. Lower-left (flip vertical)
    board_flip_v = np.flip(board, axis=0)
    snake_h, snake_v = get_snake_sequences(board_flip_v)
    bonus_h, value_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LL-H")
    bonus_v, value_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LL-V")
    all_bonuses['LL-H'] = (bonus_h, value_h)
    all_bonuses['LL-V'] = (bonus_v, value_v)
    all_details['LL-H'] = details_h
    all_details['LL-V'] = details_v
    
    # 4. Lower-right (flip both)
    board_flip_both = np.flip(np.flip(board, axis=0), axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_both)
    bonus_h, value_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LR-H")
    bonus_v, value_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LR-V")
    all_bonuses['LR-H'] = (bonus_h, value_h)
    all_bonuses['LR-V'] = (bonus_v, value_v)
    all_details['LR-H'] = details_h
    all_details['LR-V'] = details_v
    
    # Take the maximum bonus from all orientations
    snake_bonus = max(all_bonuses.values(), key=lambda x: x[0])[0] if all_bonuses else 0.0
    best_orientation = max(all_bonuses.items(), key=lambda x: x[1][0])[0] if all_bonuses else None
    
    # Collect details for the best orientation
    snake_details = []
    if best_orientation:
        best_bonus, best_value = all_bonuses[best_orientation]
        snake_details.extend(all_details[best_orientation])
        # Show other orientations that scored well
        other_bonuses = [(k, v[0], v[1]) for k, v in all_bonuses.items() if k != best_orientation and v[0] > 0]
        if other_bonuses:
            other_str = ", ".join([f"{k}: {bonus:.2f} (val={val:.0f})" for k, bonus, val in sorted(other_bonuses, key=lambda x: x[1], reverse=True)])
            snake_details.append(f"(Other orientations: {other_str})")
    
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
    if snake_details:
        print(f"    Best orientation: {best_orientation}")
        print(f"    Details:")
        for detail in snake_details:
            print(f"      {detail}")
    
    # Show what the right column looks like (the valuable pattern)
    right_column = [board[r, 3] for r in range(4)]
    right_col_nonzero = [x for x in right_column if x > 0]
    print(f"\n  Right column (valuable pattern): {right_column}")
    if len(right_col_nonzero) >= 2:
        # Check if it's decreasing (for snake pattern) or increasing (natural order)
        is_decreasing = all(right_col_nonzero[i] >= right_col_nonzero[i+1] for i in range(len(right_col_nonzero) - 1))
        is_increasing = all(right_col_nonzero[i] <= right_col_nonzero[i+1] for i in range(len(right_col_nonzero) - 1))
        if is_decreasing:
            print(f"    ✓ Right column is decreasing (snake pattern): {right_col_nonzero}")
        elif is_increasing:
            print(f"    ⚠ Right column is increasing: {right_col_nonzero} (would be decreasing when read bottom-to-top in LR-V orientation)")
        else:
            print(f"    ✗ Right column pattern is NOT monotonic: {right_col_nonzero}")
    else:
        print(f"    ✗ Right column pattern broken or incomplete: {right_col_nonzero}")
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

# Simulate all moves
game_left = Game2048()
game_left.board = board_state.copy()
game_left.move('left')
board_left = game_left.board.copy()

game_right = Game2048()
game_right.board = board_state.copy()
game_right.move('right')
board_right = game_right.board.copy()

game_up = Game2048()
game_up.board = board_state.copy()
game_up.move('up')
board_up = game_up.board.copy()

game_down = Game2048()
game_down.board = board_state.copy()
game_down.move('down')
board_down = game_down.board.copy()

print("\n" + "="*70)
print("WHY DID AI CHOOSE LEFT? (Human prefers UP to merge 64s in right column)")
print("="*70)

left_stats = analyze_board(board_left, weights, "LEFT MOVE (AI CHOSE)")
right_stats = analyze_board(board_right, weights, "RIGHT MOVE")
up_stats = analyze_board(board_up, weights, "UP MOVE (HUMAN PREFERENCE)")
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

print(f"\n{'Component':<30} {'LEFT':>12} {'RIGHT':>12} {'UP':>12} {'DOWN':>12}")
print(f"{'─'*30} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

for label, key, unit in comparison:
    left_val = left_stats.get(key, 0.0)
    right_val = right_stats.get(key, 0.0)
    up_val = up_stats.get(key, 0.0)
    down_val = down_stats.get(key, 0.0)
    print(f"{label:<30} {left_val:>12.2f} {right_val:>12.2f} {up_val:>12.2f} {down_val:>12.2f}")

print(f"\n{'='*70}")
print("CHOSEN DIRECTION: LEFT")
print(f"{'='*70}")
print(f"\nAll weight component values for LEFT direction:\n")
print(f"  Empty spaces ({left_stats['empties']} × {weights['empty_spaces']:.2f}):     {left_stats['empty_score']:8.2f}")
print(f"  Corner bonus:                              {left_stats['corner_bonus']:8.2f}")
print(f"  Corner stability penalty:                  {-left_stats['corner_stability_penalty']:8.2f}")
print(f"  Snake pattern:                             {left_stats['snake_bonus']:8.2f}")
print(f"  Monotonicity:                              {left_stats['monotonicity_score']:8.2f}")
print(f"  Smoothness:                                {left_stats['smoothness_score']:8.2f}")
print(f"  Merge potential:                           {left_stats['merge_potential']:8.2f}")
print(f"  Merge execution bonus:                     {left_stats['merge_execution_bonus']:8.2f}")
print(f"  Max tile bonus:                            {left_stats['max_tile_bonus']:8.2f}")
print(f"  Edge bonus:                                {left_stats['edge_bonus']:8.2f}")
print(f"  {'─'*60}")
print(f"  TOTAL SCORE:                               {left_stats['total_score']:8.2f}")

print(f"\n{'='*70}")
print("COMPARISON: LEFT vs DOWN vs UP vs RIGHT")
print(f"{'='*70}")
print(f"\nLEFT advantage over DOWN:  {left_stats['total_score'] - down_stats['total_score']:+.2f}")
print(f"LEFT advantage over UP:    {left_stats['total_score'] - up_stats['total_score']:+.2f}")
print(f"LEFT advantage over RIGHT:  {left_stats['total_score'] - right_stats['total_score']:+.2f}")
print(f"DOWN advantage over UP:     {down_stats['total_score'] - up_stats['total_score']:+.2f}")
print(f"\nHuman preference: UP (to merge the two 64s in right column, creating 128)")
print(f"AI chose: LEFT (score: {left_stats['total_score']:.2f})")
print(f"UP score:   {up_stats['total_score']:.2f} (difference: {up_stats['total_score'] - left_stats['total_score']:+.2f})")
print(f"DOWN score: {down_stats['total_score']:.2f} (difference: {down_stats['total_score'] - left_stats['total_score']:+.2f})")
print(f"RIGHT score: {right_stats['total_score']:.2f} (difference: {right_stats['total_score'] - left_stats['total_score']:+.2f})")

print(f"\n{'='*70}")
print("KEY INSIGHTS")
print(f"{'='*70}")
print(f"\n1. IMMEDIATE EVALUATION (shown above):")
print(f"   - Current board has two 64s in right column (rows 2 and 3)")
print(f"   - UP would merge them to create 128, improving the right column pattern")
print(f"   - LEFT merges the 2s in bottom row, creating a 4")
print(f"\n2. EXPECTIMAX LOOKAHEAD (depth=2):")
print(f"   - AI evaluates: move → spawn → move → evaluate")
print(f"   - LEFT might score better after considering future tile spawns")
print(f"   - But UP creates immediate value (128) and improves pattern")
print(f"\n3. SNAKE PATTERN ANALYSIS:")
print(f"   - With chain-breaking logic, snake pattern stops at first violation")
print(f"   - Need to check which move creates better snake patterns")
print(f"   - UP merges 64s → creates 128 in right column")
print(f"   - LEFT merges 2s → creates 4, may break patterns")
print(f"\n4. KEY QUESTION:")
print(f"   - Why does LEFT score higher than UP?")
print(f"   - Check empty spaces, corner bonus, snake pattern, merge execution bonus")
print(f"   - The difference might be in how expectimax looks ahead")

