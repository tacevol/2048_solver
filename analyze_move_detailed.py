#!/usr/bin/env python3
"""
Detailed analysis of move evaluation components
"""

import numpy as np
from src.game import Game2048
from src.expectimax import preview_after_move, _format_board

# Current board state from move 487
board_state = np.array([
    [ 512,   32,    8,    4],
    [ 256,  128,    4,    0],
    [ 128,    2,    0,    4],
    [   0,    0,    0,    0]
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

# Snake pattern scale factor (from expectimax.py)
SNAKE_SCALE_FACTOR = 0.15

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
        """Evaluate a snake pattern sequence and return bonus with details"""
        non_zero = [x for x in snake_sequence if x > 0]
        if len(non_zero) < 2:
            return 0.0, []
        
        bonus = 0.0
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
        return bonus, details
    
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
    bonus_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UL-H")
    bonus_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UL-V")
    all_bonuses['UL-H'] = bonus_h
    all_bonuses['UL-V'] = bonus_v
    all_details['UL-H'] = details_h
    all_details['UL-V'] = details_v
    
    # 2. Upper-right (flip horizontal)
    board_flip_h = np.flip(board, axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_h)
    bonus_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UR-H")
    bonus_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "UR-V")
    all_bonuses['UR-H'] = bonus_h
    all_bonuses['UR-V'] = bonus_v
    all_details['UR-H'] = details_h
    all_details['UR-V'] = details_v
    
    # 3. Lower-left (flip vertical)
    board_flip_v = np.flip(board, axis=0)
    snake_h, snake_v = get_snake_sequences(board_flip_v)
    bonus_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LL-H")
    bonus_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LL-V")
    all_bonuses['LL-H'] = bonus_h
    all_bonuses['LL-V'] = bonus_v
    all_details['LL-H'] = details_h
    all_details['LL-V'] = details_v
    
    # 4. Lower-right (flip both)
    board_flip_both = np.flip(np.flip(board, axis=0), axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_both)
    bonus_h, details_h = evaluate_snake_pattern(snake_h, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LR-H")
    bonus_v, details_v = evaluate_snake_pattern(snake_v, weights['snake_pattern'], SNAKE_SCALE_FACTOR, "LR-V")
    all_bonuses['LR-H'] = bonus_h
    all_bonuses['LR-V'] = bonus_v
    all_details['LR-H'] = details_h
    all_details['LR-V'] = details_v
    
    # Take the maximum bonus from all orientations
    snake_bonus = max(all_bonuses.values()) if all_bonuses else 0.0
    best_orientation = max(all_bonuses.items(), key=lambda x: x[1])[0] if all_bonuses else None
    
    # Collect details for the best orientation
    snake_details = []
    if best_orientation:
        snake_details.extend(all_details[best_orientation])
        # Show other orientations that scored well
        other_bonuses = [(k, v) for k, v in all_bonuses.items() if k != best_orientation and v > 0]
        if other_bonuses:
            other_str = ", ".join([f"{k}: {v:.2f}" for k, v in sorted(other_bonuses, key=lambda x: x[1], reverse=True)])
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
        print(f"    Details:")
        for detail in snake_details:
            print(f"      {detail}")
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
print("WHY DID AI CHOOSE RIGHT? (Human prefers LEFT or UP)")
print("="*70)

left_stats = analyze_board(board_left, weights, "LEFT MOVE")
right_stats = analyze_board(board_right, weights, "RIGHT MOVE (AI CHOSE)")
up_stats = analyze_board(board_up, weights, "UP MOVE")
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
print("CHOSEN DIRECTION: RIGHT")
print(f"{'='*70}")
print(f"\nAll weight component values for RIGHT direction:\n")
print(f"  Empty spaces ({right_stats['empties']} × {weights['empty_spaces']:.2f}):     {right_stats['empty_score']:8.2f}")
print(f"  Corner bonus:                              {right_stats['corner_bonus']:8.2f}")
print(f"  Corner stability penalty:                  {-right_stats['corner_stability_penalty']:8.2f}")
print(f"  Snake pattern:                             {right_stats['snake_bonus']:8.2f}")
print(f"  Monotonicity:                              {right_stats['monotonicity_score']:8.2f}")
print(f"  Smoothness:                                {right_stats['smoothness_score']:8.2f}")
print(f"  Merge potential:                           {right_stats['merge_potential']:8.2f}")
print(f"  Merge execution bonus:                     {right_stats['merge_execution_bonus']:8.2f}")
print(f"  Max tile bonus:                            {right_stats['max_tile_bonus']:8.2f}")
print(f"  Edge bonus:                                {right_stats['edge_bonus']:8.2f}")
print(f"  {'─'*60}")
print(f"  TOTAL SCORE:                               {right_stats['total_score']:8.2f}")

print(f"\n{'='*70}")
print("COMPARISON: RIGHT vs LEFT vs UP")
print(f"{'='*70}")
print(f"\nRIGHT advantage over LEFT:  {right_stats['total_score'] - left_stats['total_score']:+.2f}")
print(f"RIGHT advantage over UP:    {right_stats['total_score'] - up_stats['total_score']:+.2f}")
print(f"LEFT advantage over UP:     {left_stats['total_score'] - up_stats['total_score']:+.2f}")
print(f"\nHuman preference: LEFT or UP would be better")
print(f"AI chose: RIGHT (score: {right_stats['total_score']:.2f})")
print(f"LEFT score: {left_stats['total_score']:.2f} (difference: {left_stats['total_score'] - right_stats['total_score']:+.2f})")
print(f"UP score:   {up_stats['total_score']:.2f} (difference: {up_stats['total_score'] - right_stats['total_score']:+.2f})")

print(f"\n{'='*70}")
print("KEY INSIGHTS")
print(f"{'='*70}")
print(f"\n1. IMMEDIATE EVALUATION (shown above):")
print(f"   - LEFT and UP are clearly better than RIGHT")
print(f"   - Main difference: Merge execution bonus")
print(f"     * LEFT/UP: 28.80 (large tiles adjacent to max tile 512)")
print(f"     * RIGHT:    3.20 (small tiles adjacent to max tile 512)")
print(f"\n2. EXPECTIMAX LOOKAHEAD (depth=2):")
print(f"   - AI evaluates: move → spawn → move → evaluate")
print(f"   - RIGHT might score better after considering future tile spawns")
print(f"   - But this seems wrong - LEFT/UP create better merge opportunities")
print(f"\n3. SNAKE PATTERN:")
print(f"   - RIGHT has slightly better snake pattern (12.40 vs 11.10)")
print(f"   - But this doesn't compensate for the merge execution penalty")
print(f"\n4. POSSIBLE ISSUES:")
print(f"   - Merge execution bonus might not be weighted correctly")
print(f"   - Snake pattern might be overvalued relative to merge opportunities")
print(f"   - Expectimax depth=2 might not be seeing the 3-move sequence")
print(f"     that human sees (LEFT → LEFT → UP)")

