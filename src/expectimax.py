import numpy as np
import math
from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time

from src.game import Game2048

# ===== CONFIGURABLE WEIGHTS =====
# Tune these values to optimize performance!

DEFAULT_WEIGHTS = {
    'empty_spaces': 2.0,      # How much to value empty spaces (higher = more conservative)
    'corner_bonus': 8.0,      # Bonus for having max tile in corner
    'corner_stability': 1.0,  # Penalty for max tile not in corner (discourages moving out of corners)
    'snake_pattern': 2.0,     # Bonus for snake pattern (decreasing values)
    'monotonicity': 0.5,      # Bonus for monotonic rows/columns
    'smoothness': 0.1,        # Penalty for adjacent tile differences
    'merge_potential': 0.1,   # Bonus for immediate merge opportunities
    'max_tile_bonus': 0.0,    # Bonus for having high tiles (0 = disabled)
    'edge_bonus': 0.0,        # Bonus for tiles on edges (0 = disabled)
    'risk_aversion': 0.0,     # Risk penalty weight (higher = more risk-averse, 0 = disabled)
}

def get_valid_action_mask_for_board(board: np.ndarray) -> np.ndarray:
    """Convert board to valid action mask [up, down, left, right]"""
    g = Game2048()
    g.board = board.copy()
    valid_moves = g.get_valid_moves()
    mask = np.zeros(4, dtype=bool)
    directions = ['up', 'down', 'left', 'right']
    for i, direction in enumerate(directions):
        if direction in valid_moves:
            mask[i] = True
    return mask


def preview_after_move(board: np.ndarray, action_idx: int) -> np.ndarray:
    """Preview what the board would look like after a move (without spawning a tile)"""
    directions = ['up', 'down', 'left', 'right']
    direction = directions[action_idx]
    
    # Manual move logic without tile spawn (copy from Game2048.move but skip _add_random_tile)
    def _transform(arr: np.ndarray, direction: str):
        """Transform board for left-merge computation"""
        rotated = False
        out = arr
        if direction in ['up', 'down']:
            out = out.T
            rotated = True
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        return out, rotated
    
    def _inverse_transform(arr: np.ndarray, direction: str, rotated: bool):
        out = arr
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        if rotated:
            out = out.T
        return out
    
    def _slide_and_merge_row(row: np.ndarray):
        """Slide and merge a single row"""
        non_zero = row[row != 0]
        out = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                out.append(merged_val)
                i += 2
            else:
                out.append(non_zero[i])
                i += 1
        # Pad with zeros
        while len(out) < len(row):
            out.append(0)
        return np.array(out, dtype=int)
    
    # Transforaard for left-merge computation
    vboard, rotated = _transform(board.copy(), direction)
    
    # Process each row
    for r in range(4):
        new_row = _slide_and_merge_row(vboard[r])
        vboard[r] = new_row
    
    # Transform back to original orientation
    result = _inverse_transform(vboard, direction, rotated)
    return result


def _format_board(board: np.ndarray) -> str:
    """Format board for debug output"""
    lines = []
    for row in board:
        lines.append("[" + " ".join(f"{val:4d}" if val > 0 else "   0" for val in row) + "]")
    return "\n".join(lines)


def _move_moves_max_out_of_corner(board: np.ndarray, action_idx: int) -> bool:
    """
    Check if a move would move the max tile out of the corner.
    Returns True if the max tile is currently in a corner but would not be after the move.
    """
    max_tile = np.max(board)
    if max_tile == 0:
        return False
    
    # Check if max tile is currently in a corner
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    max_tile_in_corner = False
    for r, c in corners:
        if board[r, c] == max_tile:
            max_tile_in_corner = True
            break
    
    if not max_tile_in_corner:
        # Max tile is not in corner to begin with, so this check doesn't apply
        return False
    
    # Preview the board after the move
    nb = preview_after_move(board, action_idx)
    
    # Check if max tile (by value) is still in a corner after the move
    new_max_tile = np.max(nb)
    if new_max_tile > max_tile:
        # A merge created a new max tile - check if it's in a corner
        for r, c in corners:
            if nb[r, c] == new_max_tile:
                return False  # New max tile is in corner, so this is fine
        return True  # New max tile is not in corner
    elif new_max_tile == max_tile:
        # Same max tile value - check if any max tile is in a corner
        for r, c in corners:
            if nb[r, c] == max_tile:
                return False  # Max tile is still in a corner
        return True  # Max tile moved out of corner
    else:
        # This shouldn't happen (max tile decreased), but handle it
        return False


def evaluate_board_tunable(board: np.ndarray, weights: Dict[str, float] = None, debug: bool = False, print_board: bool = True, path: str = "") -> float:
    """Tunable evaluation function with configurable weights"""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Check for dead board (no legal moves) - apply VERY high penalty
    valid_mask = get_valid_action_mask_for_board(board)
    if not np.any(valid_mask):
        # Dead board - return a very high negative penalty to avoid this state
        # This should dominate all other evaluation factors
        dead_board_penalty = -100000.0
        if debug:
            path_depth = len(path.split('.')) if path else 0
            indent = "  " * path_depth
            path_str = f"{path}: " if path else ""
            print(f"{indent}    {path_str}EVALUATING BOARD (score={dead_board_penalty:.2f}) [DEAD BOARD - NO LEGAL MOVES]")
            if print_board:
                print(_format_board(board))
            print()
        return dead_board_penalty
    
    # Basic metrics
    empties = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    
    # Strategic positioning - bonus scales with max tile value to reward creating larger max tiles
    corner_bonus = 0.0
    if max_tile > 0:
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        if max_tile in corners:
            # Scale bonus by max tile value (normalized by 4, so 4->1.0, 8->2.0, 16->4.0, etc.)
            # This rewards creating larger max tiles in the corner
            corner_bonus = weights['corner_bonus'] * (max_tile / 4.0)
    
    # Snake pattern bonus - evaluate all 4 corner orientations and take the best
    # We evaluate snake patterns assuming max tile could be in any of 4 corners:
    # 1. Upper-left (original board)
    # 2. Upper-right (flip horizontal)
    # 3. Lower-left (flip vertical)
    # 4. Lower-right (flip both)
    # For each orientation, we evaluate both horizontal and vertical snake patterns
    snake_bonus = 0.0
    scale_factor = 0.3  # Tune this parameter
    
    def evaluate_snake_pattern(snake_sequence, weight, scale):
        """Evaluate a snake pattern sequence and return bonus
        Stops accumulating as soon as the chain is broken (first non-decreasing pair)"""
        non_zero = [x for x in snake_sequence if x > 0]
        if len(non_zero) < 2:
            return 0.0
        
        bonus = 0.0
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                val1 = non_zero[i]
                log_val = np.log2(val1) if val1 > 0 else 0
                # Scale reward by tile value: larger tiles get exponentially more reward
                bonus += weight * (1 + scale * log_val)
            else:
                # Chain is broken - stop accumulating rewards
                break
        return bonus
    
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
    bonuses = []
    
    # 1. Upper-left (original board)
    snake_h, snake_v = get_snake_sequences(board)
    bonuses.append(evaluate_snake_pattern(snake_h, weights['snake_pattern'], scale_factor))
    bonuses.append(evaluate_snake_pattern(snake_v, weights['snake_pattern'], scale_factor))
    
    # 2. Upper-right (flip horizontal)
    board_flip_h = np.flip(board, axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_h)
    bonuses.append(evaluate_snake_pattern(snake_h, weights['snake_pattern'], scale_factor))
    bonuses.append(evaluate_snake_pattern(snake_v, weights['snake_pattern'], scale_factor))
    
    # 3. Lower-left (flip vertical)
    board_flip_v = np.flip(board, axis=0)
    snake_h, snake_v = get_snake_sequences(board_flip_v)
    bonuses.append(evaluate_snake_pattern(snake_h, weights['snake_pattern'], scale_factor))
    bonuses.append(evaluate_snake_pattern(snake_v, weights['snake_pattern'], scale_factor))
    
    # 4. Lower-right (flip both)
    board_flip_both = np.flip(np.flip(board, axis=0), axis=1)
    snake_h, snake_v = get_snake_sequences(board_flip_both)
    bonuses.append(evaluate_snake_pattern(snake_h, weights['snake_pattern'], scale_factor))
    bonuses.append(evaluate_snake_pattern(snake_v, weights['snake_pattern'], scale_factor))
    
    # Take the maximum bonus from all orientations
    snake_bonus = max(bonuses) if bonuses else 0.0
    
    # Monotonicity - rewards longer contiguous sequences (strong monotonicity in one direction)
    def mono_score(arr: np.ndarray) -> float:
        s = 0.0
        # Row monotonicity - reward longer contiguous sequences quadratically
        for r in range(4):
            row = arr[r, :]
            seq_length = 0
            for i in range(3):
                if row[i] >= row[i + 1] and row[i] > 0:
                    seq_length += 1
                    # Quadratic reward: sequence of length n contributes n² (1, 4, 9, 16...)
                    # This strongly favors longer sequences over many short ones
                    s += seq_length * seq_length
                else:
                    seq_length = 0  # Reset on break in monotonicity
        # Column monotonicity - reward longer contiguous sequences quadratically
        for c in range(4):
            col = arr[:, c]
            seq_length = 0
            for i in range(3):
                if col[i] >= col[i + 1] and col[i] > 0:
                    seq_length += 1
                    s += seq_length * seq_length
                else:
                    seq_length = 0  # Reset on break in monotonicity
        return s * weights['monotonicity']
    
    # Smoothness (improved)
    def smooth_score(arr: np.ndarray) -> float:
        smooth = 0.0
        
        # Horizontal smoothness
        for r in range(4):
            for c in range(3):
                if arr[r, c] > 0 and arr[r, c + 1] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r, c + 1]))
                    smooth -= diff * weights['smoothness']
        
        # Vertical smoothness
        for c in range(4):
            for r in range(3):
                if arr[r, c] > 0 and arr[r + 1, c] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r + 1, c]))
                    smooth -= diff * weights['smoothness']
        
        return smooth
    
    # Merge potential bonus (rewards FUTURE merge opportunities)
    merge_potential = 0.0
    for r in range(4):
        for c in range(3):
            if board[r, c] > 0 and board[r, c] == board[r, c + 1]:
                merge_potential += board[r, c] * weights['merge_potential']
    for c in range(4):
        for r in range(3):
            if board[r, c] > 0 and board[r, c] == board[r + 1, c]:
                merge_potential += board[r, c] * weights['merge_potential']
    
    # Merge execution bonus - rewards high-value tiles in strategic positions
    # This compensates for merge_potential's bias toward keeping merges available
    merge_execution_bonus = 0.0
    if max_tile > 0:
        # Find max tile position
        max_pos = np.unravel_index(np.argmax(board), board.shape)
        max_r, max_c = max_pos
        
        # Reward high-value tiles adjacent to the max tile (likely from merges)
        # This rewards merges that create tiles near the max tile
        adjacent_positions = [
            (max_r - 1, max_c), (max_r + 1, max_c),
            (max_r, max_c - 1), (max_r, max_c + 1)
        ]
        for r, c in adjacent_positions:
            if 0 <= r < 4 and 0 <= c < 4 and board[r, c] > 0:
                # Reward tiles that are large (likely from merges)
                # Scale by tile value, with extra bonus if in corner
                tile_value = board[r, c]
                bonus = tile_value * 0.1  # Base bonus for high-value adjacent tile
                if (r, c) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
                    bonus *= 1.5  # Extra bonus if the merged tile is in a corner
                merge_execution_bonus += bonus
    
    # Max tile bonus (optional)
    max_tile_bonus = 0.0
    if weights['max_tile_bonus'] > 0:
        max_tile_bonus = max_tile * weights['max_tile_bonus']
    
    # Edge bonus (optional)
    edge_bonus = 0.0
    if weights['edge_bonus'] > 0:
        edge_tiles = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] > 0 and (r == 0 or r == 3 or c == 0 or c == 3):
                    edge_tiles += board[r, c]
        edge_bonus = edge_tiles * weights['edge_bonus']
    
    # Corner stability penalty - only for max tile
    corner_stability_penalty = 0.0
    if weights['corner_stability'] > 0 and max_tile > 0:
        # Check if max_tile is in a corner
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        if max_tile not in corners:
            # Penalize if max_tile is not in a corner
            corner_stability_penalty = max_tile * weights['corner_stability']
    
    # Risk assessment - penalize dangerous board states (risk-averse evaluation)
    # This makes the AI prefer safer moves over risky high-reward moves
    risk_penalty = 0.0
    if 'risk_aversion' in weights and weights['risk_aversion'] > 0:
        # Penalize boards with few empty spaces (high risk of getting stuck)
        if empties <= 2:
            risk_penalty += weights['risk_aversion'] * (3 - empties) * 10.0  # Heavy penalty for 0-2 empties
        elif empties <= 4:
            risk_penalty += weights['risk_aversion'] * (5 - empties) * 2.0   # Moderate penalty for 3-4 empties
        
        # Penalize if max tile is not in corner (risky positioning)
        if max_tile > 0:
            corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
            if max_tile not in corners:
                risk_penalty += weights['risk_aversion'] * max_tile * 0.5  # Additional risk for max tile not in corner
        
        # Penalize boards with few legal moves (high risk of getting stuck)
        num_legal_moves = np.sum(valid_mask)
        if num_legal_moves <= 2:
            risk_penalty += weights['risk_aversion'] * (3 - num_legal_moves) * 5.0  # Penalty for limited moves
    
    # Weighted combination
    score = (
        empties * weights['empty_spaces'] +     # Empty spaces
        corner_bonus +                          # Corner positioning
        snake_bonus +                           # Snake pattern
        mono_score(board) +                     # Monotonicity
        smooth_score(board) +                   # Smoothness
        merge_potential +                       # Immediate merge opportunities (future)
        merge_execution_bonus +                 # Merge execution bonus (reward completed merges)
        max_tile_bonus +                        # Max tile bonus
        edge_bonus -                            # Edge bonus
        corner_stability_penalty -               # Corner stability penalty (subtracted)
        risk_penalty                            # Risk penalty (subtracted) - makes AI more risk-averse
    )
    
    if debug:
        path_depth = len(path.split('.')) if path else 0
        indent = "  " * path_depth
        path_str = f"{path}: " if path else ""
        print(f"{indent}    {path_str}EVALUATING BOARD (score={score:.2f})")
        if print_board:
            print(_format_board(board))
        print()
    
    return score


def expectimax_best_action_tunable(board: np.ndarray, depth: int = 4, chance_sample_k: int = 8, 
                                  weights: Dict[str, float] = None, debug: bool = False, path: str = "",
                                  adaptive_depth: bool = False) -> int:
    """
    Tunable expectimax with configurable evaluation weights.
    
    Args:
        board: 4x4 game board
        depth: Base search depth (adjusted adaptively if adaptive_depth=True)
        chance_sample_k: Number of chance node samples
        weights: Evaluation function weights
        debug: Enable debug output
        path: Debug path string
        adaptive_depth: If True, increase depth for endgame (max_tile > 1024) and sparse boards
    
    Returns:
        Best action (0=up, 1=down, 2=left, 3=right)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Adaptive depth: increase depth only in very late game when it's critical
    if adaptive_depth:
        max_tile = np.max(board)
        empty_spaces = np.count_nonzero(board == 0)
        
        # Stay at baseline depth (2) unless we've reached 4096 AND board is sparse
        # This prevents the exponential slowdown during normal gameplay
        if max_tile >= 4096 and empty_spaces < 4:
            # Only increase depth when trying to merge 4096s (very late endgame)
            effective_depth = min(depth + 1, 4)  # Add 1 depth, cap at 4
        else:
            # Normal gameplay: use baseline depth (2)
            effective_depth = depth
    else:
        effective_depth = depth
    
    # Limit depth to reasonable values
    effective_depth = min(effective_depth, 6)  # Cap at depth 6 for practical performance
    
    # Don't print headers here - the move header is printed in batch_test.py
    # This function just does the tree search
    
    mask = get_valid_action_mask_for_board(board)
    valid_actions = [i for i in range(4) if mask[i]]
    directions = ['up', 'down', 'left', 'right']
    
    if not valid_actions:
        return 0

    # Filter out moves that would move max tile out of corner
    # UNLESS it's the only legal move
    safe_actions = []
    for a in valid_actions:
        nb = preview_after_move(board, a)
        if np.array_equal(nb, board):
            continue  # Skip moves with no effect
        if not _move_moves_max_out_of_corner(board, a):
            safe_actions.append(a)
    
    # If we have safe actions, use only those; otherwise use all valid actions
    actions_to_consider = safe_actions if safe_actions else valid_actions
    
    if debug and not path and safe_actions != valid_actions:
        excluded = set(valid_actions) - set(safe_actions)
        excluded_dirs = [directions[a].upper() for a in excluded]
        print(f"  ⚠️  Excluding moves that move max tile out of corner: {', '.join(excluded_dirs)}")
        print(f"  ✓ Considering moves: {', '.join([directions[a].upper() for a in actions_to_consider])}")
        print()
    
    best_action = actions_to_consider[0]
    best_val = -1e9
    
    # Use effective_depth instead of depth
    for action_idx, a in enumerate(actions_to_consider, 1):
        direction = directions[a].upper()
        action_path = str(action_idx) if not path else f"{path}.{action_idx}"
        
        if debug:
            if not path:  # Root level - clear action header
                print(f"\n  Evaluating Action: {direction}")
                print(f"  {'─' * 66}")
            else:
                indent = "  " * len(action_path.split('.'))
                print(f"{indent}[{action_path}] Evaluating action {direction}")
        
        nb = preview_after_move(board, a)
        if np.array_equal(nb, board):
            if debug:
                if not path:
                    print(f"  ⚠ Action {direction} had no effect, skipping\n")
                else:
                    indent = "  " * len(action_path.split('.'))
                    print(f"{indent}  (Action {direction} had no effect, skipping)")
            continue
        
        if debug and not path:  # Only show board at root level to reduce clutter
            print(f"  Board after {direction} move:")
            # Indent the board slightly
            board_str = _format_board(nb)
            for line in board_str.split('\n'):
                if line.strip():
                    print(f"    {line}")
                else:
                    print()
            print()
        
        # Call chance node with effective depth
        val = chance_value_tunable(nb, effective_depth, chance_sample_k, weights, debug=debug, path=action_path)
            
        if debug:
            if not path:  # Root level - clear value display
                print(f"  └─ Expected value for {direction}: {val:.2f}\n")
            else:
                indent = "  " * len(action_path.split('.'))
                print(f"{indent}[{action_path}] Action {direction} value: {val:.2f}")
                print()
            
        if val > best_val:
            best_val = val
            best_action = a
    
    if debug and not path:  # Only print move conclusion at root level
        print("\n" + "─" * 70)
        print(f"  ✓ SELECTED MOVE: {directions[best_action].upper()} (expected value: {best_val:.2f})")
        print("─" * 70)
    
    return best_action


def chance_value_tunable(b: np.ndarray, d: int, chance_sample_k: int, weights: Dict[str, float], debug: bool = False, path: str = "") -> float:
    """Tunable chance value calculation - represents tile spawns"""
    path_depth = len(path.split('.')) if path else 0
    indent = "  " * path_depth if debug else ""
    
    # Always spawn tile value 2 (simplified - not considering 4s for now)
    tile_value = 2
    
    # If depth is 0, do one spawn step then evaluate
    if d == 0:
        empties = np.argwhere(b == 0)
        if empties.size == 0:
            if debug:
                spawn_path = f"{path}.0" if path else "0"
                print(f"{indent}[L{d}] {spawn_path}: Chance node (depth=0) - No empty spaces, evaluating board")
            return evaluate_board_tunable(b, weights, debug=debug, print_board=False, path=path)
        
        # Pick one random empty position
        empty_idx = np.random.choice(len(empties))
        r, c = empties[empty_idx]
        
        nb = b.copy()
        nb[r, c] = tile_value
        
        # Define chance_path for path tracking (used even when debug=False)
        chance_path = f"{path}.C" if path else "C"
        
        if debug:
            print(f"{indent}[L{d}] {chance_path}: Chance node (depth=0) - Spawning {tile_value} at position ({r},{c})")
            print(_format_board(nb))
            print()
        
        result = evaluate_board_tunable(nb, weights, debug=debug, print_board=False, path=chance_path)
        if debug:
            print(f"{indent}[L{d}] {chance_path}: Chance node result: {result:.2f}")
        return result
    
    empties = np.argwhere(b == 0)
    if empties.size == 0:
        # No empty spaces for tile spawn, so evaluate current state
        if debug:
            chance_path = f"{path}.C" if path else "C"
            print(f"{indent}[L{d}] {chance_path}: Chance node - No empty spaces, evaluating board")
        return evaluate_board_tunable(b, weights, debug=debug, print_board=False, path=path)
    
    # Enumerate ALL possible outcomes instead of sampling
    # Always spawn tile value 2 (simplified)
    total_value = 0.0
    num_empty = len(empties)
    
    # Enumerate all possible outcomes: each empty position (always spawn 2)
    for empty_idx in range(num_empty):
        r, c = empties[empty_idx]
        
        nb = b.copy()
        nb[r, c] = tile_value
        
        # Evaluate this outcome (recursively)
        outcome_value = max_value_tunable(nb, d, chance_sample_k, weights, debug=False, path="")
        total_value += outcome_value
    
    # Return simple average of all outcomes
    expected_value = total_value / num_empty if num_empty > 0 else total_value
    if debug:
        chance_path = f"{path}.C" if path else "C"
        print(f"{indent}[L{d}] {chance_path}: Chance node (depth={d}) - Expected value: {expected_value:.2f} (from {num_empty} outcomes, tile={tile_value})")
    return expected_value


def max_value_tunable(b: np.ndarray, d: int, chance_sample_k: int, weights: Dict[str, float], debug: bool = False, path: str = "") -> float:
    """Tunable max value calculation - represents player moves"""
    path_depth = len(path.split('.')) if path else 0
    indent = "  " * path_depth if debug else ""
    directions = ['up', 'down', 'left', 'right']
    
    # If depth is 0, evaluate the current board state (after player move)
    if d == 0:
        if debug:
            max_path = f"{path}.M" if path else "M"
            print(f"{indent}[L{d}] {max_path}: Max node (depth=0) - Evaluating board")
        return evaluate_board_tunable(b, weights, debug=debug, print_board=True, path=path)
    
    if debug:
        max_path = f"{path}.M" if path else "M"
        print(f"{indent}[L{d}] {max_path}: Max node (depth={d}) - Evaluating moves")
        print(f"{indent}Current board:")
        print(_format_board(b))
        print()
    
    m = get_valid_action_mask_for_board(b)
    if not np.any(m):
        # No valid moves, so evaluate current state
        if debug:
            max_path = f"{path}.M" if path else "M"
            print(f"{indent}[L{d}] {max_path}: Max node - No valid moves, evaluating board")
        return evaluate_board_tunable(b, weights, debug=debug, print_board=False, path=path)
    
    best = -1e9
    action_num = 0
    for a in range(4):
        if not m[a]:
            continue
        
        nb = preview_after_move(b, a)
        if np.array_equal(nb, b):
            # Skip moves that have no effect - don't increment action_num
            continue
        
        # Only increment action_num for moves that actually have an effect
        action_num += 1
        direction = directions[a].upper()
        action_path = f"{path}.{action_num}" if path else str(action_num)
        
        if debug:
            action_indent = "  " * len(action_path.split('.'))
            if not path:  # Top level - show clear action label
                print(f"\n  ┌─ Evaluating Action: {direction}")
            else:
                print(f"{action_indent}[L{d}] {action_path}: Evaluating action {direction}")
        
        if debug:
            if not path:  # Top level - clearer formatting
                print(f"  │  Board after {direction} move:")
                print(_format_board(nb))
                print(f"  │")
            else:
                print(f"{action_indent}  Board after {direction} move:")
                print(_format_board(nb))
                print()
        
        # After player move, we go to chance node (tile spawn)
        # Use effective_depth for recursive calls
        val = chance_value_tunable(nb, d - 1, chance_sample_k, weights, debug=debug, path=action_path)
        if debug:
            print(f"{action_indent}[L{d}] {action_path}: Action {direction} value: {val:.2f}")
            print()
        if val > best:
            best = val
    
    if debug:
        max_path = f"{path}.M" if path else "M"
        print(f"{indent}[L{d}] {max_path}: Max node result: {best:.2f}")
    return best


# Predefined weight configurations for different strategies
WEIGHT_PRESETS = {
    'conservative': {
        'empty_spaces': 3.0,      # Very high - prioritize keeping spaces open
        'corner_bonus': 10.0,     # High - always keep max tile in corner
        'corner_stability': 2.0,  # High - strongly discourage moving out of corners
        'snake_pattern': 3.0,     # High - maintain snake pattern
        'monotonicity': 1.0,      # High - keep rows/cols ordered
        'smoothness': 0.2,        # High - minimize adjacent differences
        'merge_potential': 0.05,  # Low - don't rush merges
        'max_tile_bonus': 0.0,    # None
        'edge_bonus': 0.0,        # None
    },
    'aggressive': {
        'empty_spaces': 1.5,      # Lower - willing to fill spaces
        'corner_bonus': 6.0,      # Medium - corner is good but not critical
        'corner_stability': 0.5,  # Lower - more willing to move tiles
        'snake_pattern': 1.0,     # Lower - flexible pattern
        'monotonicity': 0.3,      # Lower - less strict ordering
        'smoothness': 0.05,       # Lower - tolerate differences
        'merge_potential': 0.2,   # High - actively seek merges
        'max_tile_bonus': 0.01,   # Small bonus for high tiles
        'edge_bonus': 0.0,        # None
    },
    'balanced': {
        'empty_spaces': 2.0,      # Medium - balanced approach
        'corner_bonus': 8.0,      # High - corner positioning
        'corner_stability': 1.0,  # Medium - discourage moving out of corners
        'snake_pattern': 2.0,     # Medium - maintain pattern
        'monotonicity': 0.5,      # Medium - some ordering
        'smoothness': 0.1,        # Medium - moderate smoothness
        'merge_potential': 0.1,   # Medium - opportunistic merges
        'max_tile_bonus': 0.0,    # None
        'edge_bonus': 0.0,        # None
    },
    'experimental': {
        'empty_spaces': 2.5,      # High - keep spaces open
        'corner_bonus': 12.0,     # Very high - corner critical
        'corner_stability': 3.0,  # Very high - never move out of corners
        'snake_pattern': 4.0,     # Very high - strict pattern
        'monotonicity': 1.5,      # Very high - strict ordering
        'smoothness': 0.3,        # Very high - very smooth
        'merge_potential': 0.05,  # Low - patient merging
        'max_tile_bonus': 0.005,  # Tiny bonus
        'edge_bonus': 0.001,      # Tiny edge bonus
    }
}


# Keep the original function for compatibility
def expectimax_best_action(board: np.ndarray, depth: int = 3, chance_sample_k: int = 6) -> int:
    """Original single-threaded version"""
    return expectimax_best_action_tunable(board, depth, chance_sample_k, DEFAULT_WEIGHTS)
