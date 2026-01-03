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
    
    # Transform board for left-merge computation
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


def evaluate_board_tunable(board: np.ndarray, weights: Dict[str, float] = None, debug: bool = False, print_board: bool = True, path: str = "") -> float:
    """Tunable evaluation function with configurable weights"""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Basic metrics
    empties = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    
    # Strategic positioning
    corner_bonus = 0.0
    if max_tile > 0:
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        if max_tile in corners:
            corner_bonus = weights['corner_bonus']
    
    # Snake pattern bonus (classic 2048 strategy)
    snake_bonus = 0.0
    snake_pattern = [
        board[0, 0], board[0, 1], board[0, 2], board[0, 3],
        board[1, 3], board[1, 2], board[1, 1], board[1, 0],
        board[2, 0], board[2, 1], board[2, 2], board[2, 3],
        board[3, 3], board[3, 2], board[3, 1], board[3, 0]
    ]
    
    # Check if high tiles follow snake pattern
    non_zero = [x for x in snake_pattern if x > 0]
    if len(non_zero) >= 2:
        # Calculate how well tiles follow decreasing pattern
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                snake_bonus += weights['snake_pattern']
    
    # Monotonicity (improved)
    def mono_score(arr: np.ndarray) -> float:
        s = 0.0
        # Row monotonicity
        for r in range(4):
            row = arr[r, :]
            for i in range(3):
                if row[i] >= row[i + 1] and row[i] > 0:
                    s += 1.0
        # Column monotonicity
        for c in range(4):
            col = arr[:, c]
            for i in range(3):
                if col[i] >= col[i + 1] and col[i] > 0:
                    s += 1.0
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
    
    # Merge potential bonus
    merge_potential = 0.0
    for r in range(4):
        for c in range(3):
            if board[r, c] > 0 and board[r, c] == board[r, c + 1]:
                merge_potential += board[r, c] * weights['merge_potential']
    for c in range(4):
        for r in range(3):
            if board[r, c] > 0 and board[r, c] == board[r + 1, c]:
                merge_potential += board[r, c] * weights['merge_potential']
    
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
    
    # Weighted combination
    score = (
        empties * weights['empty_spaces'] +     # Empty spaces
        corner_bonus +                          # Corner positioning
        snake_bonus +                           # Snake pattern
        mono_score(board) +                     # Monotonicity
        smooth_score(board) +                   # Smoothness
        merge_potential +                       # Immediate merge opportunities
        max_tile_bonus +                        # Max tile bonus
        edge_bonus -                            # Edge bonus
        corner_stability_penalty                # Corner stability penalty (subtracted)
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
                                  weights: Dict[str, float] = None, debug: bool = False, path: str = "") -> int:
    """Tunable expectimax with configurable evaluation weights"""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Limit depth to reasonable values
    depth = min(depth, 6)  # Cap at depth 6 for practical performance
    
    # Don't print headers here - the move header is printed in batch_test.py
    # This function just does the tree search
    
    mask = get_valid_action_mask_for_board(board)
    valid_actions = [i for i in range(4) if mask[i]]
    directions = ['up', 'down', 'left', 'right']
    
    if not valid_actions:
        return 0

    best_action = valid_actions[0]
    best_val = -1e9
    
    for action_idx, a in enumerate(valid_actions, 1):
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
        
        # Call chance node with full depth (not depth-1)
        val = chance_value_tunable(nb, depth, chance_sample_k, weights, debug=debug, path=action_path)
            
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
    
    # Determine if we should consider both 2 and 4, or just 2
    max_tile = np.max(b)
    consider_four = max_tile > 2048  # Only consider spawning 4 in late game
    tile_options = [(2, 1.0)] if not consider_four else [(2, 0.9), (4, 0.1)]
    
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
        
        # Pick one random tile type (weighted by probability)
        tile_probs = [p for _, p in tile_options]
        tile_vals = [tile for tile, _ in tile_options]
        tile = np.random.choice(tile_vals, p=tile_probs)
        
        nb = b.copy()
        nb[r, c] = tile
        
        if debug:
            chance_path = f"{path}.C" if path else "C"
            print(f"{indent}[L{d}] {chance_path}: Chance node (depth=0) - Spawning {tile} at position ({r},{c})")
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
    
    # Pick one random empty position
    empty_idx = np.random.choice(len(empties))
    r, c = empties[empty_idx]
    
    # Pick one random tile type (weighted by probability)
    tile_probs = [p for _, p in tile_options]
    tile_vals = [tile for tile, _ in tile_options]
    tile = np.random.choice(tile_vals, p=tile_probs)
    
    nb = b.copy()
    nb[r, c] = tile
    
    if debug:
        chance_path = f"{path}.C" if path else "C"
        print(f"{indent}[L{d}] {chance_path}: Chance node (depth={d}) - Spawning {tile} at position ({r},{c})")
        print(f"{indent}Current board:")
        print(_format_board(b))
        print(f"{indent}Board after spawn:")
        print(_format_board(nb))
        print()
    
    # After tile spawn, it's the player's turn again
    result = max_value_tunable(nb, d, chance_sample_k, weights, debug=debug, path=chance_path)
    if debug:
        print(f"{indent}[L{d}] {chance_path}: Chance node result: {result:.2f}")
    return result


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
