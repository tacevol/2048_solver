#!/usr/bin/env python3
"""
Evaluate snake chains using the existing expectimax logic
"""

import numpy as np
from src.expectimax import evaluate_board_tunable

# Snake pattern evaluation function (from expectimax.py)
def evaluate_snake_pattern(snake_sequence, weight, scale):
    """Evaluate a snake pattern sequence and return bonus
    Stops accumulating as soon as the chain is broken (first non-decreasing pair)"""
    non_zero = [x for x in snake_sequence if x > 0]
    if len(non_zero) < 2:
        return 0.0
    
    bonus = 0.0
    details = []
    
    for i in range(len(non_zero) - 1):
        if non_zero[i] >= non_zero[i + 1]:
            val1 = non_zero[i]
            log_val = np.log2(val1) if val1 > 0 else 0
            # Scale reward by tile value: larger tiles get exponentially more reward
            pair_bonus = weight * (1 + scale * log_val)
            bonus += pair_bonus
            details.append({
                'index': i,
                'pair': (non_zero[i], non_zero[i + 1]),
                'log_val': log_val,
                'pair_bonus': pair_bonus,
                'cumulative': bonus,
                'chain_length': i + 1
            })
        else:
            # Chain is broken - stop accumulating rewards
            details.append({
                'index': i,
                'pair': (non_zero[i], non_zero[i + 1]),
                'broken': True,
                'cumulative': bonus,
                'chain_length': i
            })
            break
    
    return bonus, details

# Default weights from expectimax.py
weights = {
    'snake_pattern': 1.0,  # Default weight
}
SNAKE_SCALE_FACTOR = 0.3  # From expectimax.py

# The chains to evaluate
chain_a = [1024, 512, 128, 64, 16, 32, 64, 2]
chain_b = [1024, 512, 128, 64, 0, 16, 32, 64]

print("=" * 80)
print("SNAKE CHAIN EVALUATION")
print("=" * 80)

print(f"\nChain A: {chain_a}")
print(f"Chain B: {chain_b}")

print(f"\nConfiguration:")
print(f"  snake_pattern weight: {weights['snake_pattern']}")
print(f"  scale factor: {SNAKE_SCALE_FACTOR}")
print(f"  Formula: bonus = weight * (1 + scale * log2(tile_value))")

# Evaluate chain A
bonus_a, details_a = evaluate_snake_pattern(chain_a, weights['snake_pattern'], SNAKE_SCALE_FACTOR)
print(f"\n{'='*80}")
print(f"CHAIN A EVALUATION")
print(f"{'='*80}")
print(f"Sequence (non-zero): {[x for x in chain_a if x > 0]}")

print(f"\nDetailed breakdown:")
for detail in details_a:
    if detail.get('broken'):
        print(f"  Pair {detail['index']}: {detail['pair'][0]} < {detail['pair'][1]} → CHAIN BROKEN")
        print(f"    Chain length: {detail['chain_length']} pairs")
        print(f"    Total bonus: {detail['cumulative']:.2f}")
    else:
        log_str = f", log2={detail['log_val']:.2f}" if detail.get('log_val') is not None else ""
        print(f"  Pair {detail['index']}: {detail['pair'][0]:4d} >= {detail['pair'][1]:4d} → {detail['pair_bonus']:.2f}{log_str} (cumulative: {detail['cumulative']:.2f}, chain_len: {detail['chain_length']})")

print(f"\n✅ Chain A Total Score: {bonus_a:.2f}")

# Evaluate chain B
bonus_b, details_b = evaluate_snake_pattern(chain_b, weights['snake_pattern'], SNAKE_SCALE_FACTOR)
print(f"\n{'='*80}")
print(f"CHAIN B EVALUATION")
print(f"{'='*80}")
print(f"Sequence (non-zero, zeros filtered): {[x for x in chain_b if x > 0]}")

print(f"\nDetailed breakdown:")
for detail in details_b:
    if detail.get('broken'):
        print(f"  Pair {detail['index']}: {detail['pair'][0]} < {detail['pair'][1]} → CHAIN BROKEN")
        print(f"    Chain length: {detail['chain_length']} pairs")
        print(f"    Total bonus: {detail['cumulative']:.2f}")
    else:
        log_str = f", log2={detail['log_val']:.2f}" if detail.get('log_val') is not None else ""
        print(f"  Pair {detail['index']}: {detail['pair'][0]:4d} >= {detail['pair'][1]:4d} → {detail['pair_bonus']:.2f}{log_str} (cumulative: {detail['cumulative']:.2f}, chain_len: {detail['chain_length']})")

print(f"\n✅ Chain B Total Score: {bonus_b:.2f}")

# Comparison
print(f"\n{'='*80}")
print(f"COMPARISON")
print(f"{'='*80}")
print(f"Chain A score: {bonus_a:.2f}")
print(f"Chain B score: {bonus_b:.2f}")
print(f"Difference (A - B): {bonus_a - bonus_b:.2f}")

# Show where they differ
chain_a_nonzero = [x for x in chain_a if x > 0]
chain_b_nonzero = [x for x in chain_b if x > 0]
print(f"\nChain A (non-zero): {chain_a_nonzero}")
print(f"Chain B (non-zero): {chain_b_nonzero}")

print(f"\nKey differences:")
print(f"  - Chain A has a '0' at position 4, which is filtered out")
print(f"  - Chain B has no '0' in the sequence, so the pattern continues differently")
print(f"  - The algorithm stops at the first break in the pattern")

