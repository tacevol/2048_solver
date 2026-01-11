#!/usr/bin/env python3
"""
Validate a single configuration with more games for reliable statistics
Usage: python validate_single.py <config_name> <num_games>
"""

import sys
from batch_test import batch_test_weights, print_stats

# Predefined configurations
CONFIGS = {
    'CS 2.5': {
        'empty_spaces': 2.0,
        'corner_bonus': 8.0,
        'corner_stability': 2.5,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'CS 3.0': {
        'empty_spaces': 2.0,
        'corner_bonus': 8.0,
        'corner_stability': 3.0,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'CS 4.0': {
        'empty_spaces': 2.0,
        'corner_bonus': 8.0,
        'corner_stability': 4.0,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.3,  # Increased to prioritize merges that keep max tile in corner
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'CS 2.25': {
        'empty_spaces': 2.0,
        'corner_bonus': 8.0,
        'corner_stability': 2.25,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'CS 2.75': {
        'empty_spaces': 2.0,
        'corner_bonus': 8.0,
        'corner_stability': 2.75,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'ES 2.75 (Best)': {
        'empty_spaces': 2.75,  # Optimal from ES validation (250 games) - 29.2% 2048 rate
        'corner_bonus': 8.0,
        'corner_stability': 2.75,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
    'ES 2.25 (Alternative)': {
        'empty_spaces': 2.25,  # Also very good - 28.4% 2048 rate (not statistically different)
        'corner_bonus': 8.0,
        'corner_stability': 2.75,
        'snake_pattern': 2.0,
        'monotonicity': 0.5,
        'smoothness': 0.1,
        'merge_potential': 0.1,
        'max_tile_bonus': 0.0,
        'edge_bonus': 0.0
    },
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_single.py <config_name> [num_games] [--debug]")
        print(f"\nAvailable configs: {', '.join(CONFIGS.keys())}")
        print("Example: python validate_single.py 'CS 2.5' 50")
        print("Example: python validate_single.py 'CS 2.5' 1 --debug")
        sys.exit(1)
    
    config_name = sys.argv[1]
    num_games = 50
    debug = False
    
    # Parse arguments
    if len(sys.argv) > 2:
        if sys.argv[2] == '--debug':
            debug = True
            num_games = 1  # Debug mode defaults to 1 game
        else:
            num_games = int(sys.argv[2])
    
    if len(sys.argv) > 3 and sys.argv[3] == '--debug':
        debug = True
        if num_games > 1:
            print("⚠️  Warning: Debug mode with multiple games will be very verbose!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    if config_name not in CONFIGS:
        print(f"Error: Config '{config_name}' not found")
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)
    
    weights = CONFIGS[config_name]
    
    print(f"📊 VALIDATING: {config_name}")
    print(f"🎮 Games: {num_games}")
    if debug:
        print(f"🐛 Debug mode: ENABLED (verbose output)")
    print("=" * 70)
    
    stats = batch_test_weights(weights, num_games=num_games, depth=2, debug=debug)
    print_stats(stats, config_name)
    
    # Additional statistics
    print(f"\n📈 STATISTICAL SUMMARY:")
    print("=" * 70)
    import math
    p = stats['reached_2048_rate']
    n = stats['num_games']
    if n > 0 and p > 0 and p < 1:
        se = math.sqrt(p * (1 - p) / n)
        ci_lower = max(0, p - 1.96 * se)
        ci_upper = min(1, p + 1.96 * se)
        print(f"2048 Rate: {p:.1%}")
        print(f"95% Confidence Interval: [{ci_lower:.1%}, {ci_upper:.1%}]")
    print(f"Average Score: {stats['score_mean']:.0f} ± {stats['score_std']:.0f}")
    print(f"Median Score: {stats['score_median']:.0f}")
    print(f"Best Score: {stats['best_score']}")
    print(f"Best Max Tile: {stats['best_max_tile']}")

if __name__ == "__main__":
    main()

