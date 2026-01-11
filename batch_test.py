#!/usr/bin/env python3
"""
Batch testing script for weight optimization
Runs multiple games in parallel and provides statistical analysis
"""

import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from src.game import Game2048
from src.expectimax import expectimax_best_action_tunable, _format_board

def run_single_game(weights, depth=2, chance_samples=8, max_moves=2000, debug=False):
    """Run a single game and return results"""
    game = Game2048()
    move_count = 0
    
    if debug:
        print(f"\n{'='*70}")
        print(f"STARTING NEW GAME (move {move_count})")
        print(f"{'='*70}\n")
    
    while not game.is_game_over() and move_count < max_moves:
        if debug:
            print(f"\n{'='*70}")
            print(f"MOVE {move_count + 1}")
            print(f"{'='*70}")
            print(_format_board(game.board))
            print(f"Score: {game.score}")
        
        best_action = expectimax_best_action_tunable(
            game.board, 
            depth=depth, 
            chance_sample_k=chance_samples,
            weights=weights,
            debug=False  # Suppress internal debug output
        )
        
        directions = ['up', 'down', 'left', 'right']
        direction = directions[best_action]
        
        if debug:
            print(f"\nSelected move: {direction.upper()}")
        
        success = game.move(direction)
        if not success:
            if debug:
                print("❌ Move failed! (Game over)")
            break
        
        move_count += 1
    
    if debug:
        print(f"\n{'='*70}")
        print(f"GAME OVER")
        print(f"Final score: {game.score}")
        print(f"Final max tile: {game.board.max()}")
        print(f"Total moves: {move_count}")
        print(f"{'='*70}\n")
    
    return {
        'score': game.score,
        'moves': move_count,
        'max_tile': game.board.max(),
        'reached_2048': game.board.max() >= 2048,
        'reached_1024': game.board.max() >= 1024,
        'reached_512': game.board.max() >= 512
    }

def batch_test_weights(weights, num_games=50, depth=2, chance_samples=8, max_moves=2000, debug=False):
    """Run multiple games with given weights and return statistics"""
    if debug:
        print(f"🧪 DEBUG MODE: Running {num_games} game(s) with verbose output")
        print(f"⚠️  Warning: Debug mode is very verbose!\n")
    else:
        print(f"🧪 Testing weights with {num_games} games...")
    
    # Use all but n cores for thermal protection
    total_cores = mp.cpu_count()
    num_processes = total_cores - 2 # 20 cores total, use only 16 for thermal safety
    
    if debug:
        # In debug mode, run single-threaded for cleaner output
        num_processes = 1
        print(f"🖥️  Debug mode: Using 1 process (single-threaded for clean output)")
    else:
        print(f"🖥️  Total cores: {total_cores}, Using {num_processes} CPU cores")
    
    start_time = time.time()
    
    results = []
    if debug:
        # Single-threaded execution for debug mode
        for i in range(num_games):
            result = run_single_game(weights, depth, chance_samples, max_moves, debug=debug)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all games
            futures = [
                executor.submit(run_single_game, weights, depth, chance_samples, max_moves, debug=False)
                for _ in range(num_games)
            ]
            
            print(f"📊 Submitted {len(futures)} games to {num_processes} worker processes")
            print(f"💡 Note: Only {min(num_games, num_processes)} workers will be active (limited by number of tasks)")
            print(f"\n⏳ Running games... (progress will be shown as games complete)\n")
            
            # Collect results with progress updates
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Progress update every game, or every 5 games for large batches
                update_interval = 1 if num_games <= 20 else 5
                if completed % update_interval == 0 or completed == num_games:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed > 0 else 0
                    remaining = num_games - completed
                    eta_seconds = avg_time * remaining
                    eta_minutes = eta_seconds / 60
                    
                    # Calculate running stats
                    scores = [r['score'] for r in results]
                    max_tiles = [r['max_tile'] for r in results]
                    reached_2048 = sum(1 for r in results if r['reached_2048'])
                    
                    print(f"✓ {completed}/{num_games} games completed "
                          f"(Avg: {np.mean(scores):.0f}, 2048: {reached_2048}/{completed}, "
                          f"ETA: {eta_minutes:.1f}m)")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    scores = [r['score'] for r in results]
    max_tiles = [r['max_tile'] for r in results]
    moves = [r['moves'] for r in results]
    
    stats = {
        'num_games': num_games,
        'total_time': total_time,
        'avg_time_per_game': total_time / num_games,
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'score_median': np.median(scores),
        'max_tile_mean': np.mean(max_tiles),
        'max_tile_std': np.std(max_tiles),
        'moves_mean': np.mean(moves),
        'reached_2048_rate': sum(r['reached_2048'] for r in results) / num_games,
        'reached_1024_rate': sum(r['reached_1024'] for r in results) / num_games,
        'reached_512_rate': sum(r['reached_512'] for r in results) / num_games,
        'best_score': max(scores),
        'best_max_tile': max(max_tiles)
    }
    
    return stats

def print_stats(stats, weights_name="Test"):
    """Print formatted statistics"""
    print(f"\n📊 {weights_name} Results ({stats['num_games']} games):")
    print("=" * 60)
    print(f"⏱️  Total Time: {stats['total_time']:.1f}s ({stats['avg_time_per_game']:.2f}s/game)")
    print(f"📈 Score: {stats['score_mean']:.0f} ± {stats['score_std']:.0f} (median: {stats['score_median']:.0f})")
    print(f"🏆 Max Tile: {stats['max_tile_mean']:.0f} ± {stats['max_tile_std']:.0f}")
    print(f"🎯 Moves: {stats['moves_mean']:.0f}")
    print(f"🎉 Reached 2048: {stats['reached_2048_rate']:.1%}")
    print(f"🎉 Reached 1024: {stats['reached_1024_rate']:.1%}")
    print(f"🎉 Reached 512: {stats['reached_512_rate']:.1%}")
    print(f"🏅 Best Score: {stats['best_score']}")
    print(f"🏅 Best Max Tile: {stats['best_max_tile']}")

def main():
    """Test different weight configurations"""
    
    # Test configurations
    test_configs = [
        {
            'name': 'Default',
            'weights': {
                'empty_spaces': 2.0,
                'corner_bonus': 8.0,
                'corner_stability': 1.0,
                'snake_pattern': 2.0,
                'monotonicity': 0.5,
                'smoothness': 0.1,
                'merge_potential': 0.1,
                'max_tile_bonus': 0.0,
                'edge_bonus': 0.0
            }
        },
        {
            'name': 'High Corner Stability',
            'weights': {
                'empty_spaces': 2.0,
                'corner_bonus': 8.0,
                'corner_stability': 3.0,  # Higher penalty
                'snake_pattern': 2.0,
                'monotonicity': 0.5,
                'smoothness': 0.1,
                'merge_potential': 0.1,
                'max_tile_bonus': 0.0,
                'edge_bonus': 0.0
            }
        },
        {
            'name': 'Aggressive Merging',
            'weights': {
                'empty_spaces': 1.5,
                'corner_bonus': 6.0,
                'corner_stability': 0.5,
                'snake_pattern': 1.0,
                'monotonicity': 0.3,
                'smoothness': 0.05,
                'merge_potential': 0.3,  # Higher merge bonus
                'max_tile_bonus': 0.01,
                'edge_bonus': 0.0
            }
        }
    ]
    
    # Run tests
    results = []
    for config in test_configs:
        stats = batch_test_weights(config['weights'], num_games=20, depth=2)
        print_stats(stats, config['name'])
        results.append((config['name'], stats))
    
    # Summary comparison
    print(f"\n🏆 SUMMARY COMPARISON:")
    print("=" * 60)
    print(f"{'Config':<20} {'2048 Rate':<10} {'Avg Score':<12} {'Best Score':<12} {'Best Tile':<10}")
    print("-" * 60)
    for name, stats in results:
        print(f"{name:<20} {stats['reached_2048_rate']:<10.1%} {stats['score_mean']:<12.0f} {stats['best_score']:<12} {stats['best_max_tile']:<10}")

if __name__ == "__main__":
    main()





