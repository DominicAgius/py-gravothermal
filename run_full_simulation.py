#!/usr/bin/env python3
"""
All-in-one script: Generate initial conditions and run simulation.

This script combines initialization and evolution into a single workflow.
You can run both steps together or skip initialization if you already have
initial conditions.
"""

import sys
import time
import argparse
from pathlib import Path


def run_initialization(tag, n_layers, output_dir):
    """
    Generate initial conditions
    
    Args:
        tag: Name tag for this run
        n_layers: Number of radial zones
        output_dir: Directory to save initial conditions
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("STEP 1: GENERATING INITIAL CONDITIONS")
    print("="*70)
    
    try:
        from initialization import run_initialization as init_func
        
        print(f"\nGenerating {n_layers}-layer initial conditions...")
        print(f"Tag: {tag}")
        print(f"Output: {output_dir}")
        
        init_func(tag=tag, n_layers=n_layers, output_dir=str(output_dir))
        
        print("\n✓ Initial conditions generated successfully")
        return True
        
    except ImportError:
        print("\nError: initialization.py not found")
        print("Make sure you're in the correct directory")
        return False
    except Exception as e:
        print(f"\nError generating initial conditions: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simulation(tag, steps, save_every, input_dir, output_file, verbose):
    """
    Run the gravothermal evolution simulation
    
    Args:
        tag: Name tag for this run
        steps: Number of simulation steps
        save_every: Save frequency
        input_dir: Directory containing initial conditions
        output_file: Path to save results
        verbose: Show detailed progress
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("STEP 2: RUNNING SIMULATION")
    print("="*70)
    
    try:
        from evolution import SimulationParameters, Simulator
        
        # Set up simulation parameters
        params = SimulationParameters(
            total_step=steps,
            save_step=save_every,
            tag=tag,
            input_dir=str(input_dir),
            output_file=str(output_file),
            verbose=verbose
        )
        
        print(f"\nConfiguration:")
        print(f"  Tag: {tag}")
        print(f"  Total steps: {steps:,}")
        print(f"  Save frequency: every {save_every} steps")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_file}")
        
        print(f"\nStarting simulation...")
        start_time = time.time()
        
        # Run simulation
        simulator = Simulator(params)
        simulator.initialize()
        simulator.run_simulation()
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Simulation completed successfully")
        print(f"  Execution time: {elapsed:.2f} seconds")
        print(f"  Performance: {steps/elapsed:.1f} steps/second")
        print(f"  Results saved to: {output_file}")
        
        return True
        
    except ImportError:
        print("\nError: evolution.py not found")
        print("Make sure you're in the correct directory")
        return False
    except FileNotFoundError as e:
        print(f"\nError: Could not find initial conditions")
        print(f"  {e}")
        print(f"\nMake sure initial condition files exist or use --initialize")
        return False
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate initial conditions and run gravothermal simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow: generate IC and run 10,000 steps
  python run_full_simulation.py --tag myrun --initialize --steps 10000
  
  # Just run simulation (IC already exists)
  python run_full_simulation.py --tag run01 --steps 50000
  
  # Generate new IC with 200 layers, then run
  python run_full_simulation.py --tag run02 --initialize --layers 200 --steps 100000
  
  # Quick test run (1000 steps, verbose)
  python run_full_simulation.py --tag test --initialize --steps 1000 --verbose
        """
    )
    
    # Main options
    parser.add_argument('--tag', type=str, default='run01',
                       help='Name tag for this run (default: run01)')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of simulation steps (default: 10,000)')
    
    # Initialization options
    parser.add_argument('--initialize', action='store_true',
                       help='Generate new initial conditions before running')
    parser.add_argument('--layers', type=int, default=150,
                       help='Number of radial layers for IC (default: 150)')
    
    # Simulation options
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save output every N steps (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress during simulation')
    
    # Advanced options
    parser.add_argument('--sim-only', action='store_true',
                       help='Skip initialization even if --initialize is set')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    initial_dir = script_dir / "output" / "initial" / args.tag
    output_file = script_dir / "output" / f"result_{args.tag}.txt"
    
    print("="*70)
    print("GRAVOTHERMAL EVOLUTION: FULL WORKFLOW")
    print("="*70)
    print(f"\nRun tag: {args.tag}")
    
    # Step 1: Initialize (if requested)
    if args.initialize and not args.sim_only:
        if not run_initialization(args.tag, args.layers, initial_dir.parent.parent):
            print("\n✗ Initialization failed")
            return 1
    else:
        if not initial_dir.exists():
            print(f"\n⚠ Warning: Initial condition directory not found: {initial_dir}")
            print("Consider using --initialize to generate initial conditions")
        else:
            print(f"\n⊳ Using existing initial conditions from: {initial_dir}")
    
    # Step 2: Run simulation
    if not run_simulation(
        args.tag, args.steps, args.save_every,
        initial_dir, output_file, args.verbose
    ):
        print("\n✗ Simulation failed")
        return 1
    
    # Success!
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Open analyze_evolution.ipynb to visualize results")
    print(f"  2. Or use: python -c \"from validate_outputs import parse_output_file; ")
    print(f"     data = parse_output_file('{output_file}')\"")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
