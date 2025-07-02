#!/usr/bin/env python3
"""
Test runner script for Jorg

This script provides a convenient way to run different test suites with
appropriate configurations and reporting.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Jorg tests")
    parser.add_argument(
        "--suite", choices=["unit", "integration", "all", "synthesis", "continuum", "lines", "statmech"],
        default="all", help="Test suite to run"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Skip slow tests"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--html-report", action="store_true", help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test selection
    if args.suite == "unit":
        cmd.extend(["tests/test_synthesis.py", "tests/test_continuum.py", 
                   "tests/test_lines.py", "tests/test_statmech.py"])
    elif args.suite == "integration":
        cmd.append("tests/test_integration.py")
    elif args.suite == "synthesis":
        cmd.append("tests/test_synthesis.py")
    elif args.suite == "continuum":
        cmd.append("tests/test_continuum.py")
    elif args.suite == "lines":
        cmd.append("tests/test_lines.py")
    elif args.suite == "statmech":
        cmd.append("tests/test_statmech.py")
    else:  # all
        cmd.append("tests/")
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.coverage:
        cmd.extend(["--cov=jorg", "--cov-report=term-missing"])
        if args.html_report:
            cmd.append("--cov-report=html")
    
    if args.benchmark:
        cmd.append("--benchmark-only")
    
    # Run the tests
    success = run_command(cmd, f"Jorg {args.suite} tests")
    
    if args.coverage and args.html_report and success:
        print(f"\nCoverage report generated in: {script_dir}/htmlcov/index.html")
    
    if success:
        print(f"\n✅ All {args.suite} tests passed!")
        return 0
    else:
        print(f"\n❌ Some {args.suite} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())