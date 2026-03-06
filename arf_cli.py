#!/usr/bin/env python3
"""
ARF CLI Entry Point
Usage: arf diagnose [options]
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Agentic Reliability Framework CLI")
    parser.add_argument("command", choices=["diagnose"], help="Command to run")
    args, remaining = parser.parse_known_args()

    if args.command == "diagnose":
        from agentic_reliability_framework.cli.diagnose import main as diagnose_main
        sys.argv = [sys.argv[0]] + remaining
        sys.exit(diagnose_main())
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
