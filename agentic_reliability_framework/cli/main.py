"""ARF CLI entry point."""
import sys
import argparse
from .diagnose import main as diagnose_main

def main():
    parser = argparse.ArgumentParser(description="Agentic Reliability Framework CLI")
    parser.add_argument("command", choices=["diagnose"], help="Command to run")
    args, remaining = parser.parse_known_args()

    if args.command == "diagnose":
        sys.argv = [sys.argv[0]] + remaining
        return diagnose_main()
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
