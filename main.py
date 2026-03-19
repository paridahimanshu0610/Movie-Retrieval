import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from pipeline.build_index import build_index
from pipeline.query import run_query, interactive_mode

def main():
    parser = argparse.ArgumentParser(description="Scene-based movie retrieval")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Build the index from clips/")

    search = subparsers.add_parser("query", help="Query the index")
    search.add_argument("text", nargs="*", help="Scene description")

    args = parser.parse_args()

    if args.command == "build":
        build_index()
    elif args.command == "query":
        if args.text:
            run_query(" ".join(args.text))
        else:
            interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()