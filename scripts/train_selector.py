"""Train the cost-sensitive mode policy from measured per-mode costs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.training import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--min-gain', type=float, default=1e-9)
    parser.add_argument('--objective', choices=['relative','absolute','log'], default='relative')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    args = parser.parse_args()
    if args.max_depth<0 or args.min_samples_leaf<1 or args.min_gain<0 or args.epsilon<=0:
        parser.error('invalid tree parameters')
    print(json.dumps(train_model(args.input, args.out, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, min_gain=args.min_gain,
        objective=args.objective, epsilon=args.epsilon)))


if __name__ == '__main__':
    main()
