"""Compress TSV facts and construct the one-hop COT rule set."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.io import load_facts, parse_rules, save_artifact
from scot.offline import COTCompressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--facts', required=True)
    parser.add_argument('--rules', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--Nb', type=int, default=4)
    parser.add_argument('--memory-gb', type=float, default=8)
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--temp-dir')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.T<1 or args.Nb<1 or args.memory_gb<=0 or args.partitions<1:
        parser.error('T, Nb, memory and partitions must be positive')
    os.environ['DF_MEM_GB'] = str(args.memory_gb)
    os.environ['DF_TARGET_PARTITIONS'] = str(args.partitions)
    if args.temp_dir:
        temp = Path(args.temp_dir).resolve()
        temp.mkdir(parents=True,exist_ok=True)
        os.environ['DF_TMP_DIR'] = str(temp)
    facts, rules = load_facts(args.facts), parse_rules(args.rules)
    result = COTCompressor(facts, rules, T=args.T, Nb=args.Nb,
                           mem_gb=args.memory_gb).run()
    if args.check and not result.check_coverage():
        raise RuntimeError('closure preservation or one-hop recovery check failed')
    save_artifact(args.out, result.G_c, result.omega, result.stats)
    print(json.dumps(result.stats,default=str))


if __name__ == '__main__':
    main()
