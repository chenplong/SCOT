"""Measure the four SCOT modes for selector training."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.execution import MODES, QueryWorker, collect_query, _error_result
from scot.io import load_queries
from scot.workloads import filter_training_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--holdout-queries', action='append', default=[],
                        help='exclude equivalents of these test queries before collecting labels; repeat for multiple files')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model')
    parser.add_argument('--memory-gb', type=int, default=256)
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--dataset-tag', default='custom')
    args = parser.parse_args()
    if args.repeats<1 or args.warmups<0 or args.timeout<=0 or args.memory_gb<1 or args.partitions<1:
        parser.error('invalid measurement parameters')
    try:
        queries = load_queries(args.queries)
        if args.holdout_queries:
            holdout = [record for path in args.holdout_queries for record in load_queries(path)]
            queries, exclusions = filter_training_queries(queries, holdout)
            print(json.dumps({'phase': 'workload_filter', 'excluded': exclusions,
                              'retained_queries': len(queries)}), flush=True)
    except Exception as exc:
        print(json.dumps({'phase': 'preparation', **_error_result(exc)}),flush=True)
        return 1
    if not queries:
        parser.error('query workload is empty')
    rows = []
    for qid, query in queries:
        worker = QueryWorker(args.artifact, query, model=args.model,
            memory_gb=args.memory_gb, partitions=args.partitions, probe=True)
        try:
            features, costs, measured, warm, valid = collect_query(worker,
                repeats=args.repeats, warmups=args.warmups, timeout=args.timeout)
        finally:
            worker.close()
        row = {'dataset_tag': args.dataset_tag, 'qid': qid, 'query': repr(query),
            'valid_for_training': int(valid), 'best_mode': min(MODES,key=costs.get),
            'warmups': args.warmups, 'repeats': args.repeats,
            'timeout_seconds': args.timeout, 'anomaly_cost_seconds': 240,
            'label_measurement_protocol': 'median_effective_cost_with_compilation',
            'label_cost_schema': 'median_named_columns_hold_repeated_effective_costs',
            'raw_measurements': json.dumps(measured, default=str),
            'warmup_measurements': json.dumps(warm, default=str)}
        row.update(features)
        row.update({mode+'_median_effective_cost_seconds':costs[mode] for mode in MODES})
        rows.append(row)
        print(json.dumps({'qid': qid, 'valid_for_training': int(valid), 'costs': costs}),flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if any(row['valid_for_training'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
