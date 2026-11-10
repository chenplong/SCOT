"""Answer CQs using the one SCOT compressed-graph query engine."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.execution import QueryWorker, effective_cost, measure_repeated, _error_result
from scot.io import decode_rows, load_artifact, load_queries, parse_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--query', help='CQ such as Y :- tc(0,Y).')
    inputs.add_argument('--queries', help='text, JSON or supplied workload pickle')
    parser.add_argument('--mode', choices=['auto','p00','p01','p10','p11'], default='auto')
    parser.add_argument('--model')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--memory-gb', type=int, default=8)
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.repeats<1 or args.warmups<0 or args.timeout<=0 or args.memory_gb<1 or args.partitions<1:
        parser.error('invalid measurement parameters')
    try:
        queries = [('q0',parse_query(args.query))] if args.query else load_queries(args.queries)
        if not queries:
            raise ValueError('query workload is empty')
        dictionary = load_artifact(args.artifact).get('symbol_dictionary',{}) if args.exact else {}
    except Exception as exc:
        failure = _error_result(exc)
        output = {'id': None, 'mode': args.mode, **failure, 'phase': 'preparation',
                  'warmup_measurements': [], 'measurements': [], 'answer': {}}
        print(json.dumps(output,default=str),flush=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps([output],indent=2,default=str)+'\n',encoding='utf-8')
        return 1
    outputs = []
    for qid, query in queries:
        worker = QueryWorker(args.artifact,query,model=args.model,
            memory_gb=args.memory_gb,partitions=args.partitions)
        try:
            warm, timed = measure_repeated(worker,
                None if args.mode=='auto' else args.mode, repeats=args.repeats,
                warmups=args.warmups, timeout=args.timeout, exact=args.exact)
        finally:
            worker.close()
        success = all(run['status']=='ok' for run in timed)
        answer = timed[-1].get('result',{})
        if args.exact and 'rows' in answer:
            answer['rows'] = decode_rows(answer['rows'],dictionary)
        output = {'id':qid,'mode':args.mode,'status':'ok' if success else next(
            run['status'] for run in timed if run['status']!='ok'),
            'effective_cost_seconds':statistics.mean(effective_cost(run) for run in timed)
                if success else 240.0,
            'warmup_measurements':warm,'measurements':timed,'answer':answer}
        outputs.append(output)
        print(json.dumps(output,default=str),flush=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(outputs,indent=2,default=str)+'\n',encoding='utf-8')
    return 0 if all(row['status']=='ok' for row in outputs) else 1


if __name__ == '__main__':
    raise SystemExit(main())
