import sys
import tempfile
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

from scot.core import Atom, Facts, Rule
from scot.execution import QueryWorker, collect_query, measure_repeated
from scot.io import parse_query, save_artifact


class ScriptedWorker:
    def __init__(self, statuses=()):
        self.statuses = iter(statuses)
        self.generation = 0
        self.alive = False
        self.features = {'feature': 1}
        self.calls = []

    def start(self):
        if not self.alive:
            self.alive = True
            self.generation += 1

    def close(self):
        self.alive = False

    def measure(self, mode, timeout=120, exact=False):
        self.calls.append((self.generation, mode))
        status = next(self.statuses, 'ok')
        result = {'status': status, 'elapsed_seconds': 0.01,
                  'session_id': self.generation}
        if status == 'ok':
            result['result'] = {'count': 1}
        else:
            self.close()
        return result


class MeasurementProtocolTests(unittest.TestCase):
    def test_warmup_and_timed_repetitions_share_session(self):
        worker = ScriptedWorker()
        warm, timed = measure_repeated(worker, 'P00', warmups=2, repeats=3)
        self.assertEqual(len(warm), 2)
        self.assertEqual(len(timed), 3)
        self.assertEqual(worker.generation, 1)
        self.assertEqual({run['session_id'] for run in warm + timed}, {1})

    def test_failed_timed_session_restarts_and_rewarms(self):
        worker = ScriptedWorker(('ok', 'timeout', 'ok', 'ok'))
        warm, timed = measure_repeated(worker, 'P00', warmups=1, repeats=2)
        self.assertEqual([run['status'] for run in timed], ['timeout', 'ok'])
        self.assertEqual([run['session_id'] for run in warm], [1, 2])
        self.assertEqual(worker.calls, [(1, 'P00'), (1, 'P00'),
                                        (2, 'P00'), (2, 'P00')])

    def test_failed_warmup_is_explicit_and_never_records_cold_timing(self):
        worker = ScriptedWorker(('error', 'ok', 'ok'))
        warm, timed = measure_repeated(worker, 'P00', warmups=1, repeats=2)
        self.assertEqual(timed[0]['phase'], 'warmup')
        self.assertFalse(timed[0]['measurement_executed'])
        self.assertEqual(timed[1]['status'], 'ok')
        self.assertEqual([run['session_id'] for run in warm], [1, 2])

    def test_label_collection_uses_same_protocol_for_all_modes(self):
        worker = ScriptedWorker()
        features, costs, measured, warm, valid = collect_query(
            worker, repeats=2, warmups=1)
        self.assertTrue(valid)
        self.assertEqual(worker.generation, 1)
        self.assertEqual(len(worker.calls), 12)
        self.assertTrue(all(len(warm[m]) == 1 and len(measured[m]) == 2
                            for m in costs))

    def test_real_worker_preserves_warmed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'graph.json'
            atom = Atom('r', ('X',))
            save_artifact(artifact, Facts({'r': {(1,)}}), [Rule(atom, (atom,))])
            worker = QueryWorker(artifact, parse_query('X :- r(X).'))
            try:
                warm, timed = measure_repeated(worker, 'p00', warmups=1, repeats=2)
                self.assertEqual([r['status'] for r in warm + timed], ['ok'] * 3)
                self.assertEqual({r['session_id'] for r in warm + timed}, {1})
                self.assertEqual({r['worker_pid'] for r in warm + timed}, {worker.process.pid})
                self.assertEqual([r['result']['count'] for r in timed], [1, 1])
                pid = worker.process.pid
                worker.start()
                self.assertEqual(worker.process.pid, pid)
            finally:
                worker.close()


if __name__ == '__main__':
    unittest.main()
