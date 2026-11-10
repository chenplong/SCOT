"""Isolated execution shared by querying and selector-label collection."""
from __future__ import annotations

import multiprocessing as mp
import statistics
import time

ANOMALY_COST_SECONDS = 240.0
MODES = ('P00', 'P01', 'P10', 'P11')


def _error_result(error):
    message = str(error)
    oom = isinstance(error, MemoryError) or any(word in message.lower()
        for word in ('out of memory', 'memory exhausted', 'memory limit',
                     'resources exhausted', 'resourcesexhausted', 'memoryerror'))
    timeout = isinstance(error, TimeoutError) or 'timed out' in message.lower()
    return {'status': 'oom' if oom else 'timeout' if timeout else 'error',
            'elapsed_seconds': 0, 'effective_cost_seconds': ANOMALY_COST_SECONDS,
            'error': message}


def _worker(connection, artifact_path, query, model, memory_gb, partitions, probe):
    try:
        from .io import load_artifact
        from .online.engine import COTQueryEngine
        artifact = load_artifact(artifact_path)
        engine = COTQueryEngine(artifact['G_next'], artifact['omega'],
            model_path=model, memory_gb=memory_gb, partitions=partitions)
        features = None
        if probe:
            from .online.factorized import query_frontiers
            from .online.features import flatten_features, query_raw_features
            q, groups, plans, _ = engine.compile_modes(query)
            features = flatten_features(query_raw_features(q,
                query_frontiers(q.answer_vars, q.body), groups), plans)
        connection.send({'ready': True, 'features': features})
        while True:
            request = connection.recv()
            if request is None:
                break
            mode, exact = request
            started = time.perf_counter()
            try:
                result = engine.execute(query, mode=mode, count=not exact)
                result.pop('plan', None)
                result.pop('sql', None)
                if exact and 'rows' in result:
                    # IPC/JSON results contain original values, not database IDs.
                    from .core import Const
                    result['rows'] = [tuple(value.name if isinstance(value, Const) else value
                                            for value in row) for row in result['rows']]
                connection.send({'status': 'ok',
                    'elapsed_seconds': time.perf_counter()-started,
                    'result': result})
            except Exception as exc:
                result = _error_result(exc)
                result['elapsed_seconds'] = time.perf_counter()-started
                connection.send(result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({'ready': False, 'error': repr(exc)})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class QueryWorker:
    def __init__(self, artifact, query, *, model=None, memory_gb=8,
                 partitions=1, probe=False, initialization_timeout=600):
        self.arguments = (str(artifact), query, str(model) if model else None,
                          memory_gb, partitions, probe)
        self.initialization_timeout = initialization_timeout
        self.process = self.connection = None
        self.features = None
        self.generation = 0

    def start(self):
        if self.process is not None:
            if self.process.is_alive() and self.connection is not None:
                return
            self.close()
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_worker, args=(child,)+self.arguments)
        try:
            self.process.start()
        finally:
            child.close()
        if not parent.poll(self.initialization_timeout):
            self.close()
            raise TimeoutError('worker initialization timed out')
        try:
            message = parent.recv()
        except EOFError as exc:
            self.close()
            raise RuntimeError('worker exited during initialization') from exc
        if not message.get('ready'):
            self.close()
            raise RuntimeError(message.get('error', 'worker initialization failed'))
        self.features = message['features']
        self.generation += 1

    def measure(self, mode, timeout=120, exact=False):
        if self.process is None:
            try:
                self.start()
            except Exception as exc:
                self.close()
                return _error_result(exc)
        session = self.generation
        worker_pid = self.process.pid
        try:
            self.connection.send((mode, exact))
            if not self.connection.poll(timeout):
                self.close()
                return {'status': 'timeout', 'elapsed_seconds': timeout,
                        'effective_cost_seconds': ANOMALY_COST_SECONDS,
                        'session_id': session, 'worker_pid': worker_pid}
            result = self.connection.recv()
        except (EOFError, BrokenPipeError, OSError):
            self.close()
            return {'status': 'error', 'elapsed_seconds': 0,
                    'effective_cost_seconds': ANOMALY_COST_SECONDS,
                    'session_id': session,
                    'worker_pid': worker_pid,
                    'error': 'worker exited without a result'}
        if 'status' not in result:
            self.close()
            return {'status': 'error', 'elapsed_seconds': 0,
                    'effective_cost_seconds': ANOMALY_COST_SECONDS,
                    'session_id': session,
                    'worker_pid': worker_pid,
                    'error': result.get('error', 'invalid worker result')}
        result['session_id'] = session
        result['worker_pid'] = worker_pid
        if result['status'] != 'ok':
            self.close()
        return result

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            if self.process.pid is not None:
                self.process.join(timeout=5)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def effective_cost(measurement):
    return measurement['elapsed_seconds'] if measurement['status']=='ok' else ANOMALY_COST_SECONDS


def measure_repeated(worker, mode, *, repeats=5, warmups=1, timeout=120, exact=False):
    """Warm and measure a query on one session, rewarming after a failed session.

    A failed warm-up consumes an incomplete measurement slot, marked explicitly
    with ``phase=warmup`` and ``measurement_executed=False``. This prevents a
    restarted, unprepared session from silently contributing cold timings and
    bounds retries when a query fails on every attempt.
    """
    if repeats < 1 or warmups < 0 or timeout <= 0:
        raise ValueError('invalid measurement parameters')
    warm, timed = [], []
    warmed_session = None
    for repetition in range(repeats):
        try:
            worker.start()
        except Exception as exc:
            worker.close()
            timed.append({**_error_result(exc), 'phase': 'initialization',
                          'repetition': repetition, 'measurement_executed': False})
            warmed_session = None
            continue
        session = worker.generation
        if session != warmed_session:
            failure = None
            for index in range(warmups):
                run = worker.measure(mode, timeout, exact)
                run.update(phase='warmup', before_repetition=repetition,
                           warmup_index=index, session_id=session)
                warm.append(run)
                if run['status'] != 'ok':
                    failure = run
                    worker.close()
                    break
            if failure is not None:
                timed.append({**failure, 'repetition': repetition,
                              'measurement_executed': False})
                warmed_session = None
                continue
            warmed_session = session
        run = worker.measure(mode, timeout, exact)
        run.update(phase='measurement', repetition=repetition,
                   measurement_executed=True, session_id=session)
        timed.append(run)
        if run['status'] != 'ok':
            worker.close()
            warmed_session = None
    return warm, timed


def collect_query(worker, *, repeats=5, warmups=1, timeout=120):
    try:
        worker.start()
    except Exception as exc:
        worker.close()
        from .online.features import flatten_features, query_raw_features
        from .online.model import ConjunctiveQuery
        names = flatten_features(query_raw_features(ConjunctiveQuery((), ()), (), ()), {})
        features = {name: float('nan') for name in names}
        failure = _error_result(exc)
        measured = {mode: [dict(failure) for _ in range(repeats)] for mode in MODES}
        warm = {mode: [dict(failure) for _ in range(warmups)] for mode in MODES}
        return features, {mode: ANOMALY_COST_SECONDS for mode in MODES}, measured, warm, False
    features = dict(worker.features)
    measured, warm = {}, {}
    for mode in MODES:
        warm[mode], measured[mode] = measure_repeated(worker, mode,
            repeats=repeats, warmups=warmups, timeout=timeout)
    costs = {mode: statistics.median(effective_cost(run) for run in measured[mode])
             for mode in MODES}
    counts = {run['result']['count'] for mode in MODES for run in measured[mode]
              if run['status'] == 'ok'}
    valid = (len(counts) == 1 and
             not all(any(run['status']!='ok' for run in measured[mode]) for mode in MODES))
    return features, costs, measured, warm, valid
