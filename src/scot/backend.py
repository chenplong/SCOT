"""DataFusion storage and semi-naive Datalog operations used by SCOT."""
import os
from pathlib import Path
from collections import defaultdict
import pyarrow as pa
from datafusion import RuntimeEnvBuilder, SessionConfig, SessionContext
from .core import Facts, is_var, ConstantCodec, ground_row_key

def new_ctx(mem_gb=None, partitions=None, temp_dir=None):
    memory = float(mem_gb if mem_gb is not None else os.environ.get('DF_MEM_GB', '8'))
    if memory <= 0:
        raise ValueError('memory must be positive')
    config = SessionConfig().set('datafusion.optimizer.repartition_aggregations', 'false')
    threads = partitions if partitions is not None else os.environ.get('DF_TARGET_PARTITIONS')
    if threads is not None:
        config = config.set('datafusion.execution.target_partitions', str(threads))
    runtime = RuntimeEnvBuilder().with_fair_spill_pool(int(memory * 1024 ** 3))
    temporary = temp_dir if temp_dir is not None else os.environ.get('DF_TMP_DIR')
    if temporary:
        temporary = Path(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        runtime = runtime.with_disk_manager_specified(temporary)
    else:
        runtime = runtime.with_disk_manager_os()
    return SessionContext(config=config, runtime=runtime)

class _Storage:
    """Keep user predicates and typed constants outside SQL's namespaces."""

    def _init_storage(self):
        self.codec = ConstantCodec()
        self._table_names = {}

    def _table(self, predicate, delta=False):
        if predicate not in self._table_names:
            self._table_names[predicate] = len(self._table_names)
        # Only these generated aliases are registered; user names never are.
        return 'scot_%s_%d' % ('delta' if delta else 'facts',
                               self._table_names[predicate])

    def _lit(self, value):
        return str(self.codec.encode(value))

    def _arrow(self, rows, arity):
        return pa.RecordBatch.from_arrays(
            [pa.array([self.codec.encode(row[i]) for row in rows],
                      type=pa.int64()) for i in range(arity)],
            names=['c%d' % i for i in range(arity)])

    def _empty_batch(self, arity):
        return self._arrow([], arity)

    def _sorted_rows(self, rows):
        return sorted(rows, key=ground_row_key)

class StageOneDelete(_Storage):

    def __init__(self, rules, rows_by_pred, mem_gb=None, partitions=None, log=None):
        self._init_storage()
        self.ctx = new_ctx(mem_gb=mem_gb, partitions=partitions)
        self.rules = []
        for R in rules:
            if not R.body:
                continue
            bvars = {x for a in R.body for x in a.args if is_var(x)}
            if any((is_var(t) and t not in bvars for t in R.head.args)):
                continue
            self.rules.append(R)
        self.rows = {p: set(rs) for p, rs in rows_by_pred.items()}
        self.arities = {}
        for p, rs in self.rows.items():
            if rs:
                self.arities[p] = len(next(iter(rs)))
        for R in self.rules:
            for a in (R.head,) + tuple(R.body):
                self.arities.setdefault(a.pred, len(a.args))
        self._registered = set()
        self.log = log or (lambda *a, **k: None)
        self.stats = {'deleted_1a': 0}

    def _register(self, pred):
        name = self._table(pred)
        try:
            self.ctx.deregister_table(name)
        except Exception:
            pass
        rs = self.rows.get(pred, set())
        bs = [self._arrow(self._sorted_rows(rs), self.arities[pred])] if rs else [self._empty_batch(self.arities[pred])]
        self.ctx.register_record_batches(name, [bs])
        self._registered.add(pred)

    def _ensure(self, pred):
        if pred not in self._registered:
            self._register(pred)

    def _facts(self, R):
        inner = self._sql(R)
        out = self.ctx.sql('SELECT * FROM (%s) AS q' % inner).collect()
        heads = []
        for b in out:
            if b.num_rows:
                cols = [b.column(j).to_pylist() for j in range(b.num_columns)]
                heads.extend(tuple(self.codec.decode(v) for v in row)
                             for row in zip(*cols))
        return set(heads)

    def _support(self, R):
        """Count distinct assignments to all body variables in current rows."""
        exprs, conds = ({}, [])
        for i, atom in enumerate(R.body):
            self._ensure(atom.pred)
            for j, term in enumerate(atom.args):
                column = 'b%d.c%d' % (i, j)
                if is_var(term):
                    if term in exprs:
                        conds.append('%s = %s' % (column, exprs[term]))
                    else:
                        exprs[term] = column
                else:
                    conds.append('%s = %s' % (column, self._lit(term)))
        projection = ', '.join(('%s AS v%d' % (column, i) for i, column in enumerate(exprs.values())))
        if not projection:
            projection = '1 AS matched'
        tables = ', '.join(('"%s" AS b%d' % (self._table(atom.pred), i) for i, atom in enumerate(R.body)))
        inner = 'SELECT DISTINCT %s FROM %s' % (projection, tables)
        if conds:
            inner += ' WHERE ' + ' AND '.join(conds)
        sql = 'SELECT COUNT(*) AS support FROM (%s) AS assignments' % inner
        return sum((int(batch.column(0)[0].as_py()) for batch in self.ctx.sql(sql).collect() if batch.num_rows))

    def _sql(self, R, extra=()):
        body = R.body
        exprs, conds = ({}, [])
        for i, a in enumerate(body):
            self._ensure(a.pred)
            for j, t in enumerate(a.args):
                col = 'b%d.c%d' % (i, j)
                if is_var(t):
                    if t in exprs:
                        conds.append('%s = %s' % (col, exprs[t]))
                    else:
                        exprs[t] = col
                else:
                    conds.append('%s = %s' % (col, self._lit(t)))
        sel = []
        for j, t in enumerate(R.head.args):
            sel.append('%s AS h%d' % (exprs[t] if is_var(t) else self._lit(t), j))
        frm = ', '.join(('"%s" AS b%d' % (self._table(a.pred), i) for i, a in enumerate(body)))
        for i in extra:
            for j, t in enumerate(body[i].args):
                if is_var(t):
                    sel.append('b%d.c%d AS x%d_%d' % (i, j, i, j))
                else:
                    sel.append('%s AS x%d_%d' % (self._lit(t), i, j))
        sql = 'SELECT %s FROM %s' % (', '.join(sel), frm)
        if conds:
            sql += ' WHERE ' + ' AND '.join(conds)
        return sql

    def violations_sql(self, R):
        """SQL returning up to one head instance of R (one-step over the
        stored rows) whose head fact is absent from the stored truth
        (anti-join).  Streaming, no materialisation, LIMIT 1."""
        body = R.body
        exprs, conds = ({}, [])
        for i, a in enumerate(body):
            self._ensure(a.pred)
            for j, t in enumerate(a.args):
                col = 'b%d.c%d' % (i, j)
                if is_var(t):
                    if t in exprs:
                        conds.append('%s = %s' % (col, exprs[t]))
                    else:
                        exprs[t] = col
                else:
                    conds.append('%s = %s' % (col, self._lit(t)))
        head = R.head
        self._ensure(head.pred)
        hconds, hseen = ([], {})
        for j, t in enumerate(head.args):
            if is_var(t):
                if t not in exprs:
                    raise ValueError('unsafe head variable %s in %s' % (t, R))
                if t in hseen:
                    hconds.append('h.c%d = h.c%d' % (hseen[t], j))
                else:
                    hseen[t] = j
                    hconds.append('h.c%d = %s' % (j, exprs[t]))
            else:
                hconds.append('h.c%d = %s' % (j, self._lit(t)))
        frm = ', '.join(('"%s" AS b%d' % (self._table(a.pred), i) for i, a in enumerate(body)))
        conds.append('NOT EXISTS (SELECT 1 FROM "%s" AS h WHERE %s)' % (self._table(head.pred), ' AND '.join(hconds)))
        sql = 'SELECT 1 FROM %s WHERE %s LIMIT 1' % (frm, ' AND '.join(conds))
        return sql

    def size(self):
        return sum((len(v) for v in self.rows.values()))

class DfClosure(_Storage):

    def __init__(self, rules, arities, mem_gb=None, partitions=None):
        self._init_storage()
        self.ctx = new_ctx(mem_gb=mem_gb, partitions=partitions)
        self.arities = {p: int(a) for p, a in arities.items()}
        self.rows = defaultdict(set)
        self.batches = {}
        self.delta = {}
        self.rules = []
        self.plans = []
        preds = set()
        for R in rules:
            if not R.body:
                continue
            for a in (R.head,) + tuple(R.body):
                self.arities.setdefault(a.pred, len(a.args))
            self.rules.append(R)
            preds.add(R.head.pred)
            preds.update((a.pred for a in R.body))
        self.preds = preds
        for p in preds:
            self.rows[p]
            self.batches[p] = []
            self.delta[p] = []
            self._register(p, [])
            self._register_delta(p, [])
        self._build_plans()

    def _register(self, pred, batches):
        name = self._table(pred)
        try:
            self.ctx.deregister_table(name)
        except Exception:
            pass
        bs = batches if batches else [self._empty_batch(self.arities[pred])]
        self.ctx.register_record_batches(name, [bs])

    def _register_delta(self, pred, batches):
        name = self._table(pred, delta=True)
        try:
            self.ctx.deregister_table(name)
        except Exception:
            pass
        bs = batches if batches else [self._empty_batch(self.arities[pred])]
        self.ctx.register_record_batches(name, [bs])

    def _build_plans(self):
        for R in self.rules:
            body = R.body
            exprs, conds, ok = ({}, [], True)
            for i, a in enumerate(body):
                for j, t in enumerate(a.args):
                    col = 'b%d.c%d' % (i, j)
                    if isinstance(t, str):
                        if t in exprs:
                            conds.append('%s = %s' % (col, exprs[t]))
                        else:
                            exprs[t] = col
                    else:
                        conds.append('%s = %s' % (col, self._lit(t)))
            sel = []
            for j, t in enumerate(R.head.args):
                if isinstance(t, str):
                    if t not in exprs:
                        ok = False
                        break
                    sel.append('%s AS h%d' % (exprs[t], j))
                else:
                    sel.append('%s AS h%d' % (self._lit(t), j))
            if not ok:
                continue
            for k in range(len(body)):
                frm = []
                for i, a in enumerate(body):
                    nm = self._table(a.pred, delta=i == k)
                    frm.append('"%s" AS b%d' % (nm, i))
                sql = 'SELECT DISTINCT %s FROM %s' % (', '.join(sel), ', '.join(frm))
                if conds:
                    sql += ' WHERE ' + ' AND '.join(conds)
                self.plans.append((R, k, sql))

    def _run(self, sql):
        out = self.ctx.sql(sql).collect()
        for b in out:
            if b.num_rows == 0:
                continue
            cols = [b.column(j).to_pylist() for j in range(b.num_columns)]
            yield from (tuple(self.codec.decode(v) for v in row)
                        for row in zip(*cols))

    def has(self, pred, row):
        return tuple(row) in self.rows[pred]

    def add(self, new_rows):
        """Insert rows, then close F semi-naively (DataFusion joins)."""
        by_pred = defaultdict(list)
        for pred, row in new_rows:
            t = tuple(row)
            if t in self.rows[pred]:
                continue
            self.rows[pred].add(t)
            by_pred[pred].append(t)
        if not by_pred:
            return
        for pred, rs in by_pred.items():
            self.arities.setdefault(pred, len(rs[0]))
            if pred not in self.batches:
                self.batches[pred] = []
                self.delta[pred] = []
                self._register(pred, [])
                self._register_delta(pred, [])
            b = self._arrow(rs, self.arities[pred])
            self.batches[pred].append(b)
            self.delta[pred].append(b)
            self._register(pred, self.batches[pred])
            self._register_delta(pred, self.delta[pred])
        while any((self.delta[p] for p in self.preds)):
            dnew = defaultdict(set)
            for R, k, sql in self.plans:
                if not self.delta.get(R.body[k].pred):
                    continue
                hp = R.head.pred
                for row in self._run(sql):
                    if row in self.rows[hp]:
                        continue
                    self.rows[hp].add(row)
                    dnew[hp].add(row)
            for p in self.preds:
                if p in dnew:
                    rs = list(dnew[p])
                    b = self._arrow(rs, self.arities[p])
                    self.batches[p].append(b)
                    self.delta[p] = [b]
                    self._register(p, self.batches[p])
                    self._register_delta(p, [b])
                else:
                    self.delta[p] = []
                    self._register_delta(p, [])

    def size(self):
        return sum((len(s) for s in self.rows.values()))

    def facts(self):
        out = Facts()
        for pred, s in self.rows.items():
            if s:
                out.rows[pred] = set(s)
        return out

class DfDerive(_Storage):

    def __init__(self, rules, base, arities, mem_gb=None, partitions=None):
        """base: dict pred -> set(rows); rules: list of core.Rule."""
        self._init_storage()
        self.ctx = new_ctx(mem_gb=mem_gb, partitions=partitions)
        self.arities = {p: int(a) for p, a in arities.items()}
        self.rules = []
        for R in rules:
            if not R.body:
                continue
            bvars = {x for a in R.body for x in a.args if is_var(x)}
            if any((is_var(t) and t not in bvars for t in R.head.args)):
                continue
            for a in (R.head,) + tuple(R.body):
                self.arities.setdefault(a.pred, len(a.args))
            self.rules.append(R)
        self.rows = defaultdict(set)
        self.batches = {p: [] for p in self.arities}
        self.delta = {p: [] for p in self.arities}
        self.level = {}
        self.M = {}
        for p in self.arities:
            self._register(p, [])
            self._register_delta(p, [])
        for p, rows in base.items():
            if rows:
                self.rows[p] |= set(rows)
        for p, rs in self.rows.items():
            if rs:
                self.batches[p] = [self._arrow(self._sorted_rows(rs), self.arities[p])]
                self._register(p, self.batches[p])
                self.delta[p] = list(self.batches[p])
                self._register_delta(p, self.delta[p])
                for r in rs:
                    self.level[p, r] = 0

    def _register(self, pred, batches):
        name = self._table(pred)
        try:
            self.ctx.deregister_table(name)
        except Exception:
            pass
        self.ctx.register_record_batches(name, [batches if batches else [self._empty_batch(self.arities[pred])]])

    def _register_delta(self, pred, batches):
        name = self._table(pred, delta=True)
        try:
            self.ctx.deregister_table(name)
        except Exception:
            pass
        self.ctx.register_record_batches(name, [batches if batches else [self._empty_batch(self.arities[pred])]])

    def _variant_sql(self, R, k):
        body = R.body
        sel, conds, exprs = ([], [], {})
        for i, a in enumerate(body):
            for j, t in enumerate(a.args):
                if is_var(t):
                    col = 'b%d.c%d' % (i, j)
                    if t in exprs:
                        conds.append('%s = %s' % (col, exprs[t]))
                    else:
                        exprs[t] = col
                else:
                    conds.append('b%d.c%d = %s' % (i, j, self._lit(t)))
        for j, t in enumerate(R.head.args):
            sel.append('%s AS h%d' % (exprs[t], j) if is_var(t) else '%s AS h%d' % (self._lit(t), j))
        for i, a in enumerate(body):
            for j, t in enumerate(a.args):
                if is_var(t):
                    sel.append('b%d.c%d AS b%d_%d' % (i, j, i, j))
                else:
                    sel.append('%s AS b%d_%d' % (self._lit(t), i, j))
        frm = []
        for i, a in enumerate(body):
            nm = self._table(a.pred, delta=i == k)
            frm.append('"%s" AS b%d' % (nm, i))
        sql = 'SELECT %s FROM %s' % (', '.join(sel), ', '.join(frm))
        if conds:
            sql += ' WHERE ' + ' AND '.join(conds)
        return sql

    def _run(self, sql):
        out = self.ctx.sql(sql).collect()
        for b in out:
            if b.num_rows:
                cols = [b.column(j).to_pylist() for j in range(b.num_columns)]
                # The final ROW_NUMBER output is metadata rather than a term.
                yield from (tuple(self.codec.decode(v) for v in row[:-1])
                            for row in zip(*cols))

    def run(self):
        ell = 0
        while any((self.delta[p] for p in self.arities)):
            ell += 1
            assigned = {}
            for R in self.rules:
                m = len(R.head.args)
                part = ', '.join(('t2.h%d' % j for j in range(m)))
                order = ', '.join(('t2.b%d_%d' % (i, j) for i, b in enumerate(R.body) for j in range(len(b.args))))
                for k, a in enumerate(R.body):
                    if not self.delta.get(a.pred):
                        continue
                    hpred = self._table(R.head.pred)
                    inner = self._variant_sql(R, k)
                    on = ' AND '.join(('kk.c%d = t.h%d' % (j, j) for j in range(m)))
                    sql = 'SELECT * FROM ( SELECT t2.*, ROW_NUMBER() OVER (PARTITION BY %s ORDER BY %s) AS __rn FROM (SELECT t.* FROM (%s) AS t LEFT JOIN "%s" AS kk ON %s WHERE kk.c0 IS NULL) AS t2 ) AS t3 WHERE __rn = 1' % (part, order, inner, hpred, on)
                    for row in self._run(sql):
                        head = (R.head.pred, tuple(row[:m]))
                        if head in self.level or head in assigned:
                            continue
                        body, off = ([], m)
                        for b in R.body:
                            body.append((b.pred, tuple(row[off:off + len(b.args)])))
                            off += len(b.args)
                        assigned[head] = (R, tuple(body))
            if not assigned:
                break
            by_pred = defaultdict(list)
            for (pred, _r), (R, body) in assigned.items():
                by_pred[pred].append(_r)
                self.M[pred, _r] = (R, body)
            for pred, rs in by_pred.items():
                self.batches[pred].append(self._arrow(self._sorted_rows(rs), self.arities[pred]))
                self.rows[pred] |= set(rs)
                for r in rs:
                    self.level[pred, r] = ell
            for pred in self.arities:
                self.delta[pred] = []
                self._register_delta(pred, [])
            for pred, rs in by_pred.items():
                b = self.batches[pred][-1]
                self.delta[pred] = [b]
                self._register_delta(pred, [b])
                self._register(pred, self.batches[pred])
        return self

def df_fixpoint_with_derive(rules, base, arities, mem_gb=None, partitions=None):
    d = DfDerive(rules, base, arities, mem_gb=mem_gb, partitions=partitions).run()
    return (d.rows, d.level, d.M)

def arities_of(facts, rules):
    arities = {}
    for predicate, rows in facts.rows.items():
        if rows:
            arities[predicate] = len(next(iter(rows)))
    for rule in rules:
        for atom in (rule.head,) + tuple(rule.body):
            arity = len(atom.args)
            if atom.pred in arities and arities[atom.pred] != arity:
                raise ValueError('Predicate arity mismatch: ' + atom.pred)
            arities[atom.pred] = arity
    return arities

def closure(facts, rules, mem_gb=None, partitions=None):
    facts = facts if isinstance(facts, Facts) else Facts(facts)
    rules = list(rules)
    engine = DfClosure(rules, arities_of(facts, rules), mem_gb=mem_gb, partitions=partitions)
    engine.add(facts.facts())
    return engine.facts()

def one_hop(facts, rules, mem_gb=None):
    """Evaluate every rule body on the same retained graph, without feedback."""
    facts = facts if isinstance(facts, Facts) else Facts(facts)
    rules = list(rules)
    evaluator = StageOneDelete(rules, facts.rows, mem_gb=mem_gb)
    result = facts.copy()
    for rule in rules:
        for row in evaluator._facts(rule):
            result.add(rule.head.pred, row)
    return result
