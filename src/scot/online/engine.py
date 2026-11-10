"""Answer CQs over a retained graph and its complete one-hop COT rules.

The offline interface uses Atom.pred/args, Rule.head/body and CQ.head/body.
Strings in core terms are variables; core.Const values are symbolic constants.
Only SQL storage uses database-local integer codes, decoded in exact answers.
Facts can be a Facts object exposing .rows or a predicate-to-tuple-set mapping.
"""
from __future__ import annotations

from ..core import Const, ConstantCodec, ground_row_key
from .candidate import CandidateGenerator
from .factorized import query_frontiers
from .features import flatten_features, query_raw_features
from .model import (Atom, Constant, ConjunctiveQuery, Rule, Variable)
from .policy import choose_mode, load_model
from .sql_compiler import CQSqlCompiler, EmptyQuery

MODES = {'p00': (False, False), 'p01': (False, True),
         'p10': (True, False), 'p11': (True, True)}
MODE_ORDER = ('P00', 'P01', 'P10', 'P11')


def _term(value):
    return Variable(value) if isinstance(value, str) else Constant(value)


def _atom(atom):
    if isinstance(atom, Atom):
        return atom
    return Atom(str(atom.pred), tuple(_term(value) for value in atom.args))


def _query(query):
    if isinstance(query, ConjunctiveQuery):
        return query
    return ConjunctiveQuery(tuple(Variable(value) for value in query.head),
                            tuple(_atom(atom) for atom in query.body))


def _rules(rules):
    return tuple(rule if isinstance(rule, Rule) else
                 Rule(_atom(rule.head), tuple(_atom(atom) for atom in rule.body),
                      'R%d' % index)
                 for index, rule in enumerate(rules))


def _new_context(memory_gb, temp_dir, partitions):
    from ..backend import new_ctx
    return new_ctx(mem_gb=memory_gb, partitions=partitions, temp_dir=temp_dir)


class COTQueryEngine:
    def __init__(self, facts, omega, *, model_path=None, memory_gb=8,
                 temp_dir=None, partitions=None):
        self.facts = facts
        self.rows = facts.rows if hasattr(facts, 'rows') else facts
        self.omega = tuple(omega)
        self.codec = ConstantCodec()
        self.cot_rules = tuple(Rule(self._encode_atom(rule.head),
            tuple(self._encode_atom(atom) for atom in rule.body), rule.rule_id)
            for rule in _rules(self.omega))
        self.model = load_model(model_path)
        self.ctx = _new_context(float(memory_gb), temp_dir, partitions)
        self.table_aliases = self._register_tables()
        self.generator = CandidateGenerator(self.cot_rules)

    def _encode_atom(self, atom):
        terms = []
        for term in atom.terms:
            if isinstance(term, Constant):
                value = Const(term.value) if isinstance(term.value, str) else term.value
                term = Constant(self.codec.encode(value))
            terms.append(term)
        return Atom(atom.predicate, tuple(terms))

    def _register_tables(self):
        import pyarrow as pa

        arities = {}
        for rule in self.cot_rules:
            for atom in (rule.head,) + rule.body:
                old = arities.setdefault(atom.predicate, atom.arity)
                if old != atom.arity:
                    raise ValueError('inconsistent arity: %s' % atom.predicate)
        for predicate, rows in self.rows.items():
            for row in rows:
                old = arities.setdefault(predicate, len(row))
                if old != len(row):
                    raise ValueError('inconsistent arity: %s' % predicate)
        aliases = {}
        for index, predicate in enumerate(sorted(self.rows)):
            if predicate not in arities:
                continue
            alias = 'kg_p%03d' % index
            arity = arities[predicate]
            rows = sorted(self.rows.get(predicate, ()), key=ground_row_key)
            columns = [pa.array([self.codec.encode(row[column]) for row in rows], type=pa.int64())
                       for column in range(arity)]
            names = ['arg%d' % column for column in range(arity)]
            if arity == 0:
                columns, names = [pa.array([1] * len(rows), type=pa.int64())], ['__one']
            batch = pa.RecordBatch.from_arrays(columns, names=names)
            self.ctx.register_record_batches(alias, [[batch]])
            aliases[predicate] = alias
        return aliases

    @staticmethod
    def _constant_lookup(value):
        value = str(value)
        return int(value) if value.lstrip('-').isdigit() else None

    def _groups(self, cq):
        original = _query(cq)
        query = ConjunctiveQuery(original.answer_vars,
                                tuple(self._encode_atom(atom) for atom in original.body))
        frontiers = query_frontiers(query.answer_vars, query.body)
        forbidden = {term.name for atom in query.body for term in atom.terms
                     if isinstance(term, Variable)}
        forbidden.update(term.name for term in query.answer_vars)
        groups = tuple(self.generator.for_atom(atom, index,
            exposed_vars=frontiers[index], forbidden_names=forbidden)
            for index, atom in enumerate(query.body))
        return query, groups

    def compile_modes(self, cq, modes=MODE_ORDER):
        query, groups = self._groups(cq)
        plans, sqls = {}, {}
        for mode in modes:
            name = str(mode).upper()
            if name.lower() not in MODES:
                raise ValueError('unknown mode: %s' % mode)
            qfse, ufa = MODES[name.lower()]
            compiler = CQSqlCompiler(self.table_aliases, self._constant_lookup,
                query_frontier_sensitive_execution=qfse, unary_frontier_absorption=ufa)
            try:
                sql, plan = compiler._compile_factorized_direct(groups, query.answer_vars)
                plan['__compile_empty__'] = False
            except EmptyQuery:
                sql, plan = 'SELECT 1 AS one WHERE FALSE', {'__compile_empty__': True}
            plan['__sql_chars__'] = len(sql)
            plans[name], sqls[name] = plan, sql
        return query, groups, plans, sqls

    def _choose_compiled_mode(self, query, groups, plans):
        frontiers = query_frontiers(query.answer_vars, query.body)
        features = flatten_features(query_raw_features(query, frontiers, groups), plans)
        return choose_mode(features, self.model)

    def choose_mode(self, cq):
        query, groups, plans, _ = self.compile_modes(cq)
        return self._choose_compiled_mode(query, groups, plans)

    def compile(self, cq, mode=None):
        automatic = mode is None or str(mode).lower() == 'auto'
        modes = MODE_ORDER if automatic else (str(mode).upper(),)
        query, groups, plans, sqls = self.compile_modes(cq, modes)
        chosen = self._choose_compiled_mode(query, groups, plans) if automatic else mode
        name = str(chosen).upper()
        return name, sqls[name], plans[name]

    def execute(self, cq, mode=None, count=True):
        chosen, sql, plan = self.compile(cq, mode)
        statement = 'SELECT COUNT(*) AS n FROM (%s) AS scot_q' % sql if count else sql
        batches = self.ctx.sql(statement).collect()
        if count:
            value = sum(int(batch.column(0)[0].as_py()) for batch in batches if batch.num_rows)
            return {'mode': chosen, 'count': value, 'plan': plan}
        rows = []
        for batch in batches:
            rows.extend(tuple(self.codec.decode(value) for value in row)
                        for row in zip(*(batch.column(column).to_pylist()
                                         for column in range(batch.num_columns))))
        if not _query(cq).answer_vars:
            rows = [()] if rows else []
        return {'mode': chosen, 'rows': rows, 'plan': plan}
