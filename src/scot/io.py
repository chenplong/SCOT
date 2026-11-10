"""Text inputs, primitive query workloads and portable JSON artifacts."""
from __future__ import annotations

import ast
from collections import defaultdict
import io
import json
from pathlib import Path
import pickle
import re

from .core import Atom, CQ, Const, Facts, Rule


def split_atoms(text):
    """Split comma-separated atoms or terms without splitting quoted values."""
    parts, start, depth, quote, escaped = [], 0, 0, None, False
    for pos, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                raise ValueError('unmatched closing parenthesis')
        elif char == ',' and depth == 0:
            parts.append(text[start:pos].strip())
            start = pos + 1
    if quote or depth:
        raise ValueError('unclosed quote or parenthesis')
    parts.append(text[start:].strip())
    if any(not p for p in parts):
        raise ValueError('empty atom or term')
    return parts


def _constant(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('constants must be strings or integers')
    return Const(value) if isinstance(value, str) else value


def _text_term(text, variables=True):
    text = text.strip()
    if not text:
        raise ValueError('empty term')
    if text[:1] in ("'", '"'):
        value = ast.literal_eval(text)
        if not isinstance(value, str):
            raise ValueError('quoted constants must be strings')
        return Const(value)
    if re.fullmatch(r'-?\d+', text):
        return int(text)
    if variables and (text.startswith('?') or re.fullmatch(r'[A-Z][A-Za-z0-9_]*', text)):
        return text
    return Const(text)


def parse_atom(text):
    text = text.strip().removesuffix('.').rstrip()
    match = re.fullmatch(r'([^\s(]+)\((.*)\)', text)
    if not match:
        raise ValueError('bad atom: %s' % text)
    terms = split_atoms(match.group(2))
    for term in terms:
        if term[:1] not in ("'", '"') and any(c in term for c in '()'):
            raise ValueError('function terms are not supported')
    return Atom(match.group(1), tuple(_text_term(t) for t in terms))


def _split_rule(text):
    quote, escaped, depth = None, False, 0
    for pos, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and text[pos:pos + 2] == ':-':
            return text[:pos], text[pos + 2:]
    raise ValueError("expected 'head :- body.'")


def _validate_rule(rule):
    if not rule.body:
        raise ValueError('rule bodies must be nonempty')
    bound = {v for atom in rule.body for v in atom.args if isinstance(v, str)}
    if not {v for v in rule.head.args if isinstance(v, str)} <= bound:
        raise ValueError('head variables must occur in the body')


def parse_rules(path, drop_sameas=False):
    """Load positive Datalog, accepting prefix and predicate declarations.

    Prefix spellings are preserved, so fact predicates must use the same names.
    sameAs is an ordinary predicate unless explicitly excluded by the caller.
    """
    rules, seen, arities = [], set(), {}
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(('#', '.')) or re.match(r'(?i)^(?:@?prefix|base)\b', line):
            continue
        try:
            head, body = _split_rule(line)
            rule = Rule(parse_atom(head), tuple(dict.fromkeys(parse_atom(a) for a in split_atoms(body.strip().removesuffix('.').rstrip()))))
            _validate_rule(rule)
            if drop_sameas and any('sameAs' in a.pred for a in (rule.head,) + rule.body):
                continue
            for atom in (rule.head,) + rule.body:
                if atom.pred in arities and arities[atom.pred] != len(atom.args):
                    raise ValueError('inconsistent arity for %s' % atom.pred)
                arities[atom.pred] = len(atom.args)
            if rule not in seen:
                seen.add(rule)
                rules.append(rule)
        except (ValueError, SyntaxError) as error:
            raise ValueError('%s:%d: %s' % (path, line_no, error)) from error
    return rules


def load_facts(path):
    tables, arities = defaultdict(set), {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip('\r\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2 or not parts[0]:
                raise ValueError('%s:%d: expected predicate and TSV arguments' % (path, number))
            pred, values = parts[0], tuple(_text_term(x, variables=False) for x in parts[1:])
            if pred in arities and arities[pred] != len(values):
                raise ValueError('%s:%d: inconsistent arity for %s' % (path, number, pred))
            arities[pred] = len(values)
            tables[pred].add(values)
    return Facts(tables)


def parse_query(text):
    head_text, body_text = _split_rule(text.strip().removesuffix('.').rstrip())
    head_text = head_text.strip()
    if '(' in head_text:
        match = re.fullmatch(r'[^\s(]+\((.*)\)', head_text)
        if not match:
            raise ValueError('invalid query head')
        head_text = match.group(1)
    head = tuple(x.strip() for x in head_text.split(',') if x.strip())
    body = tuple(parse_atom(x) for x in split_atoms(body_text))
    variables = {term for atom in body for term in atom.args if isinstance(term, str)}
    if not set(head) <= variables:
        raise ValueError('every answer variable must occur in the body')
    return CQ(head, body)


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('class loading is disabled for query workloads')

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('persistent objects are disabled')


def query_from_record(record):
    body = []
    for source in record['atoms']:
        terms = []
        for term in source['terms']:
            if not isinstance(term, dict) or len(term) != 1:
                raise ValueError('query terms must be {var: name} or {const: value}')
            if 'var' in term and isinstance(term['var'], str):
                terms.append(term['var'])
            elif 'const' in term:
                terms.append(_constant(term['const']))
            else:
                raise ValueError('invalid query term')
        body.append(Atom(str(source['predicate']), tuple(terms)))
    if not body:
        raise ValueError('query bodies must be nonempty')
    head = tuple(record.get('projection', ()))
    bound = {v for atom in body for v in atom.args if isinstance(v, str)}
    if not all(isinstance(v, str) for v in head) or not set(head) <= bound:
        raise ValueError('every projected variable must occur in the body')
    return CQ(head, tuple(body))


def load_queries(path):
    """Return (query ID, CQ) pairs without importing any pickle classes."""
    path = Path(path)
    if path.suffix.lower() in ('.pkl', '.pickle'):
        payload = PrimitiveUnpickler(io.BytesIO(path.read_bytes())).load()
    elif path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        return [('q%d' % number, parse_query(line)) for number, line in
                enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1)
                if line.strip() and not line.lstrip().startswith('#')]
    records = payload.get('queries') if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError('a workload must contain a queries list')
    queries, ids = [], set()
    for number, record in enumerate(records, 1):
        if isinstance(record, str):
            qid, query = 'q%d' % number, parse_query(record)
        elif isinstance(record, dict):
            qid = str(record.get('id', 'q%d' % number))
            query = parse_query(record['query']) if 'query' in record else query_from_record(record)
        else:
            raise ValueError('query records must be objects or CQ strings')
        if qid in ids:
            raise ValueError('duplicate query ID: %s' % qid)
        ids.add(qid)
        queries.append((qid, query))
    return queries


def _ground_value(value):
    if isinstance(value, Const):
        return value.name
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError('ground constants must be strings or integers; use Const for symbols')


def _atom_record(atom):
    return {'predicate': atom.pred,
            'terms': [{'var': t} if isinstance(t, str) else {'const': _ground_value(t)} for t in atom.args]}


def _record_atom(record):
    return Atom(record['predicate'], tuple(t['var'] if 'var' in t else _constant(t['const']) for t in record['terms']))


def save_artifact(path, facts, omega, stats=None, *, symbol_dictionary=None):
    """Store the graph and COT rules using typed JSON values.

    The symbol_dictionary keyword is accepted for callers of schema version 1.
    """
    tables = defaultdict(list)
    for pred, values in facts.facts():
        tables[pred].append([_ground_value(v) for v in values])
    cot = []
    for rule in omega:
        cot.append({'head': _atom_record(rule.head), 'body': [_atom_record(a) for a in rule.body]})
    payload = {'format': 'scot-json', 'schema_version': 2, 'graph': dict(tables),
               'cot': cot, 'stats': stats or {}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')


def load_artifact(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if payload.get('format') != 'scot-json' or payload.get('schema_version') not in (1, 2):
        raise ValueError('expected a SCOT JSON deployment artifact')
    tables = {}
    for pred, rows in payload['graph'].items():
        arities = {len(row) for row in rows}
        if len(arities) > 1:
            raise ValueError('inconsistent graph arity for %s' % pred)
        tables[pred] = {tuple(_constant(v) for v in row) for row in rows}
    facts = Facts(tables)
    omega = [Rule(_record_atom(r['head']), tuple(_record_atom(a) for a in r['body'])) for r in payload['cot']]
    return {'G_c': facts, 'G_next': facts, 'omega': omega, 'stats': payload.get('stats', {}),
            # Exact answers are decoded by the query engine.
            'symbol_dictionary': {}, 'schema_version': payload['schema_version']}


def decode_rows(rows, symbol_dictionary):
    return [tuple(value.name if isinstance(value, Const) else
                  symbol_dictionary.get(str(value), value) for value in row) for row in rows]
