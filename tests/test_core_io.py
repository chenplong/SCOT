"""Typed-constant, pruning, and portable-artifact regression checks."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from scot.backend import one_hop
from scot.core import Atom, CQ, Const, ConstantCodec, Facts, Rule, remove_redundant
from scot.io import decode_rows, load_artifact, save_artifact
from scot.online.engine import COTQueryEngine


class TypedConstantTests(unittest.TestCase):
    def test_symbol_is_distinct_from_legacy_fingerprint(self):
        symbol = Const('alice')
        fingerprint = int(symbol)
        self.assertNotEqual(symbol, fingerprint)
        self.assertNotEqual(symbol, 'alice')
        self.assertEqual(len({symbol, fingerprint}), 2)
        self.assertEqual(Facts({'p': {(symbol,), (fingerprint,)}}).size(), 2)

    def test_dynamic_encoding_is_injective_in_every_insertion_order(self):
        values = (Const('1'), Const('alice'), -1, -2, int(Const('alice')))
        for ordered in itertools.permutations(values):
            codec = ConstantCodec()
            codes = [codec.encode(value) for value in ordered]
            self.assertEqual(len(set(codes)), len(values))
            self.assertEqual([codec.decode(code) for code in codes], list(ordered))
            # Adding later values must never change an earlier assignment.
            self.assertEqual([codec.encode(value) for value in ordered], codes)

    def test_large_integers_and_int64_boundaries_roundtrip(self):
        values = (0, 1, (1 << 63) - 1, -(1 << 63), 1 << 63,
                  -(1 << 63) - 1, 1 << 100, -(1 << 100), Const('0'))
        codec = ConstantCodec()
        codes = [codec.encode(value) for value in values]
        self.assertEqual(len(set(codes)), len(values))
        self.assertTrue(all(-(1 << 63) <= code < (1 << 63) for code in codes))
        self.assertEqual([codec.decode(code) for code in codes], list(values))
        self.assertEqual(codec.encode(0), 0)
        self.assertEqual(codec.encode((1 << 63) - 1), (1 << 63) - 1)


class RulePruningTests(unittest.TestCase):
    def test_equivalent_subsumers_keep_a_representative_and_one_hop_answers(self):
        first = Rule(Atom('h', ('X',)),
                     (Atom('e', ('X', 'Y')), Atom('e', ('X', 'Z'))))
        second = Rule(Atom('h', ('X',)), (Atom('e', ('X', 'Y')),))
        facts = Facts({'e': {(1, 2), (1, 5), (3, 4)}})
        expected = set(one_hop(facts, [first, second]).facts())
        for rules in ([first, second], [second, first]):
            pruned = remove_redundant(rules)
            self.assertEqual(len(pruned), 1)
            self.assertEqual(set(one_hop(facts, pruned).facts()), expected)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'graph.json'
        self.symbol = Const('alice')
        self.fingerprint = int(self.symbol)
        self.values = (self.symbol, self.fingerprint, Const('1'), 1, -1,
                       1 << 100, -(1 << 100))
        self.facts = Facts({'p': {(value,) for value in self.values}})
        self.rules = [Rule(Atom('p', ('X',)), (Atom('p', ('X',)),)),
                      Rule(Atom('q', (self.symbol,)),
                           (Atom('p', (self.fingerprint,)),))]

    def test_v2_preserves_original_values_and_rule_constants(self):
        save_artifact(self.path, self.facts, self.rules, {'check': True})
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(payload['schema_version'], 2)
        self.assertIn(['alice'], payload['graph']['p'])
        self.assertIn([self.fingerprint], payload['graph']['p'])
        restored = load_artifact(self.path)
        self.assertEqual(set(restored['G_c'].facts()), set(self.facts.facts()))
        self.assertEqual(restored['omega'], self.rules)
        self.assertEqual(restored['stats'], {'check': True})
        engine = COTQueryEngine(restored['G_c'], restored['omega'])
        query = CQ(('X',), (Atom('p', ('X',)),))
        for mode in ('p00', 'p01', 'p10', 'p11'):
            answer = engine.execute(query, mode, count=False)
            self.assertEqual(set(answer['rows']), {(value,) for value in self.values})

    def test_v1_legacy_dictionary_does_not_relabel_real_numeric_answers(self):
        save_artifact(self.path, self.facts, self.rules)
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        payload['schema_version'] = 1
        payload['symbol_dictionary'] = {str(self.fingerprint): 'alice'}
        self.path.write_text(json.dumps(payload), encoding='utf-8')
        restored = load_artifact(self.path)
        engine = COTQueryEngine(restored['G_c'], restored['omega'])
        query = CQ(('X',), (Atom('p', ('X',)),))
        for mode in ('p00', 'p01', 'p10', 'p11'):
            rows = engine.execute(query, mode, count=False)['rows']
            decoded = set(decode_rows(rows, restored['symbol_dictionary']))
            self.assertIn(('alice',), decoded)
            self.assertIn((self.fingerprint,), decoded)
            self.assertIn(('1',), decoded)
            self.assertIn((1,), decoded)
            self.assertEqual(len(decoded), len(self.values))

    def test_export_rejects_boolean_and_float_fact_constants(self):
        for invalid in (True, False, 1.25):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    save_artifact(self.path, Facts({'p': {(invalid,)}}), [])

    def test_export_rejects_boolean_and_float_rule_constants(self):
        for invalid in (True, False, 1.25):
            with self.subTest(invalid=invalid):
                rule = Rule(Atom('q', (invalid,)), (Atom('p', ('X',)),))
                with self.assertRaises((TypeError, ValueError)):
                    save_artifact(self.path, self.facts, [rule])


if __name__ == '__main__':
    unittest.main()
