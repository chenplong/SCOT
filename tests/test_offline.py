"""Regression checks for Datalog compression and isolated SQL namespaces."""
from pathlib import Path
import itertools
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from scot.backend import DfClosure, arities_of, closure, one_hop
from scot.core import Atom, Const, Facts, Rule, enumerate_derivations
from scot.offline import COTCompressor


def reference_closure(facts, rules):
    result = facts.copy()
    while True:
        additions = {head for rule in rules
                     for head, _ in enumerate_derivations(rule, result)}
        changed = False
        for pred, row in additions:
            changed = result.add(pred, row) or changed
        if not changed:
            return result


def exhaustive_closure(facts, rules):
    """Tiny independent oracle: ground every variable over the active domain."""
    result = set(facts.facts())
    domain = {v for _, row in result for v in row}
    domain.update(v for rule in rules for atom in (rule.head,) + rule.body
                  for v in atom.args if not isinstance(v, str))
    while True:
        additions = set()
        for rule in rules:
            variables = sorted({v for atom in rule.body for v in atom.args
                                if isinstance(v, str)})
            for values in itertools.product(domain, repeat=len(variables)):
                binding = dict(zip(variables, values))

                def ground(atom):
                    return atom.pred, tuple(binding.get(v, v) for v in atom.args)

                if all(ground(atom) in result for atom in rule.body):
                    additions.add(ground(rule.head))
        if additions <= result:
            return result
        result.update(additions)


class OfflineRegressionTests(unittest.TestCase):
    def assert_exact_compression(self, facts, rules, **kwargs):
        expected = reference_closure(facts, rules)
        result = COTCompressor(facts, rules, **kwargs).run()
        self.assertEqual(set(result.G_star.facts()), set(expected.facts()))
        self.assertTrue(set(result.G_c.facts()) <= set(facts.facts()))
        self.assertEqual(set(one_hop(result.G_c, result.omega).facts()),
                         set(expected.facts()))
        return result

    def test_transitive_chain_keeps_head_binding(self):
        facts = Facts({'t': {(1, 2), (2, 3)}})
        rules = [
            Rule(Atom('p', ('X', 'Y')),
                 (Atom('t', ('X', 'Y')), Atom('t', ('Y', 'Z')))),
            Rule(Atom('t', ('X', 'Z')),
                 (Atom('t', ('X', 'Y')), Atom('t', ('Y', 'Z')))),
        ]
        result = self.assert_exact_compression(facts, rules, T=1, Nb=1)
        self.assertTrue(result.G_star.has('p', (1, 2)))

    def test_transitive_chain_keeps_external_binding(self):
        facts = Facts({'t': {(1, 2), (2, 3)}, 'u': {(2,)}})
        rules = [
            Rule(Atom('p', ('X', 'Y')),
                 (Atom('t', ('X', 'Z')), Atom('t', ('Z', 'Y')),
                  Atom('u', ('Z',)))),
            Rule(Atom('t', ('X', 'Z')),
                 (Atom('t', ('X', 'Y')), Atom('t', ('Y', 'Z')))),
        ]
        result = self.assert_exact_compression(facts, rules, T=1, Nb=1)
        self.assertEqual(result.G_star.rows['p'], {(1, 3)})

    def test_delta_table_cannot_shadow_user_predicate(self):
        facts = Facts({'p': {(1,)}, 'p__d__': {(2,)},
                       'scot_facts_0': {(3,)}, 'scot_delta_0': {(4,)}})
        rules = [Rule(Atom('r', ('X',)), (Atom('p', ('X',)),)),
                 Rule(Atom('q', ('X',)), (Atom('p__d__', ('X',)),)),
                 Rule(Atom('s', ('X',)), (Atom('scot_facts_0', ('X',)),)),
                 Rule(Atom('v', ('X',)), (Atom('scot_delta_0', ('X',)),))]
        expected = reference_closure(facts, rules)
        self.assertEqual(set(closure(facts, rules).facts()), set(expected.facts()))
        result = self.assert_exact_compression(facts, rules, T=1, Nb=1)
        self.assertEqual(result.G_star.rows['r'], {(1,)})

    def test_incremental_numeric_and_symbolic_values_stay_distinct(self):
        symbol = Const('alice')
        rules = [Rule(Atom('q', ('X',)), (Atom('p', ('X',)),))]
        values = (symbol, -1, int(symbol), 1 << 100, -(1 << 100))
        all_facts = Facts({'p': {(value,) for value in values}})
        engine = DfClosure(rules, arities_of(all_facts, rules))
        engine.add([('p', (symbol,))])
        engine.add([('p', (value,)) for value in values[1:]])
        self.assertEqual(engine.rows['q'], {(value,) for value in values})
        self.assert_exact_compression(all_facts, rules, T=1, Nb=2)

    def test_repeated_head_and_typed_constants(self):
        text_one = Const('1')
        facts = Facts({'r': {(text_one,), (1,), (-1,)}})
        rules = [Rule(Atom('p', ('X', 'X')), (Atom('r', ('X',)),)),
                 Rule(Atom('q', (text_one,)), (Atom('r', (1,)),))]
        result = self.assert_exact_compression(facts, rules, T=2, Nb=4)
        self.assertEqual(result.G_star.rows['p'],
                         {(text_one, text_one), (1, 1), (-1, -1)})

    def test_small_programs_against_exhaustive_reference(self):
        predicates = {'p': 2, 'q': 2, 'p__d__': 2, 'u': 1, 'v': 1}
        domain = (0, 1, -1, Const('1'), Const('alice'))
        for seed in range(12):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                facts = Facts({pred: {tuple(rng.choice(domain) for _ in range(arity))
                                     for _ in range(4)}
                               for pred, arity in predicates.items()})
                rules = []
                for _ in range(5):
                    body = []
                    for _ in range(rng.randint(1, 2)):
                        pred = rng.choice(tuple(predicates))
                        body.append(Atom(pred, tuple(rng.choice(('X', 'Y', 'Z', 0, Const('1')))
                                                     for _ in range(predicates[pred]))))
                    variables = sorted({term for atom in body for term in atom.args
                                        if isinstance(term, str)})
                    pred = rng.choice(tuple(predicates))
                    choices = variables + [0, Const('1')]
                    rules.append(Rule(Atom(pred, tuple(rng.choice(choices)
                                                      for _ in range(predicates[pred]))),
                                      tuple(body)))
                expected = exhaustive_closure(facts, rules)
                self.assertEqual(set(closure(facts, rules).facts()), expected)
                result = self.assert_exact_compression(facts, rules, T=2, Nb=4)
                self.assertEqual(set(result.G_star.facts()), expected)


if __name__ == '__main__':
    unittest.main()
