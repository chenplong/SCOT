"""Online semantic regressions for all four factorized execution modes."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from scot.core import Atom, CQ, Const, Facts, Rule
from scot.io import load_artifact, save_artifact
from scot.offline import COTCompressor
from scot.online.candidate import CandidateGenerator
from scot.online.engine import COTQueryEngine
from scot.online.model import (Atom as OnlineAtom, Constant, QueryItem,
                               Rule as OnlineRule, Variable)
from scot.online.sql_compiler import CQSqlCompiler


MODES = ('p00', 'p01', 'p10', 'p11')


class IdentityRuleTests(unittest.TestCase):
    def test_identity_uses_term_types_and_values(self):
        cases = ((Constant(1), Constant('1')),
                 (Variable('X'), Constant('?X')))
        for left, right in cases:
            with self.subTest(left=left, right=right):
                rule = OnlineRule(OnlineAtom('p', (left, right)),
                                  (OnlineAtom('p', (right, left)),))
                self.assertFalse(CandidateGenerator._is_identity(rule))
        atom = OnlineAtom('p', (Variable('X'), Constant(1)))
        self.assertTrue(CandidateGenerator._is_identity(OnlineRule(atom, (atom,))))

    def test_numeric_and_quoted_numeric_constant_roundtrip(self):
        symbol = Const('1')
        facts = Facts({'p': {(symbol, 1)}})
        rule = Rule(Atom('p', (1, symbol)), (Atom('p', (symbol, 1)),))
        compressed = COTCompressor(facts, [rule], T=2, Nb=4).run()
        self.assertTrue(compressed.coverage_ok)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'artifact.json'
            save_artifact(path, compressed.G_c, compressed.omega)
            artifact = load_artifact(path)
            engine = COTQueryEngine(artifact['G_c'], artifact['omega'])
            query = CQ((), (Atom('p', (1, symbol)),))
            for mode in MODES:
                with self.subTest(mode=mode):
                    self.assertEqual(set(engine.execute(query, mode, count=False)['rows']),
                                     {()})

    def test_variable_style_symbol_is_not_an_identity_variable(self):
        symbol = Const('?V0')
        facts = Facts({'p': {(symbol, 1)}})
        rule = Rule(Atom('p', ('X', symbol)), (Atom('p', (symbol, 'X')),))
        compressed = COTCompressor(facts, [rule], T=2, Nb=4).run()
        engine = COTQueryEngine(compressed.G_c, compressed.omega)
        query = CQ(('Y',), (Atom('p', ('Y', symbol)),))
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(set(engine.execute(query, mode, count=False)['rows']),
                                 {(1,)})


class MappedUnaryAbsorptionTests(unittest.TestCase):
    @staticmethod
    def engine_and_query(left, right):
        facts = Facts({'r': {(1, 10), (2, 20), (3, 30)},
                       's': {(10,), (20,), (30,)},
                       'u': {(value,) for value in left},
                       'v': {(value,) for value in right}})
        rule = Rule(Atom('p', ('A', 'A')),
                    (Atom('r', ('A', 'B')), Atom('s', ('B',))))
        query = CQ(('X', 'Y'), (Atom('p', ('X', 'Y')),
                                Atom('u', ('X',)), Atom('v', ('Y',))))
        return COTQueryEngine(facts, [rule]), query

    def test_identified_outputs_keep_both_unary_intersections(self):
        for left, right, expected in (({1, 2}, {2, 3}, {(2, 2)}),
                                      ({1}, {3}, set())):
            engine, query = self.engine_and_query(left, right)
            for mode in MODES:
                with self.subTest(left=left, right=right, mode=mode):
                    self.assertEqual(set(engine.execute(query, mode, count=False)['rows']),
                                     expected)

    def test_qfse_filters_identified_outputs_before_projection(self):
        engine, query = self.engine_and_query({1, 2}, {2, 3})
        _, sql, _ = engine.compile(query, 'p11')
        stages = sql[sql.index('f_g0_c1_s0 AS'):sql.index('f_g0_union AS')]
        self.assertIn('FROM f_uv_qv0 uv WHERE uv.qv0 =', stages)
        self.assertIn('FROM f_uv_qv1 uv WHERE uv.qv1 =', stages)
        self.assertIn('FROM kg_p', stages)

    def test_mapped_variables_and_constant_filters_enter_first_stage(self):
        item = QueryItem(0, (OnlineAtom('r', (Variable('A'), Variable('B'))),),
                         'COT', 'r', ('X', 'Y', 'W'),
                         (('W', Constant(7)),),
                         (('X', Variable('A')), ('Y', Variable('A')),
                          ('W', Constant(7))))
        compiler = CQSqlCompiler({'r': 'r'}, lambda value: int(value))
        ctes, _, stats = compiler._compile_candidate_branch(
            item, group_index=0, candidate_index=0,
            var_aliases={'X': 'x', 'Y': 'y', 'W': 'w'},
            absorbed_unary_views={'X': 'vx', 'Y': 'vy', 'W': 'vw'})
        first = ctes[0]
        self.assertIn('FROM vx uv WHERE uv.x =', first)
        self.assertIn('FROM vy uv WHERE uv.y =', first)
        self.assertIn('FROM vw uv WHERE uv.w = 7', first)
        self.assertEqual(stats['absorbed_unary_filters'], 3)


if __name__ == '__main__':
    unittest.main()
