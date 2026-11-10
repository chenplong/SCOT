from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.io import parse_query
from scot.workloads import equivalent_queries, filter_training_queries


class WorkloadGuardTests(unittest.TestCase):
    def test_renaming_and_atom_order_preserve_equivalence(self):
        self.assertTrue(equivalent_queries(parse_query('X :- p(X,Y),q(Y).'),
                                           parse_query('A :- q(B),p(A,B).')))

    def test_constants_answers_and_variable_identifications_are_preserved(self):
        pairs = [('X,Y :- p(X,Y).', 'Y,X :- p(X,Y).'),
                 ('X :- p(X,X).', 'X :- p(X,Y).'),
                 ('X :- p(X,1).', 'X :- p(X,"1").')]
        for first, second in pairs:
            self.assertFalse(equivalent_queries(parse_query(first), parse_query(second)))

    def test_filter_does_not_mutate_queries_or_holdout(self):
        source = [('leak', parse_query('X :- p(X,Y),q(Y).')),
                  ('keep', parse_query('X :- r(X).')),
                  ('duplicate', parse_query('A :- r(A).'))]
        test = [('test', parse_query('A :- q(B),p(A,B).'))]
        retained, excluded = filter_training_queries(source, test)
        self.assertEqual([qid for qid, _ in retained], ['keep'])
        self.assertEqual([row['reason'] for row in excluded],
                         ['holdout_equivalent', 'training_duplicate'])
        self.assertEqual(len(source), 3)
        self.assertEqual(len(test), 1)

    def test_long_connected_query_matches_without_whole_body_permutations(self):
        first = parse_query('X0 :- ' + ','.join('p(X%d,X%d)' % (i, i + 1) for i in range(12)) + '.')
        second = parse_query('A0 :- ' + ','.join('p(A%d,A%d)' % (i, i + 1) for i in reversed(range(12))) + '.')
        self.assertTrue(equivalent_queries(first, second))


if __name__ == '__main__':
    unittest.main()
