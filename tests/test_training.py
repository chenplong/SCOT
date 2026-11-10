import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scot.online.policy import choose_mode
from scot.training import COST_COLUMNS, train_model


class TrainingCodeTests(unittest.TestCase):
    def write_input(self, path):
        fields = ['valid_for_training', 'feat_signal', 'feat_missing',
                  'label_measurement_protocol', 'label_cost_schema'] + list(COST_COLUMNS)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for signal in range(4):
                row = dict(valid_for_training=1, feat_signal=signal, feat_missing='',
                           label_measurement_protocol='unit_fixture', label_cost_schema='fixture')
                row.update(dict(zip(COST_COLUMNS, [10, 20, 1, 30])))
                writer.writerow(row)

    def test_saved_tree_loads_with_missing_feature_and_nondefault_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'fixture.csv', Path(directory) / 'model.json'
            self.write_input(source)
            summary = train_model([source], target)
            model = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(summary['rows'], 4)
            self.assertEqual(choose_mode({'feat_signal': 1}, model), 'p10')
            self.assertEqual(choose_mode({'feat_signal': float('nan')}, target), 'p10')
            metadata = model['training_data']
            self.assertEqual(metadata['unavailable_features'], ['feat_missing'])
            self.assertEqual(metadata['source_files'][0]['sha256'],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['label_measurement_protocol_counts'], {'unit_fixture': 4})


if __name__ == '__main__':
    unittest.main()
