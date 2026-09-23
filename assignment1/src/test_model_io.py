"""Run: .venv/bin/python -m unittest discover -s src -v"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import model_io
from network import ClassificationNetwork
from training import create_training_model


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original = model_io.SOURCE.read_bytes()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.pth'
        self.model = ClassificationNetwork(device='cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        model_io.save_checkpoint(self.model, self.path, self.optimizer, epoch=7)
        self.checkpoint = model_io.read_checkpoint(self.path)

    def check_source(self, source):
        return model_io.equivalent_source(self.path, self.checkpoint, source)

    def test_equivalent_refactor(self):
        source = self.original.replace(b'batch_size', b'number_of_images')
        source = source.replace(b'extract_sensor_values', b'read_hud')
        source = source.replace(b'x.reshape(number_of_images, -1)', b'torch.flatten(x, start_dim=1)')
        source = source.replace(b'1.1 d)', b'Updated documentation')
        self.assertTrue(self.check_source(source))

    def test_numerically_equivalent_expression(self):
        self.assertTrue(self.check_source(self.original.replace(
            b'x = observation / 255.0', b'x = observation * (1.0 / 255.0)')))

    def test_changed_activation(self):
        self.assertFalse(self.check_source(self.original.replace(b'torch.nn.ReLU()', b'torch.nn.Sigmoid()')))

    def test_changed_gradient_with_same_output(self):
        self.assertFalse(self.check_source(self.original.replace(
            b'return self.fc(x)', b'return self.fc(x).detach()')))

    def test_changed_train_only_behavior(self):
        self.assertFalse(self.check_source(self.original.replace(
            b'return self.fc(x)', b'return self.fc(x) + (1 if self.training else 0)')))

    def test_changed_action_mapping(self):
        self.assertFalse(self.check_source(self.original.replace(
            b'return self.action_classes[cls]', b'return self.action_classes[(cls + 1) % 9]')))

    def test_changed_layer_shape(self):
        self.assertFalse(self.check_source(self.original.replace(
            b'Conv2d(3, 8,', b'Conv2d(3, 16,')))

    def test_zero_weights_do_not_hide_operation_change(self):
        self.checkpoint['state_dict'] = {k: torch.zeros_like(v) for k, v in self.checkpoint['state_dict'].items()}
        torch.save(self.checkpoint, self.path)
        self.assertFalse(self.check_source(self.original.replace(b'torch.nn.ReLU()', b'torch.nn.Sigmoid()')))

    def test_missing_and_tampered_snapshot(self):
        snapshot = self.path.with_suffix('.network.py')
        snapshot.write_bytes(self.original + b'\n# tampered')
        self.assertFalse(self.check_source(self.original))
        snapshot.unlink()
        reasons = []
        self.assertFalse(model_io.equivalent_source(self.path, self.checkpoint, self.original, reasons))
        self.assertIn(str(snapshot), reasons[0])
        self.assertIn('없습니다', reasons[0])

    def test_invalid_source(self):
        reasons = []
        self.assertFalse(model_io.equivalent_source(self.path, self.checkpoint, b'not valid python!', reasons))
        self.assertIn('SyntaxError', reasons[0])

    def test_hash_match_and_strict_weights(self):
        with patch.object(model_io, 'equivalent_source', side_effect=AssertionError('should not probe')):
            model, checkpoint = model_io.load_model(self.path, 'cpu')
        self.assertEqual(checkpoint['epoch'], 7)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value)
        self.checkpoint['state_dict'].pop('fc.0.bias')
        torch.save(self.checkpoint, self.path)
        with self.assertRaises(model_io.IncompatibleModelError):
            model_io.load_model(self.path, 'cpu')

    def test_auto_resumes_equivalent_and_restarts_changed(self):
        current = Path(self.directory.name) / 'network.py'
        real_load = model_io.load_model
        def load(path, device, **kwargs):
            return real_load(path, device, source=current, **kwargs)
        current.write_bytes(self.original.replace(b'1.1 d)', b'New docstring'))
        with patch.object(model_io, 'load_model', side_effect=load):
            resumed = create_training_model(str(self.path), 'cpu', automatic=True)
            self.assertEqual(resumed._resume_checkpoint['epoch'], 7)
            self.assertTrue(resumed._resume_checkpoint['optimizer_state'])
            self.assertEqual(resumed._training_start, str(self.path))
            with self.assertRaises(model_io.IncompatibleModelError):
                create_training_model(str(self.path), 'cpu', automatic=False)
            current.write_bytes(self.original.replace(b'torch.nn.ReLU()', b'torch.nn.Sigmoid()'))
            restarted = create_training_model(str(self.path), 'cpu', automatic=True)
            self.assertTrue(restarted._auto_restart)
            self.assertEqual(restarted._training_start, 'scratch')

    def test_timeout_fails_closed(self):
        import subprocess
        with patch.object(model_io.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 60)):
            reasons = []
            self.assertFalse(model_io.equivalent_source(self.path, self.checkpoint, self.original, reasons))
            self.assertIn('60초', reasons[0])

    def test_supplied_baseline_snapshot_matches_checkpoint(self):
        baseline = Path(__file__).resolve().parents[1] / 'models' / 'baseline.pth'
        checkpoint = model_io.read_checkpoint(baseline)
        saved = baseline.with_suffix('.network.py').read_bytes()
        self.assertEqual(model_io.source_hash(saved), checkpoint['network_sha256'])


if __name__ == '__main__':
    unittest.main()
