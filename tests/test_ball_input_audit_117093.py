"""Synthetic array/path checks; never opens dataset payloads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'tools/audit_ball_reference_inputs_117093.py'
SPEC = importlib.util.spec_from_file_location('input_audit', SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class InputAuditTests(unittest.TestCase):
    def test_equal_values_independent_of_memory_layout(self):
        a = np.arange(12, dtype=np.float32).reshape(3,4)
        report = AUDIT.compare_arrays(a, np.asfortranarray(a))
        self.assertTrue(report['exact_equal'])
        self.assertEqual(report['a_sha256'], report['b_sha256'])

    def test_uint8_difference_does_not_wrap(self):
        report = AUDIT.compare_arrays(np.array([0],np.uint8),np.array([255],np.uint8))
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['maximum_absolute_difference'],255)
        self.assertEqual(report['differing_values'],1)

    def test_channel_swap_fails_exact_equality(self):
        a = np.array([[[1,2,3]]],dtype=np.uint8)
        self.assertFalse(AUDIT.compare_arrays(a,a[:,:,::-1])['exact_equal'])

    def test_dtype_shape_or_nonfinite_rejected(self):
        for a,b in [(np.zeros(2,np.uint8),np.zeros(2,np.float32)),
                    (np.zeros(2),np.zeros(3)),
                    (np.array([float('nan')]),np.array([0.])),
                    (np.array([float('inf')]),np.array([0.]))]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):
                AUDIT.compare_arrays(a,b)

    def test_path_traversal_cannot_read_outside_base(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'allowed'
            base.mkdir()
            (Path(temp)/'outside.txt').write_text('synthetic')
            with self.assertRaises(ValueError):
                AUDIT.contained(base,'../outside.txt')


if __name__ == '__main__':
    unittest.main()
