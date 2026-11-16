"""Synthetic failures of association and GPU evidence, no payload/runtime access."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


association = load('v3_ball_association_probe')
benchmark = load('v3_gpu_benchmark')


def ball(x, y=20):
    return dict(center_raw_px=[x, y], positive_box_size=True, best_class='sports ball')


class AssociationChecks(unittest.TestCase):
    def test_candidate_reordering_keeps_proposals_for_moving_and_stationary_balls(self):
        tracker = association.TrackletProposals()
        first = tracker.update('H1', 0, 'tile', [ball(0), ball(100)])['proposals']
        second = tracker.update('H1', 1, 'tile', [ball(100), ball(2)])['proposals']
        self.assertEqual(second[0]['tracklet_id'], first[1]['tracklet_id'])
        self.assertEqual(second[1]['tracklet_id'], first[0]['tracklet_id'])
        self.assertEqual([p['step_raw_px'] for p in second], [0, 2])
        self.assertTrue(all(not p['accepted_as_match_ball'] for p in second))

    def test_gap_period_and_empty_observation_prevent_linking(self):
        for middle, last_period, last_frame in [(None, 'H1', 2), (None, 'H2', 1), ([], 'H1', 2)]:
            tracker = association.TrackletProposals()
            old = tracker.update('H1', 0, 'v', [ball(2)])['proposals'][0]
            if middle is not None:
                tracker.update('H1', 1, 'v', middle)
            new = tracker.update(last_period, last_frame, 'v', [ball(2)])['proposals'][0]
            self.assertNotEqual(old['tracklet_id'], new['tracklet_id'])
            self.assertIsNone(new['step_raw_px'])

    def test_duplicate_centres_and_equal_distance_abstain_instead_of_arbitrary_link(self):
        tracker = association.TrackletProposals()
        tracker.update('H1', 0, 'v', [ball(-1), ball(1)])
        current = tracker.update('H1', 1, 'v', [ball(0)])['proposals']
        self.assertIsNone(current[0]['previous_frame'])
        tracker = association.TrackletProposals()
        tracker.update('H1', 0, 'v', [ball(0)])
        result = tracker.update('H1', 1, 'v', [ball(0), ball(0)])['proposals']
        self.assertTrue(all(p['previous_frame'] is None for p in result))

    def test_missing_nonfinite_and_wrong_class_do_not_make_false_zero_ball(self):
        invalid = [ball(float('nan')), dict(ball(0), best_class='person'),
                   dict(ball(0), positive_box_size=False), {}]
        result = association.TrackletProposals().update('H1', 0, 'v', invalid)
        self.assertEqual(result['proposals'], [])

    def test_views_are_separate_and_duplicate_or_unordered_frame_fails(self):
        tracker = association.TrackletProposals()
        tracker.update('H1', 3, 'v1', [ball(0)])
        other = tracker.update('H1', 4, 'v2', [ball(0)])['proposals'][0]
        self.assertIsNone(other['previous_frame'])
        for frame in (3, 2):
            with self.assertRaises(ValueError):
                tracker.update('H1', frame, 'v1', [ball(0)])

    def test_cuda_registration_and_profile_both_required(self):
        with self.assertRaises(RuntimeError):
            benchmark.require_cuda(['CPUExecutionProvider'])
        benchmark.require_cuda(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        with self.assertRaises(RuntimeError):
            benchmark.profile_providers([dict(cat='Node', args=dict(provider='CPUExecutionProvider'))])
        result = benchmark.profile_providers([dict(cat='Node', args=dict(provider='CUDAExecutionProvider'))])
        self.assertEqual(result, {'CUDAExecutionProvider': 1})

    def test_import_does_not_open_payloads_or_start_processes(self):
        with patch.object(Path, 'open', side_effect=AssertionError('unexpected IO')):
            with patch('subprocess.check_output', side_effect=AssertionError('unexpected process')):
                load('v3_ball_association_probe')
                load('v3_gpu_benchmark')


if __name__ == '__main__':
    unittest.main()
