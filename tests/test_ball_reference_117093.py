"""Synthetic checks for independent-reference validation; never open match data."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validate_ball_reference_117093 import check_center, provenance, tile_to_raw, validate_frame


def reviewed(**changes):
    row = dict(scene_reviewed='true', used_for_fitting='false', annotator='synthetic',
        model_predictions_seen='false', provider_ball_positions_seen='false',
        all_visible_ball_objects_reviewed='true', match_ball_visibility='visible',
        match_ball_identity='match_confirmed', match_center_x_raw='2000',
        match_center_y_raw='500', center_uncertainty_px='')
    row.update(changes)
    return row


class BallReferenceChecks(unittest.TestCase):
    def test_raw_image_bounds_and_nonfinite(self):
        self.assertEqual(check_center('4095','1079'), (4095.,1079.))
        for x,y in [('4096','0'),('0','1080'),('-1','0'),('nan','0'),('0','inf'),(True,0)]:
            with self.subTest(x=x,y=y), self.assertRaises(ValueError):
                check_center(x,y)

    def test_tile_coordinates_translate_without_resizing(self):
        tile=dict(width=1024,height=540,x_origin=3072,y_origin=540)
        self.assertEqual(tile_to_raw(1023,539,tile), (4095.,1079.))
        with self.assertRaises(ValueError): tile_to_raw(1024,0,tile)

    def test_visible_confirmed_match_needs_coordinates(self):
        self.assertTrue(validate_frame(reviewed())['visible_match_centre'])
        with self.assertRaises(ValueError): validate_frame(reviewed(match_center_x_raw=''))

    def test_ambiguous_identity_cannot_inherit_coordinates(self):
        with self.assertRaises(ValueError): validate_frame(reviewed(match_ball_identity='uncertain'))
        result=validate_frame(reviewed(match_ball_identity='uncertain',match_center_x_raw='',match_center_y_raw=''))
        self.assertFalse(result['visible_match_centre'])

    def test_occluded_match_stays_missing(self):
        with self.assertRaises(ValueError): validate_frame(reviewed(match_ball_visibility='occluded'))
        result=validate_frame(reviewed(match_ball_visibility='occluded',match_center_x_raw='',match_center_y_raw=''))
        self.assertTrue(result['reviewed']);self.assertFalse(result['visible_match_centre'])

    def test_unreviewed_frame_cannot_contain_a_guess(self):
        with self.assertRaises(ValueError): validate_frame(reviewed(scene_reviewed=''))

    def test_unknown_exposure_remains_pending(self):
        self.assertFalse(provenance(reviewed(model_predictions_seen='')))
        self.assertTrue(provenance(reviewed(model_predictions_seen='true')))
        with self.assertRaises(ValueError): provenance(reviewed(model_predictions_seen='no'))

    def test_invalid_fitting_flag_and_uncertainty(self):
        with self.assertRaises(ValueError): validate_frame(reviewed(used_for_fitting='true'))
        for value in ('0','-1','nan','inf'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_frame(reviewed(center_uncertainty_px=value))


if __name__ == '__main__':
    unittest.main()
