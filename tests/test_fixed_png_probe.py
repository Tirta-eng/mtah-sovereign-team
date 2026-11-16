"""Synthetic identity/missingness tests, with no dataset/model access."""
import copy
import importlib.util
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'tools/v3_fixed_png_probe.py'
SPEC=importlib.util.spec_from_file_location('fixed_png_probe',SOURCE)
PROBE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def candidate(x,y,query=0):
    return dict(center_raw_px=[x,y],query=query,best_class='sports ball',positive_box_size=True)


class FixedPngTests(unittest.TestCase):
    def test_half_open_crop_boundary(self):
        self.assertTrue(PROBE.in_crop(1024,540,(1024,540,1024,540)))
        self.assertFalse(PROBE.in_crop(2048,600,(1024,540,1024,540)))
        self.assertFalse(PROBE.in_crop(1100,1080,(1024,540,1024,540)))

    def test_nonfinite_boolean_or_wrong_class_excluded(self):
        for c in [candidate(float('nan'),1),candidate(True,1),
                  dict(candidate(1,1),best_class='person'),dict(candidate(1,1),positive_box_size=False)]:
            self.assertFalse(PROBE.eligible(c))

    def test_empty_proposals_stay_missing(self):
        row=PROBE.distances_to_reference([],dict(centre_raw_px=[1,1]),(0,0,10,10))
        self.assertIsNone(row['minimum_raw_px'])
        self.assertFalse(row['accepted_match'])

    def test_exact_ties_preserved_and_not_accepted(self):
        row=PROBE.distances_to_reference([candidate(1,2,5),candidate(3,2,8)],
            dict(centre_raw_px=[2,2]),(0,0,10,10))
        self.assertEqual(row['minimum_raw_px'],1)
        self.assertEqual(row['minimising_queries'],[5,8])
        self.assertFalse(row['accepted_match'])

    def test_reference_outside_crop_not_evaluated(self):
        self.assertIsNone(PROBE.distances_to_reference([candidate(10,1)],
            dict(centre_raw_px=[10,1]),(0,0,10,10)))

    def test_not_visible_match_excluded_but_spare_retained(self):
        frame=dict(image_id='synthetic',match_ball_visibility='not_visible',match_ball_identity='not_assessable',
            match_center_x_raw='',match_center_y_raw='')
        other=dict(image_id='synthetic',object_id='spare',role='spare',center_x_raw='3',center_y_raw='4')
        refs=PROBE.reference_points(frame,[other])
        self.assertEqual(refs,[dict(object_id='spare',role='spare',centre_raw_px=[3.,4.])])

    def test_frame_or_tensor_mismatch_rejected(self):
        image=dict(image_id='synthetic',source_frame_0_based=100,raw_png_sha256='png',
            views=[dict(view='full',tensor_sha256='tensor',crop_raw_xywh=[0,0,10,10])])
        good=dict(image_id='synthetic',source_frame_0_based=100,raw_png_sha256='png',view='full',
            tensor_sha256='tensor',crop_raw_xywh=[0,0,10,10])
        PROBE.check_binding(good,image)
        for key,value in [('source_frame_0_based',76),('raw_png_sha256','other'),('tensor_sha256','other')]:
            bad=copy.deepcopy(good)
            bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):PROBE.check_binding(bad,image)


if __name__=='__main__':
    unittest.main()
