"""Synthetic repeat differences; no dataset, model or annotation access."""
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('repeat_analysis',Path(__file__).resolve().parents[1]/'tools/analyze_ball_repeatability_117093.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


def point(x,y,role='match',visibility='visible'):return dict(centre=[x,y],role=role,visibility=visibility)


class RepeatAnalysisTests(unittest.TestCase):
    def test_signed_displacement_and_euclidean_norm(self):
        row=A.compare(point(10,10),point(7,14))
        self.assertEqual((row['dx'],row['dy'],row['distance_px']),(-3,4,5))

    def test_unavailable_visibility_or_identity_not_forced(self):
        for b in [None,point('','',visibility='not_visible'),point(1,1,role='uncertain')]:
            self.assertIsNone(A.compare(point(1,1),b)['distance_px'])

    def test_nonfinite_boolean_or_outside_rejected(self):
        for xy in [(float('nan'),1),(True,1),(4096,1),(-1,1),(1,float('inf'))]:
            with self.subTest(xy=xy),self.assertRaises(ValueError):A.compare(point(1,1),point(*xy))

    def test_large_disagreement_and_missing_retained_in_summary(self):
        result=A.describe([dict(distance_px=v) for v in [0,1,2,50,None]])
        self.assertEqual(result['maximum_px'],50)
        self.assertEqual(result['median_px'],1.5)
        self.assertEqual(result['unavailable_pairs'],1)
        self.assertEqual(result['recorded_pairs'],5)

    def test_equal_frame_number_does_not_override_pixel_mismatch(self):
        key=dict(raw_png_sha256='correct',source_frame_0_based=100)
        A.verify_pixel_identity(key,dict(sha256='correct',source_frame_0_based=100),dict(sha256='correct'))
        with self.assertRaises(ValueError):A.verify_pixel_identity(key,dict(sha256='correct',source_frame_0_based=100),dict(sha256='wrong'))


if __name__=='__main__':unittest.main()
