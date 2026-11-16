"""Synthetic coordinate/preprocessing checks; no dataset/model/network access."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

SOURCE=Path(__file__).resolve().parents[1]/'tools/v3_visual_probe.py'
spec=importlib.util.spec_from_file_location('probe',SOURCE)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class VisualProbeChecks(unittest.TestCase):
    def test_processor_preserves_rgb_and_rescale_without_imagenet_normalization(self):
        config=dict(size={'height':640,'width':640},do_normalize=False,do_pad=False,
                    resample=2,do_rescale=True,do_resize=True,rescale_factor=1/255)
        tensor=probe.preprocess(Image.new('RGB',(1280,320),(255,128,0)),config)
        self.assertEqual(tensor.shape,(1,3,640,640))
        np.testing.assert_allclose(tensor[0,:,0,0],[1,128/255,0],atol=1e-7)
        self.assertEqual(tensor.dtype,np.float32)
        with self.assertRaises(ValueError):
            probe.preprocess(Image.new('RGB',(1,1)),dict(config,do_normalize=True))

    def test_boxes_use_crop_offset_and_keep_best_class_confusion(self):
        logits=np.array([[[0.,5.],[1.,-2.]]]);boxes=np.array([[[.5,.5,.2,.4],[.5,.5,.2,.4]]])
        result=probe.candidate_boxes(logits,boxes,0,(100,200,1000,500),{'0':'sports ball','1':'person'})
        np.testing.assert_allclose(result[0]['box_raw_xyxy'],[500,350,700,550])
        self.assertEqual(result[1]['best_class'],'person')
        self.assertTrue(all(not r['accepted_as_ball'] for r in result))

    def test_invalid_outputs_fail_instead_of_becoming_false_observations(self):
        with self.assertRaises(ValueError):
            probe.candidate_boxes(np.array([[[np.nan]]]),np.zeros((1,1,4)),0,(0,0,10,10),{'0':'sports ball'})

    def test_fisheye_inverse_recovers_known_world_including_outside_pitch(self):
        k=np.array([[1000.,0,500],[0,1000,300],[0,0,1]])
        d=np.array([.01,-.001,.0001,0]);h=np.array([[10.,0,100],[0,10,200],[0,0,1]])
        for world in ([20.,30.],[-10.,80.],[50.,10.]):
            und=h@np.r_[world,1.];ray=np.linalg.solve(k,und);ray=ray[:2]/ray[2]
            radius=np.linalg.norm(ray);theta=np.arctan(radius)
            distorted=theta*(1+d[0]*theta**2+d[1]*theta**4+d[2]*theta**6+d[3]*theta**8)
            pixel=k@np.r_[ray*(distorted/radius),1.]
            actual=probe.ground_plane_proxy(pixel[:2],k,d,k,h)
            np.testing.assert_allclose(actual,world,atol=1e-7)

    def test_invalid_geometry_is_missing_not_zero_distance(self):
        eye=np.eye(3)
        self.assertIsNone(probe.ground_plane_proxy([1,2],eye,np.zeros(4),eye,np.zeros((3,3))))
        self.assertIsNone(probe.ground_plane_proxy([np.nan,2],eye,np.zeros(4),eye,eye))

    def test_import_has_no_dataset_model_or_subprocess_access(self):
        module=importlib.util.module_from_spec(spec)
        with patch.object(Path,'open',side_effect=AssertionError('unexpected dataset open')):
            with patch('subprocess.check_output',side_effect=AssertionError('unexpected process')):
                spec.loader.exec_module(module)


if __name__=='__main__':
    unittest.main()
