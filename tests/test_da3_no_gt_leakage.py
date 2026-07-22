import inspect

from ego_video_camera import da3_adapter


def test_da3_inference_call_does_not_pass_gt_camera_parameters():
    source = inspect.getsource(da3_adapter._make_ray_streaming_class)
    inference_call = source[source.index("self.model.inference") : source.index("predictions.depth")]
    assert "extrinsics=" not in inference_call
    assert "intrinsics=" not in inference_call
    assert "align_to_input_ext_scale" not in inference_call
    assert "use_ray_pose=True" in inference_call
