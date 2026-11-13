"""Official StreamTTT model/window helpers with raw JAKE video input.

No visual-feature cache. The only Conv3D adjustment is tensor storage layout.
QA forks/restores all native recurrent fields through the existing audited adapter.
"""
import hashlib
import torch

from baselines.streambench.adapters.streamttt import StreamTTTMethod


class NativeStreamTTT(StreamTTTMethod):
    def __init__(self, config):
        super().__init__(config)
        self.identity['raw_input_protocol'] = 'qwen_vl_utils_smart_resize_bicubic_native_video_v1'
        self.identity['pixels'] = 256 * 32 * 32
        self._capture_options = False
        self._answer_logits = None

    def initialize(self):
        already = self._model is not None
        super().initialize()
        if not already:
            layer = self._model.model.visual.patch_embed.proj
            layer.to(memory_format=torch.channels_last_3d)
            layer.register_forward_pre_hook(
                lambda module, values: (values[0].contiguous(memory_format=torch.channels_last_3d),))

    def _video_inputs(self, frames, timestamps):
        from qwen_vl_utils.vision_process import smart_resize
        from torchvision.transforms import InterpolationMode, functional
        video = torch.stack(frames)
        h, w = smart_resize(video.shape[-2], video.shape[-1], factor=32,
                            min_pixels=self.identity['pixels'], max_pixels=self.identity['pixels'])
        # Match the official fetch_video raw-tensor resize, before normalization.
        video = functional.resize(video, [h, w], interpolation=InterpolationMode.BICUBIC,
                                  antialias=True).float()
        metadata = {'fps': 1.0, 'frames_indices': [float(t['global_timestamp']) for t in timestamps],
                    'total_num_frames': int(float(timestamps[-1]['global_timestamp'])) + 1,
                    'video_backend': 'jake_pts_native'}
        return self._windowing.prepare_video_only_inputs(
            self._processor, [video], [metadata], {'do_sample_frames': False}, 'cpu')

    def _observe_window(self, frames, timestamps):
        inputs = self._video_inputs(frames, timestamps)
        if self._token_offset == 0:
            prefix, _ = self._windowing.build_prompt_ids(self._processor, '', 'cpu')
            inputs['input_ids'] = torch.cat((prefix, inputs['input_ids']), dim=1)
        allowed = {'input_ids', 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts'}
        inputs = {k: v for k, v in inputs.items() if k in allowed}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)
        self._forward(inputs)

    def _forward(self, inputs):
        logits = super()._forward(inputs)
        if self._capture_options:
            self._answer_logits = logits.detach().float().cpu()
        return logits

    def answer(self, question, choices=None):
        self._capture_options = True
        try:
            result = super().answer(question, choices)
            labels = [chr(65+i) for i in range(len(choices))]
            ids = [self._processor.tokenizer(': '+label).input_ids[-1] for label in labels]
            result['raw_option_logits'] = dict(zip(labels, self._answer_logits[0, ids].tolist()))
            return result
        finally:
            self._capture_options = False
            self._answer_logits = None

    def audit_native_tokens(self, frames, timestamps):
        """Official full/split tokenization invariant, using raw temporal pairs."""
        full = self._video_inputs(frames, timestamps)
        a = self._video_inputs(frames[:2], timestamps[:2])
        b = self._video_inputs(frames[2:], timestamps[2:])
        assert torch.equal(full['input_ids'], torch.cat([a['input_ids'], b['input_ids']], dim=1))
        assert torch.equal(full['pixel_values_videos'], torch.cat([a['pixel_values_videos'], b['pixel_values_videos']]))
        return {'status': 'passed', 'native_video_ids_reconstruct': True,
                'native_temporal_patch_pixels_reconstruct': True, 'video_grid_thw': full['video_grid_thw'].tolist()}


def state_digest(value):
    """Smoke-only value audit of all continuation fields; not sampled statistics."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((tuple(x.shape), x.dtype)).encode())
            h.update(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()
