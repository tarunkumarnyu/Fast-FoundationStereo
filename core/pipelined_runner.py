"""Pipelined TRT runner for Fast-FoundationStereo.

Overlaps feature extraction (frame N+1) with post processing (frame N)
using separate CUDA streams. Throughput = max(feat+gwc, post) instead of
feat+gwc+post.
"""

import torch
import tensorrt as trt
from core.submodule import build_gwc_volume_triton


class PipelinedTrtRunner:
    def __init__(self, args, feature_engine_path, post_engine_path):
        self.args = args
        self.max_disp = args.max_disp
        self.cv_group = args.get('cv_group', 8)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load feature engine
        with open(feature_engine_path, 'rb') as f:
            self.feat_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        self.feat_ctx = self.feat_engine.create_execution_context()

        # Load post engine
        with open(post_engine_path, 'rb') as f:
            self.post_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        self.post_ctx = self.post_engine.create_execution_context()

        # Separate CUDA streams for pipeline overlap
        self.stream_feat = torch.cuda.Stream()
        self.stream_post = torch.cuda.Stream()

        # Events for synchronization
        self.feat_done = torch.cuda.Event()
        self.post_done = torch.cuda.Event()

        # Pipeline state: previous frame's result
        self.prev_disp = None
        self.post_pending = False

        # Get post engine input names
        self.post_in_names = []
        for i in range(self.post_engine.num_io_tensors):
            name = self.post_engine.get_tensor_name(i)
            if self.post_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.post_in_names.append(name)

    def _trt_dtype_to_torch(self, dt):
        if dt == trt.DataType.FLOAT: return torch.float32
        if dt == trt.DataType.HALF: return torch.float16
        if dt == trt.DataType.BF16: return torch.bfloat16
        if dt == trt.DataType.INT32: return torch.int32
        if dt == trt.DataType.INT8: return torch.int8
        if dt == trt.DataType.BOOL: return torch.bool
        raise RuntimeError(f"Unsupported TRT dtype: {dt}")

    def _run_engine(self, engine, context, inputs, stream):
        """Run TRT engine on specified CUDA stream."""
        # Ensure inputs are on GPU and cast to expected dtypes
        for name, tensor in list(inputs.items()):
            if not tensor.is_cuda:
                inputs[name] = tensor.cuda()
            expected = self._trt_dtype_to_torch(engine.get_tensor_dtype(name))
            if inputs[name].dtype != expected:
                inputs[name] = inputs[name].to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            context.set_input_shape(name, tuple(inputs[name].shape))

        # Allocate outputs
        outputs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                dtype = self._trt_dtype_to_torch(engine.get_tensor_dtype(name))
                outputs[name] = torch.empty(shape, device='cuda', dtype=dtype)

        # Set tensor addresses
        for name, tensor in inputs.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))

        # Execute on specified stream
        ok = context.execute_async_v3(stream.cuda_stream)
        assert ok, "TRT execution failed"
        return outputs

    def forward_sequential(self, image1, image2):
        """Standard sequential execution (for comparison)."""
        feat_out = self._run_engine(self.feat_engine, self.feat_ctx,
                                     {'left': image1, 'right': image2},
                                     self.stream_feat)
        self.stream_feat.synchronize()

        gwc_volume = build_gwc_volume_triton(
            feat_out['features_left_04'].half(),
            feat_out['features_right_04'].half(),
            self.max_disp // 4, self.cv_group)

        post_inputs = {k: v for k, v in feat_out.items() if k in self.post_in_names}
        post_inputs['gwc_volume'] = gwc_volume

        out = self._run_engine(self.post_engine, self.post_ctx,
                                post_inputs, self.stream_post)
        self.stream_post.synchronize()
        return out['disp']

    def forward_pipelined(self, image1, image2):
        """Pipelined execution: overlap feat(N+1) with post(N).

        Returns disparity from the PREVIOUS frame (1-frame latency).
        First call returns None.
        """
        # If post_runner from previous frame is still running, wait for it
        if self.post_pending:
            self.post_done.synchronize()
            self.post_pending = False

        # D2H copy while no streams are busy
        result = self.prev_disp.cpu() if self.prev_disp is not None else None

        # Stage 1: Feature extraction + GWC on stream_feat
        with torch.cuda.stream(self.stream_feat):
            feat_out = self._run_engine(self.feat_engine, self.feat_ctx,
                                         {'left': image1, 'right': image2},
                                         self.stream_feat)

        # Need feat_out before gwc, so sync stream_feat
        self.stream_feat.synchronize()

        # GWC volume computation (on default stream, fast)
        gwc_volume = build_gwc_volume_triton(
            feat_out['features_left_04'].half(),
            feat_out['features_right_04'].half(),
            self.max_disp // 4, self.cv_group)

        # Prepare post inputs
        post_inputs = {k: v for k, v in feat_out.items() if k in self.post_in_names}
        post_inputs['gwc_volume'] = gwc_volume

        # Stage 2: Post processing on stream_post (non-blocking)
        with torch.cuda.stream(self.stream_post):
            out = self._run_engine(self.post_engine, self.post_ctx,
                                    post_inputs, self.stream_post)

        # Record event so we know when post is done
        self.post_done.record(self.stream_post)
        self.prev_disp = out['disp']
        self.post_pending = True

        return result

    def flush(self):
        """Get the last pending result after pipeline is done."""
        if self.post_pending:
            self.post_done.synchronize()
            self.post_pending = False
        return self.prev_disp
