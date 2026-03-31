"""Export ONNX with GWC baked into post_runner — eliminates the separate GWC step.

Result: feature_runner.onnx + post_gwc_runner.onnx (includes GWC inside)
At runtime: just two TRT enqueueV3 calls, no custom CUDA kernels.
"""
import warnings, argparse, logging, os, sys
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import omegaconf, yaml, torch, torch.nn as nn, torch.nn.functional as F
from omegaconf import OmegaConf
from core.foundation_stereo import FastFoundationStereo, TrtFeatureRunner, TrtPostRunner
import Utils as U


class PostWithGWC(nn.Module):
    """Wraps GWC + PostRunner into a single traceable module."""

    def __init__(self, post_runner, max_disp, num_groups, normalize=False):
        super().__init__()
        self.post_runner = post_runner
        self.max_disp_4 = max_disp // 4
        self.num_groups = num_groups
        self.normalize = normalize

    def _build_gwc(self, refimg_fea, targetimg_fea):
        """Pure PyTorch GWC — ONNX traceable (no unfold/flip)."""
        B, C, H, W = refimg_fea.shape
        maxdisp = self.max_disp_4
        num_groups = self.num_groups
        cpg = C // num_groups

        # Build shifted target volume via padding + slicing (ONNX-safe)
        # For each disparity d, shift target right by d pixels (pad left, crop right)
        ref = refimg_fea.view(B, num_groups, cpg, H, W)
        volume = torch.zeros(B, num_groups, maxdisp, H, W,
                             device=refimg_fea.device, dtype=refimg_fea.dtype)

        for d in range(maxdisp):
            if d == 0:
                tar_shifted = targetimg_fea
            else:
                # Shift right by d: take target[:, :, :, :-d] and pad d zeros on left
                tar_shifted = F.pad(targetimg_fea[:, :, :, :-d], (d, 0))
            tar = tar_shifted.view(B, num_groups, cpg, H, W)
            volume[:, :, d, :, :] = (ref * tar).sum(dim=2)

        return volume.contiguous()

    def forward(self, features_left_04, features_left_08, features_left_16,
                features_left_32, features_right_04, stem_2x):
        # Build GWC volume from left/right 1/4 features
        gwc_volume = self._build_gwc(features_left_04, features_right_04)
        # Run post processing (pass all features including _16)
        disp = self.post_runner(features_left_04, features_left_08,
                                 features_left_16, features_left_32,
                                 features_right_04, stem_2x, gwc_volume)
        return disp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--valid_iters', type=int, default=4)
    parser.add_argument('--max_disp', type=int, default=128)
    parser.add_argument('--low_memory', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()

    feature_runner = TrtFeatureRunner(model)
    post_runner = TrtPostRunner(model)
    feature_runner.cuda().eval()
    post_runner.cuda().eval()

    H, W = args.height, args.width
    assert H % 32 == 0 and W % 32 == 0

    left_img = torch.randn(1, 3, H, W).cuda().float() * 255
    right_img = torch.randn(1, 3, H, W).cuda().float() * 255

    # 1. Export feature runner (same as before)
    print("Exporting feature_runner.onnx...")
    torch.onnx.export(
        feature_runner,
        (left_img, right_img),
        f'{args.save_path}/feature_runner.onnx',
        opset_version=17,
        input_names=['left', 'right'],
        output_names=['features_left_04', 'features_left_08', 'features_left_16',
                       'features_left_32', 'features_right_04', 'stem_2x'],
        do_constant_folding=True,
        dynamo=False
    )

    # 2. Run feature extraction to get real tensors
    feats = feature_runner(left_img, right_img)
    features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x = feats

    # 3. Create unified post+GWC module
    post_gwc = PostWithGWC(post_runner, args.max_disp, model.cv_group, normalize=False)
    post_gwc.cuda().eval()

    # Test it
    disp_test = post_gwc(features_left_04.float(), features_left_08.float(),
                          features_left_16.float(), features_left_32.float(),
                          features_right_04.float(), stem_2x.float())
    print(f"Test output: {disp_test.shape}, range [{disp_test.min():.1f}, {disp_test.max():.1f}]")

    # 4. Export unified post+GWC
    print("Exporting post_gwc_runner.onnx...")
    torch.onnx.export(
        post_gwc,
        (features_left_04.float(), features_left_08.float(),
         features_left_16.float(), features_left_32.float(),
         features_right_04.float(), stem_2x.float()),
        f'{args.save_path}/post_gwc_runner.onnx',
        opset_version=17,
        input_names=['features_left_04', 'features_left_08', 'features_left_16',
                      'features_left_32', 'features_right_04', 'stem_2x'],
        output_names=['disp'],
        do_constant_folding=True,
        dynamo=False
    )

    # Save config
    cfg = OmegaConf.to_container(model.args)
    cfg['unified_post'] = True  # flag for runtime to know GWC is baked in
    cfg['cv_group'] = model.cv_group
    cfg['image_size'] = [H, W]
    with open(f'{args.save_path}/onnx.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)

    print(f"Done! Engines to build:")
    print(f"  trtexec --onnx={args.save_path}/feature_runner.onnx --saveEngine={args.save_path}/feature_runner.engine --fp16")
    print(f"  trtexec --onnx={args.save_path}/post_gwc_runner.onnx --saveEngine={args.save_path}/post_gwc_runner.engine --fp16")
