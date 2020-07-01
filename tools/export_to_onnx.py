# export model to onnx
""""""


import os
import re
import cv2
import argparse
import functools
import subprocess
import warnings
from scipy.special import softmax
import moviepy.editor as mpy
import numpy as np
import torch

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmaction.models import build_recognizer
from mmaction.datasets.transforms import GroupImageTransform

from eval_print.src.eval_print import eval_print
# REPRODUCIBILITY
torch.manual_seed(0)

class ThinWrapper(torch.nn.Module):
    def __init__(self, recognizer):
        super(ThinWrapper, self).__init__()
        self.recognizer = recognizer
        # self.conv1 = torch.nn.Conv3d(8,3,3)


    def forward(self, img_group_0:torch.Tensor):
        img_group = img_group_0

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.recognizer.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.recognizer.extract_feat(img_group)

        if self.recognizer.necks is not None:
            x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
            x, _ = self.recognizer.necks(x)
            x = x.squeeze(2)
            num_seg = 1

        if self.recognizer.with_spatial_temporal_module:
            x = self.recognizer.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + x.shape[1:])
        if self.recognizer.with_segmental_consensus:
            x = self.recognizer.segmental_consensus(x)
            x = x.squeeze(1)
        if self.recognizer.with_cls_head:
            x = self.recognizer.cls_head(x)
        return x


    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        if not self.fcn_testing:
            # 1crop * 1clip 
            assert num_modalities == 1
            img_group = kwargs['img_group_0']

            bs = img_group.shape[0]
            img_group = img_group.reshape(
                (-1, self.in_channels) + img_group.shape[3:])
            num_seg = img_group.shape[0] // bs

            x = self.extract_feat(img_group)

            if self.necks is not None:
                x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
                x, _ = self.necks(x)
                x = x.squeeze(2)
                num_seg = 1

            if self.with_spatial_temporal_module:
                x = self.spatial_temporal_module(x)
            x = x.reshape((-1, num_seg) + x.shape[1:])
            if self.with_segmental_consensus:
                x = self.segmental_consensus(x)
                x = x.squeeze(1)
            if self.with_cls_head:
                x = self.cls_head(x)
            if self.onnx_compatible:
                return x
            else:
                return x.cpu().numpy()
        else:
            # fcn testing
            assert num_modalities == 1
            img_group = kwargs['img_group_0']

            bs = img_group.shape[0]
            img_group = img_group.reshape(
                (-1, self.in_channels) + img_group.shape[3:])
            # standard protocol i.e. 3 crops * 2 clips
            num_seg = self.backbone.nsegments * 2
            # 3 crops to cover full resolution
            num_crops = 3
            img_group = img_group.reshape((num_crops, num_seg) + img_group.shape[1:])

            x1 = img_group[:, ::2, :, :, :]
            x2 = img_group[:, 1::2, :, :, :]
            img_group = torch.cat([x1, x2], 0)
            num_seg = num_seg // 2
            num_clips = img_group.shape[0]
            img_group = img_group.view(num_clips * num_seg, img_group.shape[2], img_group.shape[3], img_group.shape[4])

            if self.flip:
                img_group = self.extract_feat(torch.flip(img_group, [-1]))
            x = self.extract_feat(img_group)
            if self.necks is not None:
                x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
                x, _ = self.necks(x)
            else:
                x = x.reshape((-1, num_seg) + x.shape[1:]).transpose(1, 2)
            x = self.cls_head(x)

            prob = torch.nn.functional.softmax(x.mean([2, 3, 4]), 1).mean(0, keepdim=True).detach()

            if self.onnx_compatible:
                return prob
            else:
                return prob.cpu().numpy()

        return result


def init_recognizer(config, checkpoint=None, label_file=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.backbone.pretrained = None
    config.model.spatial_temporal_module.spatial_size = 8
    model = build_recognizer(
        config.model, train_cfg=None, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if label_file is not None:
            classes = [line.rstrip() for line in open(label_file, 'r').readlines()]
            model.CLASSES = classes
        else:
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use something-something-v2 classes by default.')
                model.CLASSES = get_classes('something=something-v2')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def main():
    """main func."""
    # options
    parser = argparse.ArgumentParser(description="export model to onnx format")
    parser.add_argument('config', type=str, default=None, help='model init config')
    parser.add_argument('checkpoint', type=str, default=None)
    parser.add_argument('save_onnx_as', type=str, default=None)
    parser.add_argument('--label_file', type=str, default='demo/category.txt')
    parser.add_argument('--device', type=str, default='cuda:0', help='you can set "cpu"')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_onnx_as), exist_ok=True)

    # init model
    model = init_recognizer(args.config, checkpoint=args.checkpoint, label_file=args.label_file, device=args.device,)
    model = ThinWrapper(model)

    # inp
    from tools.deploy_onnx import load_frames, extract_frames
    from easydict import EasyDict as edict


    if 0:
        args.video_file='demo/demo.mp4'
        args.frame_folder=None
    else:
        args.frame_folder='frames'

    args.label_file='demo/category.txt'
    args.rendered_output='demo/demo_pred_onnx.mp4'

    cfg = edict()
    cfg.data = edict()
    cfg.data.test = edict()
    cfg.data.test.input_size = 256
    cfg.data.test.img_scale = 256

    if 1:
        # prepare category names
        classes = [line.rstrip() for line in open(args.label_file, 'r').readlines()]

        # Obtain video frames
        if args.frame_folder is not None:
            print('Loading frames in {}'.format(args.frame_folder))
            import glob

            # Here, make sure after sorting the frame paths have the correct temporal order
            frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
            seg_frames, raw_frames = load_frames(frame_paths)
            fps = 4
        else:
            print('Extracting frames using ffmpeg...')
            seg_frames, raw_frames, fps = extract_frames(args.video_file, 8)

        # build the data pipeline
        test_transform = GroupImageTransform(
            crop_size=cfg.data.test.input_size,
            oversample=None,
            resize_crop=False,
            **dict(mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375], to_rgb=True))
        # prepare data
        frames, *l = test_transform(
            seg_frames, (cfg.data.test.img_scale, cfg.data.test.img_scale),
            crop_history=None,
            flip=False,
            keep_ratio=False,
            div_255=False,
            is_flow=False)

        # import ipdb; ipdb.set_trace()
        channels = 3
        length = 8
        height = 256
        width = 256

        # print("frames.shape")
        # print(frames.shape)
        # print(frames[0,0,::16,::16])
        inp = torch.tensor(frames).unsqueeze(0)
        eval_print("inp.size()")
        outputs = model(inp)
        # import ipdb; ipdb.set_trace()

        print(len(outputs))
        # outputs = outputs[0]
        if not isinstance(outputs, tuple):
            outputs = [outputs]

        out = [o.detach().cpu().numpy() for o in outputs]
        for o in out:
            print(o.shape)
            # print(o[0,0,:16,:16])

        import pickle as pkl
        pkl.dump(out, open("a.pkl", 'wb'))
        # np.save("a.npy", out)

    # export as onnx
    input_names = [ "input" ]
    output_names = [ "output" ]

    batch_size = 1
    length = 8
    channels = 3
    height = 256
    width = 256

    dummy_input = torch.randn(batch_size, length, channels, height, width, device=args.device)
    eval_print("dummy_input.shape")
    torch.onnx.export(model, dummy_input, args.save_onnx_as,
                      verbose=True, 
                      input_names=input_names, 
                      output_names=output_names)
    print("save_done")


if __name__=="__main__":
    main()
