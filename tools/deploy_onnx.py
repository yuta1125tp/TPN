import time
import os
import caffe2.python.onnx.backend as backend
import numpy as np
import onnx
import onnxruntime as ort
from easydict import EasyDict as edict

from test_video import load_frames, extract_frames
from mmaction.datasets.transforms import GroupImageTransform


from scipy.special import softmax


def test2(args):
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

    frames = frames[None,::]  # [1,3,8,256,256]

    print(f"load onnxfile: {args.onnx_model_file}")
    ort_session = ort.InferenceSession(args.onnx_model_file)

    results = ort_session.run(None, {'input': frames})

    print("len(results)")
    print(len(results))

    print(results[0].shape)

    import pickle as pkl
    outputs = pkl.load(open("a.pkl", 'rb'))
    print("r: onnx, o: pytorch")
    for i, (r, o) in enumerate(zip(results, outputs)):
        print("i: {}".format(i))
        print("shape r vs o: {}, {}".format(r.shape, o.shape))
        print("mean r vs o: {}, {}".format(r.mean(), o.mean()))
        print("max r vs o: {}, {}".format(r.max(), o.max()))
        print("min r vs o: {}, {}".format(r.min(), o.min()))
        print("all close: {}".format(np.allclose(r,o)))
        print("diff abs max: {}".format(np.max(np.abs(r-o))))


    result = results[0]

    prob = softmax(result.squeeze())
    idx = np.argsort(-prob)

    # Output the prediction.
    video_name = args.frame_folder if args.frame_folder is not None else args.video_file
    print('RESULT ON ' + video_name)
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(prob[idx[i]], classes[idx[i]]))

    # # Render output frames with prediction text.
    # if args.rendered_output is not None:
    #     prediction = model.CLASSES[idx[0]]
    #     rendered_frames = render_frames(raw_frames, prediction)
    #     clip = mpy.ImageSequenceClip(rendered_frames, fps=fps)
    #     clip.write_videofile(args.rendered_output)


def test1(args):
    # Load the ONNX model
    model = onnx.load(args.onnx_model_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)


    channels=3
    length = 8
    height = 256
    width = 256

    dummy_input = np.random.randn(channels, length, height, width).astype(np.float32)

    # rep = backend.prepare(model, device=args.device.upper()) # or "CUDA:0" or "CPU"
    # # For the Caffe2 backend:
    # #     rep.predict_net is the Caffe2 protobuf for the network
    # #     rep.workspace is the Caffe2 workspace for the network
    # #       (see the class caffe2.python.onnx.backend.Workspace)
    # outputs = rep.run(dummy_input)
    # # To run networks with more than one input, pass a tuple
    # # rather than a single numpy ndarray.
    # print(outputs[0])


    ort_session = ort.InferenceSession(args.onnx_model_file)


    num_trials=20
    num_warmup=5
    for i in range(num_warmup+num_trials):
        if i==num_warmup:
            tic=time.time()
        outputs = ort_session.run(None, {'input': dummy_input})
    dtime = time.time()-tic
    print("{}[sec]/{}[iters] -> {}[sec/iters]".format(dtime, num_trials, dtime/num_warmup))
    print(outputs[0])

if __name__=="__main__":
    args = edict()
    args.mode = 'test2'
    args.onnx_model_file = "chkp/onnx/sthv2_tpn.onnx"
    args.onnx_model_file = "foo/fooo2.onnx"
    args.device='cpu'

    if args.mode == 'test1':
        test1(args)
    elif args.mode=='test2':
        test2(args)