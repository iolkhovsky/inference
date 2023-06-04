import argparse
import numpy as np
import torch
import torchvision

from pl_module import ImageClassifier
from dataset import ToTensor
from utils import (
    DummyProfiler, save_prediction_report, create_graph_image
)


DEFAULT_CHECKPOINT = 'logs/2023-06-0413-11-36-184494/version_0/checkpoints/epoch-0001-loss-0.000000-acc-0.531250.ckpt'


def parse_args():
    parser = argparse.ArgumentParser(prog='Classifier inference test')
    parser.add_argument(
        '--device', default='cpu',
        help='Execution device',
    )
    parser.add_argument(
        '--batch_size', default=1,
        help='Batch size',
    )
    parser.add_argument(
        '--checkpoint', default=DEFAULT_CHECKPOINT,
        help='Checkpoint path',
    )
    parser.add_argument(
        '--iterations', default=100,
        help='Iterations to run',
    )
    parser.add_argument(
        '--report', type=str, default='inference.html',
        help='Path to export report'
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Amount of warmup iterations'
    )
    parser.add_argument(
        '--core_type', type=str, default='VitClassifier',
        help='Amount of warmup iterations'
    )
    return parser.parse_args()


def run_inference(args):
    net = ImageClassifier.load_from_checkpoint(args.checkpoint, core_type=args.core_type)
    net.freeze()

    print(f'Checkpoint:\n{args.checkpoint}')
    print(f'Model:\n{net}')
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    profiler = DummyProfiler('Inference')
    it = iter(loader)
    images, descriptions, colors = [], [], []

    for i in range(args.warmup):
        net(torch.rand(4, 3, 32, 32))

    for idx in range(args.iterations):
        imgs, labels = next(it)
        with profiler:
            scores, classes = net(imgs)
        accuracy = (labels == classes).float().mean()
        print(f'Iteration #{idx} Accuracy: {accuracy}')

        if args.report:
            for img, target, prediction, score in zip(imgs, labels, classes, scores):
                images.append(img.numpy())
                descriptions.append(
                    f'Target: {target}\nPrediction: {prediction}\nScore: {score}'
                )
                colors.append(
                    (196, 255, 196) if target == prediction else (255, 196, 196)
                )

    if args.report:
        profiler_graph = create_graph_image(profiler._sample, (100, 300))
        images.insert(0, profiler_graph)
        descriptions.insert(0, 'Profiler graph')
        colors.insert(0, (192, 192, 192))

        save_prediction_report(
            images=images,
            descriptions=descriptions,
            output_file='inference.html',
            img_size=(128, 128),
            summary=profiler.summary(),
            colors=colors,
        )
   
if __name__ == '__main__':
    run_inference(parse_args())
