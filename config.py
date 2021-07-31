import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=1e-03)
    parser.add_argument('-i', '--image_size', default=(180, 180))
    parser.add_argument('-bs', '--batch_size', default=32)
    parser.add_argument('-s', '--seed', default=42)
    parser.add_argument('-dr', '--dropout', default=0.5)
    parser.add_argument('-p', '--patience', default=1)
    parser.add_argument('-rf', '--rotation_factor', default=0.1)
    return parser