import argparse

import PIL
import numpy as np
import os
import chainer
import scipy
from chainer import cuda
from chainer import serializers
from inception_score.inception_score_script import Inception
from inception_score.inception_score_script import compute_inception_score
from inception_score import download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--model', type=str, default='inception_score.model')
    return parser.parse_args()


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def get_image(image_path,
              resize_height=128, resize_width=128,
              ):
    image = imread(image_path)
    return transform(image,
                     resize_height, resize_width)


def transform(image,
              resize_height=128, resize_width=128):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image).reshape((3, 128, 128))


def get_images(arg_data_list):
    # print('getting images')
    return [get_image(filename) for filename in arg_data_list]


def main(arg_data_list='', arg_samples=-1, arg_gpu=2, arg_model='inception_score.model', d=0):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if d == 1:
        # print('in d == 1')
        # print('dir_path is ..', dir_path)

        args_object = download.ArgsObject(downloads_dir=os.path.join(dir_path, 'downloads'),
                                          outfile='inception_score.model')
        download.main(args_object)

    # print('loading model')
    # Load trained model
    model = Inception()
    filename = os.path.join(dir_path, 'downloads', arg_model)
    serializers.load_hdf5(filename, model)
    # print('setting gpu ')
    # print('arg_gpu')
    if arg_gpu is not None and arg_gpu >= 0:
        # print('cuda device')
        cuda.get_device(arg_gpu).use()
        # print('to gpu')
        model.to_gpu()
    # print('converting images to np array')
    # Load images
    # train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
    samples = get_images(arg_data_list)
    # print('done reading images..')
    samples_1 = np.array(samples).astype(np.float32)
    # print('converted the array of images to np array')
    # Use all 60000 images, unless the number of samples are specified
    ims = samples_1
    if arg_samples > 0:
        ims = ims[:arg_samples]
    # print('done converting np array')
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = compute_inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)

    return mean, std


if __name__ == '__main__':
    args = parse_args()
    main(args.model, args.gpu, args.samples)
