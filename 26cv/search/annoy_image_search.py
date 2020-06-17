# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
from scipy.misc import imread


# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)


def build_ann(names, matrix, ann_path="img_feature_list.ann", dim=2048, n_trees=10):
    ann = AnnoyIndex(dim)
    count = 0
    for name, vec in zip(names, matrix):
        ann.add_item(count, vec)
        count += 1
    ann.build(n_trees)
    ann.save(ann_path)
    print("save ann to %s done" % ann_path)
    return ann


def load_ann(ann_path="img_feature_list.ann", dim=2048):
    ann = AnnoyIndex(dim)
    ann.load(ann_path)
    return ann


class Matcher(object):
    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.ann = build_ann(self.names, self.matrix)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def match(self, image_path, topN=5):
        vector = extract_features(image_path)
        # getting top 5 records
        nearest_ids, distances = self.ann.get_nns_by_vector(vector, topN, include_distances=True)
        print(nearest_ids, distances)
        nearest_img_paths = [self.names.tolist()[i] for i in nearest_ids]
        return nearest_img_paths, distances


def show_img(path):
    img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()


def run():
    images_path = 'data/images/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images
    sample = random.sample(files, 3)
    batch_extractor(images_path)
    ma = Matcher('features.pck')
    for s in sample:
        print('Query image: file_path:%s' % s)
        show_img(s)
        names, distances = ma.match(s, topN=3)
        print('Result images ========================================')
        for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            file_path = os.path.join(images_path, names[i])
            print('Match %s, image file path:%s' % (1 - distances[i], file_path))
            show_img(file_path)


if __name__ == '__main__':
    run()
