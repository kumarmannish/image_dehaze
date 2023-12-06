import numpy
import math
import cv2
import os
import argparse
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-o","--original", required=True, type=str, help="require original file path")
ap.add_argument("-s","--contrast", required=True, type=str, help="require contrast file path")
args = ap.parse_args()

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20*math.log10(pixel_max/math.sqrt(mse))

def main():
    original = cv2.imread(args.original)
    contrast = cv2.imread(args.contrast)

    o_height, o_width, o_channel = original.shape
    contrast = cv2.resize(contrast, dsize=(o_width, o_height), interpolation=args.interpolation)

    print("Image PSNR Mean: {}".format(psnr(original, contrast)))

if __name__ == '__main__':
    main()