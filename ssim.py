from skimage.metrics import structural_similarity
import argparse
import cv2
import os
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-o","--original", required=True, type=str, help="require original image")
ap.add_argument("-s","--contrast", required=True, type=str, help="require contrast image")
args = ap.parse_args()

def main():
    original = cv2.imread(args.original)
    contrast = cv2.imread(args.contrast)

    o_height, o_width, o_channel = original.shape
    contrast = cv2.resize(contrast, dsize=(o_width, o_height), interpolation=args.interpolation)

    o_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(o_gray, c_gray, full = True)
    (diff*255).astype('uint8')
    print("Image SSIM Mean: {}".format(score))

if __name__ == '__main__':
    main()