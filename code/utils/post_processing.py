import numpy as np
import cv2
import pandas as pd
from copy import deepcopy
import argparse

def decode_rle_to_mask(rle, height, width):
    # csv 파일로 제출한 RLE 결과를 mask map 으로 디코딩 합니다.
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def filter_largest_contour(mask):
    # mask map에서 가장 큰 영역만을 남겨 필터링 합니다.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)
    
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(largest_mask, [hull], -1, 1, thickness=-1)
    filtered_mask = mask * largest_mask
    
    return filtered_mask

def process(csv_path, output_csv_path):
    ori_data = pd.read_csv(csv_path)
    new_data = deepcopy(ori_data)

    for i in range(len(new_data['rle'])):
        before = new_data.loc[i, 'rle']
        # rle -> mask 변환
        pred = decode_rle_to_mask(before, height=2048, width=2048)
        # mask -> filter_mask 변환
        filter_mask = filter_largest_contour(pred)
        # filter_mask -> rle 변환
        after = encode_mask_to_rle(filter_mask)
        new_data.loc[i, 'rle'] = after
    
    new_data.to_csv(output_csv_path, index=False)
    print(f'Success Create new csv file to {output_csv_path}')

if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Apply Negative Sample Masking")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the original csv path")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to the new csv path")

    args = parser.parse_args()

    process(args.csv_path, args.output_csv_path)
