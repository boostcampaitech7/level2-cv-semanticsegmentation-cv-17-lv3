import os
import cv2
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
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

def inference(args):
    model = YOLO(args.model_path).cuda()
    infer_images = sorted(glob(args.test_data_path))
    
    rles = []
    filename_and_class = []

    for idx, infer_image in tqdm(enumerate(infer_images)):
        result = model.predict(infer_image, imgsz=2048)[0]
        boxes = result.boxes.data.cpu().numpy()
        scores, classes = boxes[:, 4].tolist(), boxes[:, 5].astype(np.uint8).tolist()
        masks = result.masks.xy
        
        datas = [[a, b, c] for a, b, c, in zip(classes, scores, masks)]
        datas = sorted(datas, key=lambda x: (x[0], -x[1]))
        img_name = infer_image.split("/")[-1]
        
        is_checked = [False] * 29
        csv_idx, data_idx = 0, 0
        while data_idx < len(datas):
            c, s, mask_pts = datas[data_idx]
            # 동일한게 있으면 pass
            if is_checked[c]:
                data_idx += 1
                continue
            
            empty_mask = np.zeros((2048, 2048), dtype=np.uint8)
            if c == csv_idx:
                is_checked[c] = True
                pts = [[int(x[0]), int(x[1])] for x in mask_pts]
                cv2.fillPoly(empty_mask, [np.array(pts)], 1)
                rle = encode_mask_to_rle(empty_mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{img_name}")
                data_idx += 1
            else:
                rle = encode_mask_to_rle(empty_mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[csv_idx]}_{img_name}")
            csv_idx += 1
            
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    if not os.path.exists('./result'):                                                           
        os.makedirs('./result')
    df.to_csv(os.path.join('result', args.file_name), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/data/ephemeral/home/euna/YOLO/runs/segment/train_8x_aug/weights/best.pt", help="model path (.pt)")
    parser.add_argument('--test_data_path', type=str, default="/data/ephemeral/home/euna/data/test/*/*/*.png", help='train data path')
    parser.add_argument('--file_name', type=str, default='yolo_8x_aug.csv', help='result file nale')
    args = parser.parse_args()
    inference(args)