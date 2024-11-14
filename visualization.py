import streamlit as st
import os
import json
import cv2
import numpy as np
from PIL import Image
import pandas as pd

NUM_CLASS = 29
NUM_IMAGES_PER_PAGE = 8
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# 고정된 색상을 정의합니다. 각 클래스가 고유의 색상을 갖도록 설정합니다.
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 128, 128), (128, 255, 128), (128, 128, 255), (192, 192, 192), (128, 128, 128),
    (64, 64, 64), (255, 192, 203), (255, 165, 0), (173, 255, 47), (75, 0, 130), (60, 179, 113)
]

color_map = dict(zip(CLASSES, colors))

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def draw_ann(image, polygons, labels, thickness=2):
    overlay = image.copy()
    for i, polygon_data in enumerate(polygons):
        points = np.array(polygon_data, np.int32).reshape((-1, 1, 2))
        color = color_map.get(labels[i], (255, 255, 255))
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [points], color)

        # 폴리곤 중심에 흰색 텍스트로 클래스 이름 표시
        centroid = np.mean(points, axis=0).astype(int).flatten()  # 중심 좌표 계산
        cv2.putText(
            overlay, labels[i], (centroid[0], centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

    # 투명도 적용 (alpha 값으로 조절)
    alpha = 0.6
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result

def train_viz(data_dir):
    image_dir = os.path.join(data_dir, 'DCM')
    json_dir = os.path.join(data_dir, 'outputs_json')

    image_paths = []
    for folder in sorted(os.listdir(image_dir)):
        folder_path = os.path.join(image_dir, folder)
        image_paths.extend([os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))])

    # 페이지 인덱스 관리
    page_index = st.session_state.get("page_index", 0)
    total_pages = (len(image_paths) + NUM_IMAGES_PER_PAGE - 1) // NUM_IMAGES_PER_PAGE

    # 페이지 이동 버튼과 입력란을 같은 줄에 배치
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous Page") and page_index > 0:
            page_index -= 1
    with col3:
        if st.button("Next Page") and page_index < total_pages - 1:
            page_index += 1

    st.session_state["page_index"] = page_index

    # 현재 페이지의 이미지 가져오기
    start_idx = page_index * NUM_IMAGES_PER_PAGE
    end_idx = start_idx + NUM_IMAGES_PER_PAGE
    images_on_page = image_paths[start_idx:end_idx]

    # 이미지 출력 (4개씩 2줄로 나란히)
    for idx, image_path in enumerate(images_on_page):
        # 이미지 로드
        image = load_image(image_path)
        
        # JSON 파일에서 polygon 정보 불러오기
        json_path = image_path.replace(image_dir, json_dir).replace('.png', '.json')
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                polygon_data = json.load(f)
            labels = [annotation['label'] for annotation in polygon_data['annotations']]
            polygons = [annotation['points'] for annotation in polygon_data['annotations']]
            annotated_image = draw_ann(image, polygons=polygons, labels=labels)
        else:
            annotated_image = image

        # Streamlit에서 4개씩 2줄로 출력
        col = idx % 4
        if col == 0:
            row_container = st.container()
            with row_container:
                row_columns = st.columns(4)
        row_columns[col].image(annotated_image, caption=f'{image_path}', use_container_width=True)

    # 페이지 정보 출력
    st.write(f"Page {page_index + 1}/{total_pages}")

    goto_page = st.number_input("Go to Page", min_value=1, max_value=total_pages, value=page_index + 1, step=1)
    page_index = int(goto_page) - 1  # 입력된 페이지 번호에 따라 페이지 인덱스 변경

data_dir = st.sidebar.text_input("Data Directory", "data")
mode = st.sidebar.selectbox("Mode", ["train", "test"])

st.write(f"Selected Directory: {data_dir}")
st.write(f"Selected Mode: {mode}")

# 선택한 data_dir과 mode를 사용해 앱의 주요 내용을 실행합니다.
if mode == "test":
    data_dir = os.path.join(data_dir, 'test', 'DCM')
    # test_viz(data_dir=data_dir) 후에 구현 예정
elif mode == "train":
    data_dir = os.path.join(data_dir, 'train')
    train_viz(data_dir=data_dir)
