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

def rle_to_mask(rle, shape):
    """ RLE 문자열을 디코딩하여 마스크 이미지를 반환합니다. """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape)

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

def draw_mask(image, masks, labels, thickness=2):
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = color_map.get(labels[i], (255, 255, 255))  # 클래스에 대응하는 색상
        mask_indices = mask.nonzero()
        overlay[mask_indices] = color

        # 중심에 클래스 이름 표시
        centroid = np.mean(np.column_stack(mask_indices), axis=0).astype(int)
        cv2.putText(
            overlay, labels[i], (centroid[1], centroid[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

    # 투명도 적용
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

    # 총 페이지 수 계산
    total_pages = (len(image_paths) + NUM_IMAGES_PER_PAGE - 1) // NUM_IMAGES_PER_PAGE

    # 현재 페이지 인덱스 설정
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
    page_index = st.session_state["page_index"]

    # 페이지 이동 버튼과 입력란을 같은 줄에 배치
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous Page") and page_index > 0:
            st.session_state["page_index"] -= 1
            page_index = st.session_state["page_index"]
    with col3:
        if st.button("Next Page") and page_index < total_pages - 1:
            st.session_state["page_index"] += 1
            page_index = st.session_state["page_index"]

    # 페이지 설정 입력 창
    new_page = st.number_input("Go to Page", min_value=1, max_value=total_pages, value=page_index + 1, step=1, key="page_input")

    # 페이지 입력이 기존 페이지 인덱스와 다를 경우 업데이트
    if new_page - 1 != page_index:
        st.session_state["page_index"] = new_page - 1
        page_index = new_page - 1

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

    # 현재 페이지 / 전체 페이지 표시
    st.write(f"Page {page_index + 1}/{total_pages}")

def test_viz(data_dir):
    images_per_page = 2
    csv_file = 'sample_submission.csv'
    data = pd.read_csv(csv_file)

    # 이미지 경로 목록 생성
    image_path_list = []
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        image_path_list.extend([os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))])

    # 총 페이지 수 계산
    total_pages = (len(image_path_list) + images_per_page - 1) // images_per_page

    # 현재 페이지 설정
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 1
    page_number = st.session_state["page_number"]

    # 페이지 이동 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Image") and page_number > 1:
            st.session_state["page_number"] -= 1
            page_number = st.session_state["page_number"]
    with col2:
        if st.button("Next Image") and page_number < total_pages:
            st.session_state["page_number"] += 1
            page_number = st.session_state["page_number"]

    # 페이지 설정 입력 창을 아래에 표시
    new_page = st.number_input("Set Page", min_value=1, max_value=total_pages, value=page_number, step=1, key="page_input")
    
    # 만약 입력된 페이지가 현재 세션 상태와 다를 경우 업데이트
    if new_page != page_number:
        st.session_state["page_number"] = new_page
        page_number = new_page

    # 현재 페이지에 표시할 이미지 가져오기
    image_index = (page_number - 1) * images_per_page
    images_on_page = image_path_list[image_index:image_index + images_per_page]
    annotated_images = []

    for current_image_path in images_on_page:
        image = Image.open(current_image_path)
        image = np.array(image.convert("RGB"))

        # 해당 이미지의 마스크와 클래스 정보 로드
        masks = []
        labels = []
        image_name = current_image_path.split('/')[-1]
        for _, row in data[data['image_name'] == image_name].iterrows():
            label = row['class']
            rle = row['rle']
            if type(rle) == float:
                continue
            mask = rle_to_mask(rle, image.shape[:2])
            masks.append(mask)
            labels.append(label)

        # 어노테이션이 적용된 이미지 생성
        annotated_image = draw_mask(image, masks, labels)
        annotated_images.append(annotated_image)

    # 어노테이션 이미지를 한 줄에 나란히 표시
    cols = st.columns(2)
    for i, annotated_image in enumerate(annotated_images):
        with cols[i]:
            st.image(annotated_image, caption=f"Labeled Image: {images_on_page[i]}", use_container_width=True)

    # 현재 페이지 / 전체 페이지 표시
    st.markdown(f"### Page {page_number}/{total_pages}")
    
    # 입력된 페이지 번호에 따라 `page_number` 업데이트
    if new_page != st.session_state["page_number"]:
        st.session_state["page_number"] = new_page

data_dir = st.sidebar.text_input("Data Directory", "data")
mode = st.sidebar.selectbox("Mode", ["train", "test"])

st.write(f"Selected Directory: {data_dir}")
st.write(f"Selected Mode: {mode}")

# 선택한 data_dir과 mode를 사용해 앱의 주요 내용을 실행합니다.
if mode == "test":
    data_dir = os.path.join(data_dir, 'test', 'DCM')
    test_viz(data_dir=data_dir)
elif mode == "train":
    data_dir = os.path.join(data_dir, 'train')
    train_viz(data_dir=data_dir)
