import os
import os.path as osp
from pathlib import Path
import argparse

def rename_files_in_folders(base_dir, extension):
    # `base_dir` 내의 모든 서브 폴더 찾기
    folder_paths = [f for f in Path(base_dir).iterdir() if f.is_dir()]

    for folder in folder_paths:
        # 폴더 내 PNG 파일만 필터링
        png_files = sorted(folder.glob(f"*.{extension}"))

        if len(png_files) >= 2:
            # 첫 번째 파일 이름 수정
            first_file = png_files[0]
            new_first_name = first_file.stem + "_R" + first_file.suffix
            first_file.rename(folder / new_first_name)

            # 두 번째 파일 이름 수정
            second_file = png_files[1]
            new_second_name = second_file.stem + "_L" + second_file.suffix
            second_file.rename(folder / new_second_name)
        
        else:
            print(f"Skipping folder {folder}, not enough {extension.upper()} files.")

    print(f'Success Rename Files in Folder {base_dir}')

if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Rename files in folders")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to the JSON directory")

    args = parser.parse_args()

    # 이미지 및 JSON 디렉토리에 대해 함수 실행
    rename_files_in_folders(args.image_dir, 'png')
    rename_files_in_folders(args.json_dir, 'json')

