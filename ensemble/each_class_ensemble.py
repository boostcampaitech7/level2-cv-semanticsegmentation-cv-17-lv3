import pandas as pd
from tqdm import tqdm

classes = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

csv_paths = {
    "finger-1": "./path/to/.csv",
    "finger-2": "./path/to/.csv",
    "finger-3": "./path/to/.csv",
    "finger-4": "./path/to/.csv",
    "finger-5": "./path/to/.csv",
    "finger-6": "./path/to/.csv",
    "finger-7": "./path/to/.csv",
    "finger-8": "./path/to/.csv",
    "finger-9": "./path/to/.csv",
    "finger-10": "./path/to/.csv",
    "finger-11": "./path/to/.csv",
    "finger-12": "./path/to/.csv",
    "finger-13": "./path/to/.csv",
    "finger-14": "./path/to/.csv",
    "finger-15": "./path/to/.csv",
    "finger-16": "./path/to/.csv",
    "finger-17": "./path/to/.csv",
    "finger-18": "./path/to/.csv",
    "finger-19": "./path/to/.csv",
    "Trapezium": "./path/to/.csv",
    "Trapezoid": "./path/to/.csv",
    "Capitate": "./path/to/.csv",
    "Hamate": "./path/to/.csv",
    "Scaphoid": "./path/to/.csv",
    "Lunate": "./path/to/.csv",
    "Triquetrum": "./path/to/.csv",
    "Pisiform": "./path/to/.csv",
    "Radius": "./path/to/.csv",
    "Ulna": "./path/to/.csv"
}

# 결과 파일
result_csv = "merged_data.csv"

base_df = pd.read_csv(csv_paths['finger-1'])
unique_images = base_df['image_name'].unique()

merged_data = []


for image in tqdm(unique_images, desc="Merging data for unique images"):
    
    for cls in classes:
        try:
            df = pd.read_csv(csv_paths[cls])

            row = df[(df['image_name'] == image) & (df['class'] == cls)]

            if not row.empty:
                merged_data.append(row)
                
        except Exception as e:
            print(f"Error processing {image} for class {cls}: {e}")

if merged_data:
    final_df = pd.concat(merged_data, ignore_index=True)

    final_df.to_csv(result_csv, index=False)

    print(final_df.head())
    print("\nTotal rows:", len(final_df))

else:
    print("No data was merged.")