import os
import shutil

import kagglehub


def main() -> None:
    target_path = "assets/datasets/"
    os.makedirs(target_path, exist_ok=True)

    path = kagglehub.dataset_download("anhoangvo/acdc-dataset")

    source_path = "/home/lap15406/.cache/kagglehub/"
    print("Path to datasets files:", path)
    shutil.move(source_path, target_path)
    print("Dataset downloaded and moved to:", target_path)

if __name__ == "__main__":
    main()
