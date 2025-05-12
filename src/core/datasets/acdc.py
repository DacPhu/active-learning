import os
import shutil
import tomllib
from typing import Final

import kagglehub

with open("configs/development.toml", "rb") as f:
    CONFIG: Final = tomllib.load(f)


def main() -> None:
    target_path = CONFIG["datasets"]["target"]
    os.makedirs(target_path, exist_ok=True)

    path = kagglehub.dataset_download(CONFIG["datasets"]["kaggle"]["acdc"])

    source_path = CONFIG["datasets"]["kaggle"]["source"]
    print("Path to datasets files:", path)
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        print("Dataset downloaded and moved to:", target_path)
    else:
        print(f"Source path does not exist: {source_path}")


if __name__ == "__main__":
    main()
