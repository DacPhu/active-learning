import kagglehub

# Download latest version
path = kagglehub.dataset_download("anhoangvo/acdc-dataset")

print("Path to dataset files:", path)