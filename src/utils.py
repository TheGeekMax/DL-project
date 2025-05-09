import os

def create_if_non_existant(path : str):
    if not os.path.exists(path):
        os.makedirs(path)

def download_dataset_if_non_existant(url : str, path : str):
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}")
        response = req.get(url)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Error downloading {url} : {response.status_code}")
            sys.exit(1)