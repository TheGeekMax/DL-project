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

def get_args(args, arg_name: str, default: str):
    if arg_name in args:
        index = args.index(arg_name)
        if index + 1 < len(args):
            return args[index + 1]
    return default


def get_args_int(args, arg_name: str, default: int):
    if arg_name in args:
        index = args.index(arg_name)
        if index + 1 < len(args):
            return int(args[index + 1])
    return default


def get_args_float(args, arg_name: str, default: float):
    if arg_name in args:
        index = args.index(arg_name)
        if index + 1 < len(args):
            return float(args[index + 1])
    return default


def get_args_bool(args, arg_name: str, default: bool):
    if arg_name in args:
        index = args.index(arg_name)
        if index + 1 < len(args):
            return args[index + 1].lower() == "true"
    return default


def get_args_str(args, arg_name: str, default: str):
    if arg_name in args:
        index = args.index(arg_name)
        if index + 1 < len(args):
            return args[index + 1]
    return default

