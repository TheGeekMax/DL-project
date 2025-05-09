
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
