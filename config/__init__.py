class ConfigDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def from_dict(obj) -> ConfigDict:
    if not isinstance(obj, dict):
        return obj
    d = ConfigDict()
    for k, v in obj.items():
        d[k] = from_dict(v)
    return d
