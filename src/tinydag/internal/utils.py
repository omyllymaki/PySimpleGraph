import pickle
from typing import Any


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_pickle(path: str, data: Any) -> None:
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
