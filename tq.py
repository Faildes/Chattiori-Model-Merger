from tqdm.notebook import tqdm as tqdmn
from tqdm.auto import tqdm as tqdma

def tqdm(*args, **kwargs):
    try:
        return tqdmn(*args, **kwargs)
    except:
        return tqdma(*args, **kwargs)
