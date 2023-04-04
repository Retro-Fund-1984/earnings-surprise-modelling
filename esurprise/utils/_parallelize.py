import os
from joblib import Parallel, delayed
from tqdm import tqdm


def get_n_jobs(n_jobs):
    max_jobs = os.cpu_count()
    n_jobs = 1 if n_jobs is None else int(n_jobs)
    if n_jobs > max_jobs:
        raise RuntimeError("Cannot assign more jobs than the number of CPUs.")
    elif n_jobs == -1:
        return max_jobs
    else:
        return n_jobs


def parallel_loop(function, iterable, n_jobs=None, progress_bar=False, description=None):
    """
    Using tqdm because it also shows elapsed time. ``rich.progress.track`` looks cooler
    though.

    NOTE: This is a simple function that tracks job starts, not completions. To track
    job completions instead the implementation could get overly complicated, and more
    volatile to changes in future Python versions/setups.
    """
    n_jobs = get_n_jobs(n_jobs)
    iterable = tqdm(iterable, description) if progress_bar else iterable
    return Parallel(n_jobs=n_jobs)(delayed(function)(i) for i in iterable)
