import time
from datagen import load_books, BOOKS
start = time.time()
loaded_books = load_books(BOOKS)
print(f"Loaded books in {time.time() - start:.2f} seconds")

import multiprocessing
def worker(worker_id, lb, q):
    pass

if __name__ == '__main__':
    start = time.time()
    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(16, initializer=worker, initargs=(0, loaded_books, q))
    print(f"Pool started in {time.time() - start:.2f} seconds")
