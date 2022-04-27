"""Parquet embedding reader, read embeddings from parquet files in streaming"""

import pandas as pd
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
from collections import namedtuple
from embedding_reader.get_file_list import get_file_list
from embedding_reader.piece_builder import build_pieces, PIECES_BASE_COLUMNS
from threading import Semaphore
import math


class ParquetMetadataReader:
    """Parquet reader class, implements init to read the files headers and call to produce embeddings batches"""

    def __init__(self, embeddings_folder, embedding_column_names):
        self.embeddings_folder = embeddings_folder
        self.fs, embeddings_file_paths = get_file_list(embeddings_folder, "parquet")

        self.embedding_column_names = [embedding_column_names] if isinstance(embedding_column_names, str) else embedding_column_names

        def file_to_header(filename):
            try:
                with self.fs.open(filename, "rb") as f:
                    parquet_file = pq.ParquetFile(f, memory_map=True)
                    return (None, [filename, parquet_file.metadata.num_rows])
            except Exception as e:  # pylint: disable=broad-except
                return e, (filename, None)

        headers = []
        count_before = 0
        with ThreadPool(10) as p:
            for err, c in tqdm(p.imap(file_to_header, embeddings_file_paths), total=len(embeddings_file_paths)):
                if err is not None:
                    raise Exception(f"failed reading file {c[0]}") from err
                if c[1] == 0:
                    continue
                headers.append([*c, count_before])
                count_before += c[1]

        self.headers = pd.DataFrame(headers, columns=["filename", "count", "count_before"])
        self.count = self.headers["count"].sum()
        if self.count == 0:
            raise ValueError("No embeddings found in folder {}".format(embeddings_folder))
        
        self.dimension = None
        self.byte_per_item = None
        self.total_size = None

    def __call__(self, batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True):
        if end is None:
            end = self.count

        if end > self.count:
            end = self.count
        if batch_size > end - start:
            batch_size = end - start

        if max_piece_size is None:
            max_piece_size = 1
        if parallel_pieces is None:
            parallel_pieces = 10

        pieces = build_pieces(
            headers=self.headers, batch_size=batch_size, start=start, end=end, max_piece_size=max_piece_size
        )

        cols = PIECES_BASE_COLUMNS
        Piece = namedtuple("Count", cols)

        def read_piece(piece):
            try:
                start = piece.piece_start
                end = piece.piece_end
                path = piece.filename

                with self.fs.open(path, "rb") as f:
                    length = end - start
                    table = pq.read_table(f, use_threads=False)
                    table_slice = table.slice(start, length)

                    embeddings_raw = []
                    for embedding_column_name in self.embedding_column_names:
                        embeddings_raw.append(
                            table_slice[embedding_column_name].to_pylist()
                        )

                    return (None, (embeddings_raw, None, piece,))
            except Exception as e:  # pylint: disable=broad-except
                return e, (None, None, piece)

        semaphore = Semaphore(parallel_pieces)
        stopped = False

        def piece_generator(pieces):
            for piece in (Piece(*parts) for parts in zip(*[pieces[col] for col in cols])):
                if stopped:
                    break
                semaphore.acquire()
                yield piece

        batch = None
        batch_meta = None

        if show_progress:
            pbar = tqdm(total=len(pieces))
        with ThreadPool(parallel_pieces) as p:
            for err, (data, _, piece) in p.imap(read_piece, piece_generator(pieces)):
                if err is not None:
                    stopped = True
                    semaphore.release()
                    raise Exception(
                        f"failed reading file {piece.filename} from {piece.piece_start} to {piece.piece_end}"
                    ) from err
                try:
                    if batch is None:
                        batch = list(([] for _ in range(len(data))))
                    for i, column_data in enumerate(data):
                        batch[i].extend(column_data)
                    if piece.last_piece:
                        meta_batch_df = pd.DataFrame(
                            np.arange(start=piece.batch_start, stop=piece.batch_end), columns=["i"]
                        )
                        yield batch, meta_batch_df
                        batch = None

                    if show_progress:
                        pbar.update(1)
                    semaphore.release()
                except Exception as e:  # pylint: disable=broad-except
                    stopped = True
                    semaphore.release()
                    raise e

        if show_progress:
            pbar.close()
