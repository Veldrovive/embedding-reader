from embedding_reader import EmbeddingReader
import random
import numpy as np
import pytest
import pandas as pd

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

from tests.fixtures import build_test_collection_parquet_metadata

def test_metadata_embedding_reader(tmpdir):
    min_size = 1
    max_size = 10
    nb_files = 5

    tmp_dir, sizes, expected_array = build_test_collection_parquet_metadata(tmpdir, min_size=min_size, max_size=max_size, nb_files=nb_files,)
    total_size = sum(sizes)
    batch_size = random.randint(min_size, max_size)
    logger.warning(f"tmp_dir: {tmp_dir}. sizes: {sizes}. total_size: {total_size}, batch_size: {batch_size}")
    embedding_reader = EmbeddingReader(
        tmp_dir,
        file_format='parquet_metadata',
        embedding_column="metadata",
        metadata_folder=tmp_dir,
    )

    it = embedding_reader(batch_size=batch_size)
    all_batches = list(it)
    logger.warning(f"all_batches: {all_batches}.")