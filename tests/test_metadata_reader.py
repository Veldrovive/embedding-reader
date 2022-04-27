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
    max_size = 1024
    nb_files = 5

    tmp_dir, sizes, expected_array = build_test_collection_parquet_metadata(tmpdir, min_size=min_size, max_size=max_size, nb_files=nb_files,)
    batch_size = random.randint(min_size, max_size)
    embedding_reader = EmbeddingReader(
        tmp_dir,
        file_format='parquet_metadata',
        embedding_column=["metadata", "metadata2"]
    )

    it = embedding_reader(batch_size=batch_size)
    all_batches = list(it)
    meta1, meta2 = [], []
    for (meta1_batch, meta_2_batch), _ in all_batches:
        meta1.extend(meta1_batch)
        meta2.extend(meta_2_batch)
    recovered_data = [meta1, meta2]
    assert recovered_data == expected_array