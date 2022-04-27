"""Reader module exposes the reading functionality of all formats"""

from embedding_reader.numpy_reader import NumpyReader
from embedding_reader.parquet_reader import ParquetReader
from embedding_reader.parquet_numpy_reader import ParquetNumpyReader
from embedding_reader.parquet_metadata_reader import ParquetMetadataReader

class CombinedEmbeddingReader:
    def __init__(self, embedding_readers):
        self.embedding_readers = embedding_readers
        self.num_readers = len(self.embedding_readers)
        assert len(set((reader.count for reader in self.embedding_readers))) == 1, "All readers must have the same count"
        self.count = embedding_readers[0].count
    
    def __call__(self, batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True):
        generators = [None] * self.num_readers
        for i, reader in enumerate(self.embedding_readers):
            generators[i] = reader(batch_size, start=start, end=end, max_piece_size=max_piece_size, parallel_pieces=parallel_pieces, show_progress=False)
        while True:
            try:
                data = [None] * self.num_readers
                for i, generator in enumerate(generators):
                    data_point, meta = next(generator)
                    data[i] = data_point
                yield data
            except StopIteration:
                return
            
class EmbeddingReader:
    """reader class, implements init to read the files headers and call to produce embeddings batches"""

    def __init__(
        self,
        embeddings_folder,
        file_format="npy",
        embedding_column="embedding",
        meta_columns=None,
        metadata_folder=None,
    ):
        if file_format == "npy":
            self.reader = NumpyReader(embeddings_folder)
        elif file_format == "parquet":
            self.reader = ParquetReader(
                embeddings_folder, embedding_column_name=embedding_column, metadata_column_names=meta_columns
            )
        elif file_format == "parquet_npy":
            self.reader = ParquetNumpyReader(embeddings_folder, metadata_folder, meta_columns)
        elif file_format == "parquet_metadata":
            self.reader = ParquetMetadataReader(embeddings_folder, embedding_column)
        else:
            raise ValueError("format must be npy, parquet or parquet_npy")

        self.dimension = self.reader.dimension
        self.count = self.reader.count
        self.byte_per_item = self.reader.byte_per_item
        self.total_size = self.reader.total_size
        self.embeddings_folder = embeddings_folder

    def __call__(self, batch_size, start=0, end=None, max_piece_size=None, parallel_pieces=None, show_progress=True):
        return self.reader(batch_size, start, end, max_piece_size, parallel_pieces, show_progress)
