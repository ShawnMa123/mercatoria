from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mercatoria.config import EmbeddingConfig
from mercatoria.embed import EmbeddingClient


class EmbeddingCacheConcurrencyTests(unittest.TestCase):
    def test_concurrent_embeds_share_cache_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EmbeddingConfig(cache_path=str(Path(tmp) / "cache.db"), dim=3)
            cfg.batch_size = 2
            cfg.concurrency = 2
            client = EmbeddingClient(cfg)
            client._post = lambda texts: np.ones((len(texts), 3), dtype=np.float32)  # type: ignore[method-assign]

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: client.embed([f"text-{i}"]), range(16)))

            rows = client._cache.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            self.assertEqual(rows, 16)
            client._cache.close()


if __name__ == "__main__":
    unittest.main()
