import asyncio
import logging
from typing import List, Optional

from .tsm_config import tsm_config

logger = logging.getLogger(__name__)

_model = None
_model_lock = asyncio.Lock()


def _load_model():
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    model_name = tsm_config.embedding.model
    device = tsm_config.embedding.device
    logger.info(f"Loading local embedding model: {model_name} on {device}")
    _model = SentenceTransformer(model_name, device=device)
    dim = _model.get_sentence_embedding_dimension()
    logger.info(f"Local embedding model ready, dim={dim}")
    return _model


def _encode_sync(text: str) -> List[float]:
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


async def get_local_embedding(text: str) -> List[float]:
    async with _model_lock:
        if _model is None:
            await asyncio.to_thread(_load_model)
    return await asyncio.to_thread(_encode_sync, text)
