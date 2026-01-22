from __future__ import annotations

from typing import Any, Dict

import requests

from .config import settings


DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_PREDICT = 900


def generate_json(
    prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_predict: int | None = None,
) -> Dict[str, Any]:
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
                "top_p": top_p if top_p is not None else DEFAULT_TOP_P,
                "num_predict": num_predict if num_predict is not None else DEFAULT_NUM_PREDICT,
            },
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    return payload
