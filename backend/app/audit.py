from typing import Dict

from .storage import insert_audit


def record(event_type: str, payload: Dict[str, object]) -> None:
    insert_audit(event_type, payload)
