from typing import Dict, List, Optional, Any


def infer_on_example(
    model,
    tokenizer,
    labeled_example: Dict[str, Any],
    target_colname: str,
    unlabeled_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    return
