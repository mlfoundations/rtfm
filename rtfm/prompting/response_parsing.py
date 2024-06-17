import json
import re
from typing import Union, Dict


def extract_longest_valid_json(completion: str) -> Union[Dict[str, str], None]:
    """Extract the longest valid JSON, and return it as a dict."""
    json_pattern = r"\{.*?\}"
    matches = re.findall(json_pattern, completion, re.DOTALL)
    # Initialize variables to keep track of the longest valid JSON
    longest_json = None
    parsed_json = None
    max_length = 0
    for match in matches:
        try:
            # Load string as JSON to check validity
            parsed_json = json.loads(match)
            if len(match) > max_length:
                longest_json = match
                max_length = len(match)
        except json.JSONDecodeError:
            continue
    return parsed_json


def parse_json_completion(completion: str) -> Union[Dict[str, str], None]:
    """Extract the JSON from a chat completion string."""
    if "{" not in completion or "}" not in completion:
        raise ValueError(f"got bad completion {completion}")
    parsed_completion = extract_longest_valid_json(completion)
    return parsed_completion
