from datetime import datetime
import itertools
import json
from pathlib import Path
from typing import Any


def convert_cyrillic_to_latin(text: str) -> list[list[str]]:
    alphabet = {
        "А": ["A"],
        "Б": ["B"],
        "В": ["V"],
        "Г": ["G"],
        "Д": ["D"],
        "Ѓ": ["G", "Gj"],
        "Е": ["E"],
        "Ж": ["Zh", "Z"],
        "З": ["Z"],
        "Ѕ": ["Dz"],
        "И": ["I"],
        "Ј": ["J"],
        "К": ["K"],
        "Л": ["L"],
        "Љ": ["Lj", "L"],
        "М": ["M"],
        "Н": ["N"],
        "Њ": ["Nj", "N"],
        "О": ["O"],
        "П": ["P"],
        "Р": ["R"],
        "С": ["S"],
        "Т": ["T"],
        "Ќ": ["K", "KJ"],
        "У": ["U"],
        "Ф": ["F"],
        "Х": ["H"],
        "Ц": ["C"],
        "Ч": ["Ch", "C"],
        "Џ": ["Dz", "D"],
        "Ш": ["Sh", "S"],
    }

    combinations = []
    for char in text:
        upper = char.upper()
        if upper in alphabet:
            # character is cyrillic
            if char.isupper():
                combinations.append(alphabet[upper])
            else:
                combinations.append([c.lower() for c in alphabet[upper]])
        else:
            combinations.append([char])

    product = itertools.product(*combinations)
    return ["".join(p) for p in product]


def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json_to_file(save_path: Path, result: Any):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reverse_dict(d: dict[Any, Any]) -> dict[Any, Any]:
    return {v: k for k, v in d.items()}
