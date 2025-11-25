import itertools
import os
from pathlib import Path
from typing import Any


import serpapi

from thesis_graph.data import load_raw_committee_csv


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


def search_for_researcher_profiles(
    client: serpapi.Client,
    name: str,
) -> list[dict[str, Any]] | None:
    params = {
        "engine": "google_scholar",
        "q": name,
    }
    result = client.search(params).as_dict()

    if "profiles" in result:
        return result["profiles"].get("authors", [])

    return None


def main():
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_api_key is None:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    client = serpapi.Client(api_key=serpapi_api_key)

    base_data_path = Path(__file__).parent.parent / "data"
    researchers = (
        load_raw_committee_csv(base_data_path / "committee.csv")["mentor"]
        .unique()
        .tolist()
    )
    print(researchers)

    result = {}

    counter = 70

    for researcher in researchers:
        name_variants = convert_cyrillic_to_latin(researcher)

        for name_variant in name_variants:
            profiles = search_for_researcher_profiles(client, f"{name_variant}, FINKI")
            if profiles is not None and len(profiles) > 0:
                result[researcher] = profiles
                break

        counter -= 1
        if counter == 0:
            break

    print(result)


if __name__ == "__main__":
    main()
