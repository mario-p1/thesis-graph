from datetime import datetime
import itertools
import json
import os
from pathlib import Path
from typing import Any


import serpapi
import tqdm

from thesis_graph.data import load_thesis_csv

base_data_path = Path(__file__).parent.parent / "data"


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


def get_scholar_profiles(
    client: serpapi.Client,
    query: str,
) -> list[dict[str, Any]] | None:
    params = {
        "engine": "google_scholar",
        "q": query,
    }
    result = client.search(params).as_dict()

    if "profiles" in result:
        return result["profiles"].get("authors", [])

    return None


def search_for_multiple_cyrillic_names(
    client: serpapi.Client,
    cyrillic_names: list[str],
    query_suffix="",
    max_searches: int = 5,
) -> list[dict[str, Any]]:
    result_profiles = []
    search_counter = max_searches

    save_path = (
        base_data_path
        / "scholar_crawls"
        / f"profiles_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for name in tqdm.tqdm(cyrillic_names):
        name_variants = convert_cyrillic_to_latin(name)

        for name_variant in name_variants:
            query = f"{name_variant}{query_suffix}"

            scholar_profiles = get_scholar_profiles(client, query)
            if scholar_profiles is not None and len(scholar_profiles) > 0:
                for scholar_profile in scholar_profiles:
                    result_profiles.append(
                        {
                            "original_name": name,
                            "search_query": query,
                            **scholar_profile,
                        }
                    )
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(result_profiles, f, ensure_ascii=False, indent=4)
                break

        search_counter -= 1
        if search_counter == 0:
            break

    return result_profiles


def main():
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_api_key is None:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    client = serpapi.Client(api_key=serpapi_api_key)

    researchers = (
        load_thesis_csv(base_data_path / "committee.csv")["mentor"].unique().tolist()
    )
    query_suffix = ", FINKI"

    # researchers = ["Ласко Баснарков", "Милош Јовановиќ"]
    # query_suffix = ""

    print("== Researchers ==")
    print(researchers)

    search_for_multiple_cyrillic_names(
        client,
        researchers,
        query_suffix=query_suffix,
        max_searches=len(researchers) * 2,
    )


if __name__ == "__main__":
    main()
