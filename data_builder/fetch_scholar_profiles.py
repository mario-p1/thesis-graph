from typing import Any

import serpapi
import tqdm

from data_builder.serpapi_client import get_serpapi_client
from thesis_graph.data import load_thesis_csv
from thesis_graph.utils import (
    base_data_path,
    convert_cyrillic_to_latin,
    get_current_time_str,
    save_json_to_file,
)


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
        base_data_path / "scholar_crawls" / f"profiles_{get_current_time_str()}.json"
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
                save_json_to_file(save_path, result_profiles)
                break

        search_counter -= 1
        if search_counter == 0:
            break

    return result_profiles


def main():
    client = get_serpapi_client()

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
