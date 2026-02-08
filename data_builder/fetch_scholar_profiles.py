from typing import Any

import serpapi
import tqdm

from data_builder.serpapi_client import get_serpapi_client
from committee_predictor.config import BASE_DATA_PATH
from committee_predictor.data import load_thesis_csv
from committee_predictor.utils import (
    convert_cyrillic_to_latin,
    get_current_time_str,
    save_json_to_file,
)


def get_scholar_profiles(
    client: serpapi.Client,
    query: str,
) -> list[dict[str, Any]]:
    params = {
        "engine": "google_scholar",
        "q": query,
    }
    result = client.search(params).as_dict()

    if "profiles" in result:
        return result["profiles"].get("authors", [])

    return []


def search_for_multiple_names(
    client: serpapi.Client,
    names: list[str],
    query_suffix="",
    max_searches: int = 5,
) -> list[dict[str, Any]]:
    result_profiles = []
    search_counter = max_searches

    save_path = (
        BASE_DATA_PATH / "scholar_crawls" / f"profiles_{get_current_time_str()}.json"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for name in tqdm.tqdm(names):
        name_variants = convert_cyrillic_to_latin(name)

        for name_variant in name_variants:
            query = f"{name_variant}{query_suffix}"

            scholar_profiles = get_scholar_profiles(client, query)

            if len(scholar_profiles) > 0:
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
    researchers = (
        load_thesis_csv(BASE_DATA_PATH / "committee.csv")["mentor"].unique().tolist()
    )
    query_suffix = ", FINKI"

    # researchers = ["CUSTOM NAME"]
    # query_suffix = ", CUSTOM AFFILIATION"

    client = get_serpapi_client()

    print("== Researchers ==")
    print(researchers)

    search_for_multiple_names(
        client,
        researchers,
        query_suffix=query_suffix,
        max_searches=len(researchers) * 2,
    )


if __name__ == "__main__":
    main()
