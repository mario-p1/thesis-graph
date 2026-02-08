import pandas as pd
from committee_predictor.config import BASE_DATA_PATH
from committee_predictor.utils import load_json_file


def main():
    profiles = load_json_file(BASE_DATA_PATH / "scholar_profiles.json")
    scholar = load_json_file(BASE_DATA_PATH / "scholar_details.json")
    n_articles_to_include = 10

    profiles_df = pd.DataFrame(
        profiles,
        columns=[
            "original_name",
            "author_id",
        ],
    )
    profiles_df = profiles_df.rename(columns={"original_name": "name"})

    scholar_rows = []
    for author_id, details in scholar.items():
        if "interests" in details["author"]:
            interests = [
                interest["title"].lower() for interest in details["author"]["interests"]
            ]
        else:
            interests = []

        if "articles" in details:
            articles = [article["title"] for article in details["articles"]][
                :n_articles_to_include
            ]
        else:
            articles = []

        row = {
            "author_id": author_id,
            "interests": "|".join(interests),
            "articles": "|".join(articles),
        }

        scholar_rows.append(row)

    scholar_df = pd.DataFrame(scholar_rows)

    merged_df = profiles_df.merge(scholar_df, on="author_id", how="left")
    merged_df = merged_df.drop(columns=["author_id"])

    merged_df.to_csv(BASE_DATA_PATH / "researchers.csv", index=False)
    print("Saved researchers.csv")


if __name__ == "__main__":
    main()
