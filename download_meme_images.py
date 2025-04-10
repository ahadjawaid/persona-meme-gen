import pandas as pd
from pathlib import Path
import requests
from tqdm.auto import tqdm

def download_image(url: str, path: Path):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise e

    path.write_bytes(r.content)

if __name__ == "__main__":
    data_path = Path("data")
    post_path = data_path / "posts.csv"

    post_data = pd.read_csv(post_path)
    post_data = post_data[(post_data["title"] != "[deleted by user]") & (post_data["post_url"].notna())]
    column_keys = post_data.columns.to_list()

    image_dir = data_path / "images"
    if not image_dir.exists() or len(list(image_dir.iterdir())) != len(post_data):
        image_dir.mkdir(parents=True, exist_ok=True)

        for row in tqdm(post_data.values):
            post_id = row[0]
            post_url = row[1]

            if not isinstance(post_url, str):
                raise Exception(f"Invalid URL for post {post_id}: {post_url}")

            extension = post_url.split(".")[-1]
            if not extension or len(extension) > 5:
                extension = ".jpg"

            post_img_path = image_dir / f"{post_id}.{extension}"
            if not post_img_path.exists():
                try:
                    download_image(post_url, post_img_path)
                except Exception as e:
                    print(f"Failed to download image for post {post_id}: {e}")
                    continue