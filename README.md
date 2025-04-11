# Persona-based Meme Generation

In this repository we explore user based meme generation based on a user's subreddit history. 

## Setup

Create environment
```
conda create -n memegen python=3.9
```

```
conda activate memegen
```

Install dependencies
```
pip install -r requirements.txt
```

## Data

You can download the dataset from the following [link](https://cometmail-my.sharepoint.com/:u:/g/personal/axj200012_utdallas_edu/EebYFTmg8LNMtbPpFuPqkFUBlqFgcjwss_6wsazgzE4e5w?e=d3avV1) (Expires July 10, 2025). You would want to unzip this dataset in the top level of the clone repository.

## Usage

Use the following command to download meme images from reddit links from the `post.csv`
```
python3 download_meme_images.py
```

Initializing the `RedditGraph`:

```
from pathlib import Path
from reddit_data import *

data_dir = Path("data")
post_path = data_dir/"posts.csv"

post_data = get_processed_posts_data_frame(post_path)
reddit_posts = convert_posts_df_to_reddit_posts(data_dir, post_data)

reddit_graph = RedditGraph(reddit_posts)
```

Accessing a redditor using `RedditGraph`:

```
redditor = list(reddit_graph.redditors.values())[0] # Indexing the first redditor
print(redditor.user_embedding.shape)
```
