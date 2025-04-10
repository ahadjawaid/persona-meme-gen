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

## Usage

Use the following command to download meme images from reddit links from the `post.csv`
```
python3 download_meme_images.py
```