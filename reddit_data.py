from pathlib import Path
import pandas as pd
from typing import Union
from dataclasses import dataclass
from typing import List, Union
from PIL import Image
from pysentimiento import create_analyzer
from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
visual_model = CLIPModel.from_pretrained(model_name).to(device)
visual_processor = CLIPProcessor.from_pretrained(model_name)
analyzer = create_analyzer(task="sentiment", lang="en")

class Redditor:
    def __init__(self, id: str):
        self.id = id
        self.posts: List[RedditPost] = []
        self.comments: List[RedditComment] = []
    
    def __repr__(self):
        return f"Redditor(id={self.id})"
    
    def add_post(self, post: 'RedditPost'):
        if post.poster_id == self.id:
            self.posts.append(post)
    
    def add_comment(self, comment: 'RedditComment'):
        if comment.commenter_id == self.id:
            self.comments.append(comment)

@dataclass
class RedditComment:
    commenter_id: str
    comment: str

    def __init__(self, commenter_id: str, comment: str):
        self.commenter_id = commenter_id
        self.text = comment

    @property
    def sentiment(self):
        return analyzer.predict(self.text).output


class RedditPost:
    def __init__(self, post_id: str,  post_url: str, img_path: Union[str, Path], poster_id: str, title: str, subreddit: str, commenter_ids: List[str], top_level_comments: List[str]):
        self.poster_id = poster_id
        self.post_id = post_id
        self.post_url = post_url
        self.subreddit = subreddit
        self.img_path = Path(img_path)
        self.title = title
        
        if len(commenter_ids) != len(top_level_comments):
            raise ValueError("commenter_ids and top_level_comments must have the same length")
        
        self.comments: List[RedditComment] = [RedditComment(commenter_ids[i], top_level_comments[i]) 
                                              for i in range(len(commenter_ids))]
    
    @property
    def image(self):
        return Image.open(self.img_path)
    
    @property
    def title_sentiment(self):
        return analyzer.predict(self.title).output
    
    @property
    def image_embedding(self):
        inputs = visual_processor(images=self.image, return_tensors="pt")
        with torch.no_grad():
            image_features = visual_model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_embedding.cpu()

    def __repr__(self):
        return f"RedditPost(post_id={self.post_id}, title={self.title}, poster_id={self.poster_id})"


def get_post_data(path: Union[Path, str]) -> pd.DataFrame:
    post_data = pd.read_csv(path)
    post_data['commenter_ids'] = post_data['commenter_ids'].map(eval)
    post_data['top_level_comments'] = post_data['top_level_comments'].map(eval)
    post_data = post_data.dropna(subset=['post_url'])
    return post_data

def get_reddit_posts(data_dir: Path, post_data: pd.DataFrame) -> List[RedditPost]:
    rows = [row.to_dict() for _, row in post_data.iterrows()]

    for row in rows:
        extension = row['post_url'].split(".")[-1]
        row['img_path'] = data_dir/"images"/f"{row['post_id']}.{extension}"

    return [RedditPost(**row) for row in rows if row['img_path'].exists()]