from pathlib import Path
import pandas as pd
from typing import List, Union, Mapping
from PIL import Image
from pysentimiento import create_analyzer
from transformers import CLIPProcessor, CLIPModel
import torch
import easyocr

# Models needed for computed properties
# Image Embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
visual_model = CLIPModel.from_pretrained(model_name).to(device)
visual_processor = CLIPProcessor.from_pretrained(model_name)

# Sentiment Analysis
analyzer = create_analyzer(task="sentiment", lang="en")

# Optical Character Recognition
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

class RedditGraph:
    def __init__(self, posts: List['RedditPost']):
        self.redditors: Mapping[str, 'Redditor'] = {}

        if len(posts) > 0:
            for post in posts:
                self.add_post(post)

    def __repr__(self):
        return f"RedditGraph(redditors_count={len(self.redditors)})"
    
    def add_post(self, post: 'RedditPost'):
        if post.poster_id not in self.redditors:
            self.redditors[post.poster_id] = Redditor(post.poster_id)
        self.redditors[post.poster_id].add_post(post)
        
        for comment in post.comments:
            if comment.commenter_id not in self.redditors:
                self.redditors[comment.commenter_id] = Redditor(comment.commenter_id)
            self.redditors[comment.commenter_id].add_comment(comment)

class Redditor:
    def __init__(self, id: str):
        self.id = id
        self.postid_set = set()
        self.posts: List[RedditPost] = []
        self.commentid_set = set()
        self.comments: List[RedditComment] = []
    
    def __repr__(self):
        return f"Redditor(id={self.id}, posts_count={len(self.posts)}, comments_count={len(self.comments)})"
    
    def add_post(self, post: 'RedditPost'):
        if post.poster_id == self.id:
            if post.post_id not in self.postid_set:
                self.postid_set.add(post.post_id)
                self.posts.append(post)
        else:
            raise ValueError(f"Post does not belong to this Redditor, poster_id: {post.poster_id}, redditor_id: {self.id}")
    
    def add_comment(self, comment: 'RedditComment'):
        if comment.commenter_id == self.id:
            if comment.id not in self.commentid_set:
                self.commentid_set.add(comment.id)
                self.comments.append(comment)
        else:
            raise ValueError("Comment does not belong to this Redditor")
    
    @property
    def user_embedding(self):
        posts_sentiments: Mapping[str, List['RedditPost']] = {sentiment: [] for sentiment in ["POS", "NEU", "NEG"]}

        for post in self.posts:
            sentiment = post.title_sentiment
            posts_sentiments[sentiment].append(post)

        for comment in self.comments:
            sentiment = comment.sentiment
            posts_sentiments[sentiment].append(comment.post)

        user_embeddings = []
        for sentiment in posts_sentiments:
            sentiment_image_embeddings = [post.image_embedding for post in posts_sentiments[sentiment]]

            if len(sentiment_image_embeddings) == 0:
                sentiment_image_embeddings.append(torch.zeros(1, 512))
                
            post_sentiment_embedding = torch.stack(sentiment_image_embeddings, axis=0)
            average_sentiment_embedding = post_sentiment_embedding.mean(axis=0).squeeze(0)
            user_embeddings.append(average_sentiment_embedding)

        return torch.stack(user_embeddings, axis=0)

class RedditComment:
    def __init__(self, source_post: 'RedditPost', commenter_id: str, comment: str):
        self.post = source_post
        self.commenter_id = commenter_id
        self.text = comment

    @property
    def sentiment(self):
        return analyzer.predict(self.text).output
    
    @property
    def id(self):
        return f"{self.commenter_id}_{hash(self.text)}"


class RedditPost:
    def __init__(
            self, 
            post_id: str,  
            post_url: str, 
            img_path: Union[str, Path], 
            poster_id: str, 
            title: str, 
            subreddit: str, 
            commenter_ids: List[str], 
            top_level_comments: List[str]
        ):
        self.poster_id = poster_id
        self.post_id = post_id
        self.post_url = post_url
        self.subreddit = subreddit
        self.img_path = Path(img_path)
        self.title = title
        
        if len(commenter_ids) != len(top_level_comments):
            raise ValueError("commenter_ids and top_level_comments must have the same length")
        
        self.comments: List[RedditComment] = [RedditComment(self, commenter_ids[i], top_level_comments[i]) 
                                              for i in range(len(commenter_ids))]
    
    @property
    def image(self):
        return Image.open(self.img_path)
    
    @property
    def image_text(self):
        return " ".join(ocr_reader.readtext(self.image, detail=0))
    
    @property
    def image_embedding(self):
        inputs = visual_processor(images=self.image, return_tensors="pt")
        with torch.no_grad():
            image_features = visual_model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_embedding.cpu()
    
    @property
    def title_sentiment(self):
        return analyzer.predict(self.title).output

    def __repr__(self):
        return f"RedditPost(post_id={self.post_id}, title={self.title}, poster_id={self.poster_id})"


def get_processed_posts_data_frame(path: Union[Path, str]) -> pd.DataFrame:
    post_data = pd.read_csv(path)
    post_data['commenter_ids'] = post_data['commenter_ids'].map(eval)
    post_data['top_level_comments'] = post_data['top_level_comments'].map(eval)
    post_data = post_data.dropna(subset=['post_url', 'poster_id'])
    return post_data

def convert_posts_df_to_reddit_posts(data_dir: Path, post_data: pd.DataFrame) -> List[RedditPost]:
    rows = [row.to_dict() for _, row in post_data.iterrows()]

    for row in rows:
        extension = row['post_url'].split(".")[-1]
        row['img_path'] = data_dir/"images"/f"{row['post_id']}.{extension}"

    return [RedditPost(**row) for row in rows if row['img_path'].exists()]