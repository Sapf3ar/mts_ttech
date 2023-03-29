from schema import Video
from utils import mongo_engine_to_json
from pydantic import BaseModel
from typing import List


class VideoCardModel(BaseModel):
    infered: bool
    description: str = None
    rating: str = None
    name: str = None
    key: str
    uid: str
    image: str = None


class VideoCardResponse(BaseModel):
    cards: List[VideoCardModel]


def get_video_cards():
    videos = [mongo_engine_to_json(video) for video in Video.objects()]
    response = VideoCardResponse(cards=videos)
    return response
