from schema import Video
from uuid import uuid4
import os
from pydantic import BaseModel
from typing import Any
from app import app


class UVideoRequest(BaseModel):
    infered: bool
    description: str = None
    rating: str = None
    name: str = None
    image: str = None
    files: Any


def upload_video(req: UVideoRequest):
    for name, file in req.files.items():
        uid = str(uuid4())
        file_name = name + uid + '.mkv'
        path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        f = open(path, 'wb+')
        f.close()
        file.save(path)
        video = Video(uid=uid, key=file_name, **req.dict(exclude={'files'}))
        video.save()
    return {}
