from schema import Video
import os
from app import app


def get_video_file(video_id: str):
    video = Video.objects(uid=video_id).first()
    if not video:
        return {'message': f'There is no video with id {video_id}'}
    return {'path': os.path.join(app.config['UPLOAD_FOLDER'], video.key)}
