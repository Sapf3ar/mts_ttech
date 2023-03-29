from __main__ import app
from get_video_file import get_video_file
from get_video_cards import get_video_cards, VideoCardResponse
from upload_video import upload_video, UVideoRequest
from flask import request
from utils import get_request_data


@app.route('/api/v1/get_video_cards')
def _video_cards():
    response: VideoCardResponse = get_video_cards()
    return response.dict()


@app.route('/api/v1/get_video_file/<string:video_id>')
def _get_video_file(video_id: str):
    return get_video_file(video_id)


@app.route('/api/v1/upload_video', methods=['POST'])
def _upload_video():
    data = get_request_data()
    print(data)
    req = UVideoRequest(**data, files=request.files)
    return upload_video(req)

