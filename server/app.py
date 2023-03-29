from flask import Flask


UPLOAD_FOLDER = 'client/src/assets'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024


@app.after_request
def after_request(response):
    if response.headers.get('mimetype') == 'video/mkv':
        response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
