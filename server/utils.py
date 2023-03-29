import traceback
from flask import request
import json


def mongo_engine_to_json(me_object):
    return me_object.to_mongo().to_dict()


def get_request_data():
    try:
        if request.form and 'data' in request.form:
            return json.loads(request.form.get('data'))
        elif request.json and 'data' in request.json:
            return json.loads(request.json['data'])
    except Exception:
        traceback.print_exc()
    return {}