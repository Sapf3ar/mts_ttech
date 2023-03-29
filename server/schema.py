from os import environ
from mongoengine import connect, StringField, Document, BooleanField


connect(
    db=environ.get("DBNAME"),
    host=f'mongodb://{environ.get("HOST")}'
)


class Video(Document):
    uid = StringField(required=True)
    key = StringField(required=True)
    infered = BooleanField(required=True)
    description = StringField()
    rating = StringField()
    name = StringField()
    image = StringField()
