"""stand-in for marker.services.openai (the worker wraps its OpenAIService and its json)"""
import json


class OpenAIService:
    def __call__(self, prompt, image, block, response_schema, *a, **k):
        return json.loads("{}")
