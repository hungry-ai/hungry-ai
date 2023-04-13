# from google.cloud import vision
import os
from typing import List


# Put your own token path here
# Assumes the script runs from outside src !!!
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hungryai-d6f8c9e2f0c9.json"


class PrMatch:
    def __init__(
        self,
    ):
        # self.client = vision.ImageAnnotatorClient()
        # self.image = vision.Image()
        pass

    def __call__(self, url: str, tags: List[str]) -> List[float]:
        # self.image.source.image_uri = url
        # response = self.client.label_detection(image=self.image)
        # print(response)
        # If tag in response, return probability, else return -1
        return [0.5] * len(tags)


# if __name__ == '__main__':
#    pr_match = PrMatch()
#    pr_match(r"https://i.imgur.com/2EUmDJO.jpg", ['sm'])
