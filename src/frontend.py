from __future__ import annotations

from .backend import Backend
from .db import Image, User


class Frontend:
    def __init__(self, backend: Backend = Backend()) -> None:
        self.backend = backend

    def story_mention(self, instagram_username: str, url: str, rating: int) -> str:
        if rating < 1 or rating > 5:
            raise ValueError("Invalid rating")

        user_id = self.backend.user_service.get_user_id(
            instagram_username, raise_if_empty=False
        )
        image_id = self.backend.image_service.add_image(url)
        self.backend.review_service.add_review(user_id, image_id, rating)

        return "review added"

    def search(self, query: str, location: str, instagram_username: str) -> str:
        if query != "":
            return "queries not supported yet"

        if location != "":
            return "locations not supported yet"

        try:
            user_id = self.backend.user_service.get_user_id(instagram_username)
        except KeyError as e:
            return str(e)

        output = []

        recommendations = self.backend.recommender_service.get_recommendations(
            user_id, 20
        )
        recommended_image_urls = [
            self.backend.image_service.get_image(image_id).url
            for image_id in recommendations
        ]
        output.append("Recommended images:")
        output.extend([f"\t{url}" for url in recommended_image_urls])

        reviews = self.backend.review_service.get_reviews(user_id)
        output.append("My reviews:")
        for review in reviews:
            image = self.backend.image_service.get_image(review.image_id)
            output.append(f"\t{review.rating} - {image.url}")

        output.append("My stats:")
        output.append("No stats available.")

        return "\n".join(output)


def main() -> None:
    frontend = Frontend()

    instructions = "Commands:\n- story_mention <instagram_username> <image_url> <rating>\n- search <query> <location> <instagram_username>"
    print(instructions)

    while True:
        req = input().split()

        if req[0] == "story_mention" and len(req) == 3:
            if len(req) != 3:
                print(
                    "Wrong number of arguments: story_mention <instagram_username> <image_url> <rating>"
                )
            resp = frontend.story_mention(req[1], req[2])
        elif req[0] == "search" and len(req) == 3:
            if len(req) != 3:
                print(
                    "Wrong number of arguments: search <query> <location> <instagram_username>"
                )
            resp = frontend.sign_in(req[1], req[2])
        else:
            print("Unrecognized command")

        print(resp)


if __name__ == "__main__":
    main()
