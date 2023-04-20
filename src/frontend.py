from __future__ import annotations

from .backend import Backend


class Frontend:
    def __init__(self, backend: Backend = Backend()) -> None:
        self.backend = backend

    def story_mention(self, instagram_username: str, url: str, rating: int) -> str:
        if rating < 1 or rating > 5:
            raise ValueError("Invalid rating")

        try:
            user = self.backend.get_user(instagram_username)
        except KeyError:
            user = self.backend.add_user(instagram_username)

        image = self.backend.add_image(url)

        self.backend.add_review(user, image, rating)

        return "review added"

    def search(self, query: str, location: str, instagram_username: str) -> str:
        if query != "":
            return "queries not supported yet"

        if location != "":
            return "locations not supported yet"

        try:
            user = self.backend.get_user(instagram_username)
        except KeyError:
            user = self.backend.add_user(instagram_username)

        recommendations = self.backend.get_recommendations(user, 20)
        recommended_image_urls = [image.url for image in recommendations]

        output = []
        output.append("Recommended images:")
        if len(recommended_image_urls) == 0:
            output.append("No recommendations available.")
        else:
            output.extend([f"\t{url}" for url in recommended_image_urls])

        reviews = self.backend.get_reviews(user)
        output.append("My reviews:")
        if len(reviews) == 0:
            output.append("No reviews available.")
        else:
            for review in reviews:
                output.append(f"\t{review.rating} - {review.image.url}")

        output.append("My stats:")
        output.append("No stats available.")

        return "\n".join(output)


def main() -> None:
    frontend = Frontend()

    instructions = """\
Commands:
- story_mention <instagram_username> <image_url> <rating>
- search <query> <location> <instagram_username>"""
    print(instructions)

    while True:
        req = input().split()

        if req[0] == "story_mention" and len(req) == 4:
            if len(req) != 3:
                print(
                    "Wrong number of arguments: story_mention <instagram_username> <image_url> <rating>"
                )
            resp = frontend.story_mention(req[1], req[2], int(req[3]))
        elif req[0] == "search" and len(req) == 4:
            if len(req) != 3:
                print(
                    "Wrong number of arguments: search <query> <location> <instagram_username>"
                )
            resp = frontend.search(req[1], req[2], req[3])
        else:
            print("Unrecognized command")

        print(resp)


if __name__ == "__main__":
    main()
