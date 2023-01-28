from __future__ import annotations

from .backend import Backend
from .db import Image, User


def signed_in(func):
    def wrapper(self, *args):
        if self.user is None:
            return "You're not signed in"
        return func(self, *args)

    return wrapper


def signed_out(func):
    def wrapper(self, *args):
        if self.user is not None:
            return "You're already signed in"
        return func(self, *args)

    return wrapper


def recommends(func):
    def wrapper(self, *args):
        resp = func(self, *args)
        return resp or self._recommend()

    return wrapper


class Frontend:
    def __init__(self, backend: Backend = Backend()) -> None:
        self.backend = backend
        self.user: None | User = None
        self.current_image: None | Image = None

    @signed_out
    def sign_up(self, email: str, password: str) -> None | str:
        try:
            self.backend.user_service.sign_up(email, password)
        except ValueError as e:
            return str(e)

        return None

    @signed_out
    @recommends
    def sign_in(self, email: str, password: str) -> None | str:
        try:
            self.user = self.backend.user_service.sign_in(email, password)
        except ValueError as e:
            return str(e)

        return None

    @signed_in
    def sign_out(self) -> None | str:
        self.user = None

        return None

    @signed_in
    @recommends
    def upload(self, url: str, rating: int) -> None | str:
        user_id = self.user.user_id  # type: ignore[union-attr]
        image_id = self.backend.image_service.add_image(url)
        try:
            self.backend.review_service.add_review(user_id, image_id, rating)
        except ValueError as e:
            return str(e)

        return None

    @signed_in
    @recommends
    def review(self, rating: int) -> None | str:
        if self.current_image is None:
            return "No image to review"

        user_id = self.user.user_id  # type: ignore[union-attr]
        image_id = self.current_image.image_id
        try:
            self.backend.review_service.add_review(user_id, image_id, rating)
        except ValueError as e:
            return str(e)

        return None

    @signed_in
    def _recommend(self) -> None | str:
        assert self.user is not None
        user_id = self.user.user_id
        image_id = self.backend.recommender_service.recommend(user_id)
        if image_id is None:
            return "No images to review - try uploading one"
        self.current_image = self.backend.image_service.get_image(image_id)
        if self.current_image is None:
            return "Error: could not find image"
        return f"Please rate the following image: {self.current_image.url}"


def main() -> None:
    frontend = Frontend()

    instructions = "Commands:\n- sign_up <email> <password>\n- sign_in <email> <password>\n- sign_out\n- upload <url> <rating>\n- review <rating>"
    print(instructions)

    while True:
        req = input().split()

        if req[0] == "sign_up" and len(req) == 3:
            resp = frontend.sign_up(req[1], req[2])
        elif req[0] == "sign_in" and len(req) == 3:
            resp = frontend.sign_in(req[1], req[2])
        elif req[0] == "sign_out" and len(req) == 1:
            resp = frontend.sign_out()
        elif req[0] == "upload" and len(req) == 3:
            if not (req[2].isdigit() and 1 <= int(req[2]) <= 5):
                print("Invalid rating")
                continue
            resp = frontend.upload(req[1], int(req[2]))
        elif req[0] == "review" and len(req) == 2:
            if not (req[1].isdigit() and 1 <= int(req[1]) <= 5):
                print("Invalid rating")
                continue
            resp = frontend.review(int(req[1]))
        else:
            print("Unrecognized command")

        if resp is not None:
            print(resp)


if __name__ == "__main__":
    main()
