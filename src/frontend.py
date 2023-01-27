from backend import Backend


def signed_in(func):
    def wrapper(self, *args):
        if self.user is None:
            print("You're not signed in")
            return
        func(self, *args)

    return wrapper


def signed_out(func):
    def wrapper(self, *args):
        if self.user is not None:
            print("You're already signed in")
            return
        func(self, *args)

    return wrapper


def recommends(func):
    def wrapper(self, *args):
        func(self, *args)
        self._recommend()

    return wrapper


class Frontend:
    def __init__(self) -> None:
        self.backend = Backend()
        self.user = None
        self.current_image = None

    @signed_out
    def sign_up(self, email: str, password: str) -> None:  # done
        try:
            self.backend.user_service.sign_up(email, password)
        except ValueError as e:
            print(e)

    @signed_out
    @recommends
    def sign_in(self, email: str, password: str) -> None:  # done
        try:
            self.user = self.backend.user_service.sign_in(email, password)
        except ValueError as e:
            print(e)

    @signed_in
    def sign_out(self) -> None:  # done
        self.user = None

    @signed_in
    @recommends
    def upload(self, url: str, rating: int) -> None:
        user_id = self.user.user_id
        image_id = self.backend.image_service.add_image(url)
        self.backend.review_service.review(user_id, image_id, rating)

    @signed_in
    @recommends
    def review(self, rating: int) -> None:
        if self.current_image is None:
            print("No image to review")
            return

        user_id = self.user.user_id
        image_id = self.current_image.image_id
        self.backend.review_service.review(user_id, image_id, rating)
        self.current_image = self.backend.recommender_service.recommend(user_id)

    def _recommend(self) -> None:
        if self.user is None:
            return

        user_id = self.user.user_id
        self.current_image = self.backend.recommender_service.recommend(user_id)
        if self.current_image is None:
            print("No images to review - try uploading one")
            return
        print("Please rate the following image:", self.current_image.url)


def main() -> None:
    frontend = Frontend()

    instructions = "Commands:\n- sign_up <email> <password>\n- sign_in <email> <password>\n- sign_out\n- upload <url> <rating>\n- review <rating>"
    print(instructions)

    while True:
        req = input().split()

        if req[0] == "sign_up" and len(req) == 3:
            frontend.sign_up(req[1], req[2])
        elif req[0] == "sign_in" and len(req) == 3:
            frontend.sign_in(req[1], req[2])
        elif req[0] == "sign_out" and len(req) == 1:
            frontend.sign_out()
        elif req[0] == "upload" and len(req) == 3:
            if not (req[2].isdigit() and 1 <= int(req[2]) <= 5):
                print("Invalid rating")
                continue
            frontend.upload(req[1], int(req[2]))
        elif req[0] == "review" and len(req) == 2:
            if not (req[1].isdigit() and 1 <= int(req[1]) <= 5):
                print("Invalid rating")
                continue
            frontend.review(int(req[1]))
        else:
            print("Unrecognized command")


if __name__ == "__main__":
    main()
