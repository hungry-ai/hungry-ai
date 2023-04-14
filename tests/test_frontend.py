from pathlib import Path

import pytest
from src.frontend import Frontend

EMAIL = "cody@gmail.com"


def test_permissions(frontend: Frontend, url: str) -> None:
    assert frontend.sign_out() == "You're not signed in"
    assert frontend.upload(url, 5) == "You're not signed in"
    assert frontend.review(5) == "You're not signed in"

    frontend.sign_up(EMAIL, "a")
    frontend.sign_in(EMAIL, "a")

    assert frontend.sign_up(EMAIL, "a") == "You're already signed in"
    assert frontend.sign_in(EMAIL, "a") == "You're already signed in"


def test_sign_up(frontend: Frontend) -> None:
    assert frontend.sign_up(EMAIL, "a") is None

    frontend.sign_out()

    assert frontend.sign_up(EMAIL, "a") == "user with this email already exists"


def test_sign_in(frontend: Frontend) -> None:
    assert frontend.sign_in(EMAIL, "a") == "no user with this email"

    frontend.sign_up(EMAIL, "a")

    assert frontend.sign_in(EMAIL, "a") == frontend._recommend()

    frontend.sign_out()

    assert frontend.sign_in(EMAIL, "a") == frontend._recommend()


def test_sign_out(frontend: Frontend) -> None:
    frontend.sign_up(EMAIL, "a")
    frontend.sign_in(EMAIL, "a")

    assert frontend.sign_out() is None


def test_upload(frontend: Frontend, url: str) -> None:
    frontend.sign_up(EMAIL, "a")
    frontend.sign_in(EMAIL, "a")

    assert frontend.upload(url, 0.5) == "invalid rating"
    assert frontend.upload(url, 6.0) == "invalid rating"
    frontend.upload(url, 5.0)


def test_review(frontend: Frontend, url: str) -> None:
    frontend.sign_up(EMAIL, "a")
    frontend.sign_in(EMAIL, "a")

    assert frontend.review(5.0) == "No image to review"

    frontend.upload(url, 5.0)

    with pytest.raises(NotImplementedError):
        frontend.review(5.0)
