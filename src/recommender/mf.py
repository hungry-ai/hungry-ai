import numpy as np

from ..images import Image, pr_match
from ..reviews import Review
from ..users import User
from .recommender import Recommender


class MFRecommender(Recommender):
    def __init__(
        self,
        Y: np.ndarray,
        tags: list[str],
        alpha: float,
        x_avg: np.ndarray,
        n_train: int,
    ) -> None:
        self.Y = Y
        self.tags = tags
        self.alpha = alpha

        self.d = Y.shape[1]

        self.X: dict[str, np.ndarray] = {}  # user_id -> x
        self.x_avg = x_avg
        self.n_train = n_train

        self.IY: dict[str, np.ndarray] = {}  # image_id -> i @ Y
        self.YTIuTIuY: dict[
            str, np.ndarray
        ] = {}  # user_id -> (I_u Y).T (I_u Y) + (alpha nnz / d) I_d
        self.YTIuTru: dict[str, np.ndarray] = {}  # user_id -> (I_u Y).T r_u

    def add_user(self, user: User) -> None:
        if user.id in self.X:
            return

        self.X[user.id] = self.x_avg
        self.YTIuTIuY[user.id] = np.zeros((self.d, self.d))
        self.YTIuTru[user.id] = np.zeros(self.d)

    def add_image(self, image: Image) -> None:
        tag_prs = [
            (self.Y[tag_index], pr_match(image.url, tag))
            for tag_index, tag in enumerate(self.tags)
        ]
        ys, i = zip(*tag_prs)

        # compute iy
        iy = np.array(ys) @ np.array(i)

        # update iy
        self.IY[image.id] = iy

    def add_review(self, review: Review) -> None:
        user = review.user
        image = review.image

        iy = self.IY[image.id]

        # compute x
        A = self.YTIuTIuY[user.id]
        A += iy.reshape(-1, 1) @ iy.reshape(1, -1)
        A += self.alpha / self.d * np.eye(self.d)
        b = self.YTIuTru[user.id]
        b += review.rating * iy

        x = np.linalg.solve(A, b)

        # update x
        self.x_avg += (x - self.X[user.id]) / (len(self.x_avg) + self.n_train)
        self.X[user.id] = x

    def predict_rating(self, user_id: str, image_id: str) -> float:
        x = self.X[user_id]
        iy = self.IY[image_id]
        return x @ iy

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        raise NotImplementedError
