import heapq
import warnings
import datetime

from ..graph import Graph, Vertex, VertexType, LocalGraph, visualize

from ..images import Image, pr_match
from ..reviews import Review
from ..tags import Tag, PytorchWordEmbedding, WordEmbedding, generate_tags_graph
from ..users import User
from .recommender import Recommender
from .knn import KNNRecommender

class KNNRecommenderLite(Recommender):
    def __init__(self, initial_word_embedding) -> None:
        self.word_embedding = initial_word_embedding
        graph = LocalGraph()
        self.tag_vtxs = generate_tags_graph(self.word_embedding, graph)
        tags = [Tag(id=word, name=word) for i, word in enumerate(self.word_embedding)]
        self.knn_recommender = KNNRecommender(graph, tags)

    def add_user(self, user: User) -> None:
        return self.knn_recommender.add_user(user)

    def add_image(self, image: Image) -> None:
        self.knn_recommender.add_image(image)

    def add_review(self, review: Review) -> None:
        self.knn_recommender.add_review(review)

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        self.knn_recommender.get_recommendations(user, num_recs)

def main():
    words = []
    # Read file contents into a string
    with open('src/recommender/food_words.txt', 'r') as f:
        file_contents = f.read()

    # Split string by spaces and commas, and convert to list
    words_list = file_contents.replace(',', ' ').split()

    # Add words from words_list to existing_list
    words.extend(words_list)

    for i in range(len(words)):
        words[i] = words[i].lower()
    words = list(set(words))

    word_embedding = PytorchWordEmbedding(words, dimension=50)
    recommender = KNNRecommenderLite(word_embedding)

    user_1 = User(id = "Alex", instagram_username = "alex_zheng_7")
    user_2 = User(id = "Cody", instagram_username = "codercody")
    user_3 = User(id = "Nozomi", instagram_username = "nozozo")

    image_1 = Image(id = "Ramen Nagi", url = "blah1")
    image_2 = Image(id = "Ramen Miso", url = "blah2")
    image_3 = Image(id = "Ramen Tokyo", url = "blah3")
    image_4 = Image(id = "Ramen Pho", url = "blah4")
    image_5 = Image(id = "Ramen Udon", url = "blah5")

    recommender.add_user(user_1)
    recommender.add_user(user_2)
    recommender.add_user(user_3)
    recommender.add_image(image_1)
    recommender.add_image(image_2)
    recommender.add_image(image_3)
    recommender.add_image(image_4)
    recommender.add_image(image_5)

    recommender.add_review(Review(id = "1", user = user_1, image = image_1, rating = 4.0, timestamp = datetime.datetime.now()))
    recommender.add_review(Review(id = "2", user = user_2, image = image_1, rating = 3.0, timestamp = datetime.datetime.now()))
    recommender.add_review(Review(id = "3", user = user_3, image = image_2, rating = 5.0, timestamp = datetime.datetime.now()))
    recommender.add_review(Review(id = "4", user = user_1, image = image_1, rating = 2.0, timestamp = datetime.datetime.now()))
    recommender.add_review(Review(id = "5", user = user_2, image = image_3, rating = 1.0, timestamp = datetime.datetime.now()))


    graph = recommender.knn_recommender.graph
    vertices = recommender.tag_vtxs

    labels = dict()
    for key in vertices:
        labels[vertices[key]] = key
    print(len(graph.vertices))

    visualize(graph, labels, "visualize_scaled.html")
    visualize(graph, labels, "visualize_unscaled.html", scaled=False)


if __name__ == "__main__":
    main()
