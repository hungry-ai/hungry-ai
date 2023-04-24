import heapq
import warnings
import datetime

from ..graph import Graph, Vertex, VertexType
from ..images import Image, pr_match
from ..reviews import Review
from ..tags import Tag, PytorchWordEmbedding, WordEmbedding, generate_tags_graph
from ..users import User
from .recommender import Recommender
from .knn_recommender import KNNRecommender

class KNNRecommenderLite(Recommender):
    def __init__(self, initial_word_embedding) -> None:
        self.word_embedding = initial_word_embedding
        graph = LocalGraph()
        self.tag_vtxs = generate_tags_graph(self.word_embedding, graph)
        tags = [Tag(id=word, name=word) for i, word in enumerate(word_embedding)]
        self.knn_recommender = KNNRecommender(graph, tags)

    def add_user(self, user: User) -> None:
        return self.knn_recommender.add_user(user)

    def add_image(self, image: Image) -> None:
        self.knn_recommender.add_image(image)

    def add_review(self, review: Review) -> None:
        self.knn_recommender.add_review(review)

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        self.knn_recommender.get_recommendations(user, num_recs)

if __name__ == "__main__":
    from ..graph import LocalGraph, visualize

    words = [
        # fmt: off
        "Food", "Cuisine", "Taste", "Delicious", "Meal", "Recipe", "Cooking", "Beverage", "Gourmet", "Flavor",
        "Dish", "Cuisines", "Ingredient", "Tasting", "Nourishment", "Gastronomy", "Feast", "Spice", "Savor",
        "Culinary", "Seasoning", "Satisfaction", "Savory", "Aroma", "Feasting", "Entree", "Gourmand", "Munch",
        "Savoriness", "Soup", "Satisfying", "Hunger", "Appetite", "Fruit", "Vegetable", "Meat", "Seafood", "Poultry",
        "Bread", "Rice", "Noodle", "Pasta", "Cheese", "Herb", "Condiment", "Dessert", "Bake", "Roast", "Grill", "Fry",
        "Boil", "Steam", "Sauté", "Braise", "Marinate", "Barbecue", "Baking", "Grilling", "Frying", "Boiling", "Steaming",
        "Sauting", "Braising", "Marinade", "Barbecuing", "Oven", "Stove", "Cookware", "Cutlery", "Tableware", "Spoon", "Fork",
        "Knife", "Chopsticks", "Mug", "Glass", "Plate", "Bowl", "Cup", "Canape", "Hors d'oeuvre", "Appetizer", "Starter",
        "Main course", "Entrée", "Side dish", "Salad", "Soup", "Bread", "Dessert", "Cake", "Pie", "Pastry", "Ice cream", "Fruit",
        "Candy", "Chocolate", "Beverage", "Juice", "Tea", "Coffee", "Wine", "Beer", "Liquor", "Alcohol", "Breakfast", "Lunch",
        "Dinner", "Supper", "Snack", "Buffet", "Banquet", "Feast", "Picnic", "Potluck", "BBQ", "Cookout", "Foodie", "Gourmet",
        "Connoisseur", "Chef", "Cook", "Baker", "Waiter", "Waitress", "Host", "Hostess", "spicy", "sweet", "sour", "bitter",
        "savory", "garlic", "herbs", "spices", "sugar", "salt", "pepper", "vanilla", "chocolate", "cheese", "butter", "oil",
        "vinegar", "lemon", "lime", "mayonnaise", "ketchup", "mustard", "barbecue", "pesto", "honey", "soy sauce", "hot sauce",
        "teriyaki", "oyster sauce", "wasabi", "ginger", "coriander", "cumin", "cinnamon", "nutmeg", "cloves", "allspice",
        "cardamom", "turmeric", "fenugreek", "curry", "paprika", "chili powder", "black pepper", "white pepper",
        "red pepper flakes", "oregano", "basil", "rosemary", "thyme", "sage", "mint", "basmati rice", "pasta", "bread",
        "potatoes", "carrots", "onions", "celery", "tomatoes", "bell peppers", "cucumber", "lettuce", "spinach", "kale",
        "beets", "radishes", "mushrooms", "zucchini", "squash", "eggplant", "asparagus", "broccoli", "cauliflower",
        "brussels sprouts", "green beans", "peas", "corn", "avocado", "mango", "banana", "apple", "orange", "grapes",
        "strawberries", "blueberries", "raspberries", "blackberries", "cherries", "peaches", "plums", "apricots", "pomegranate",
        "kiwi", "pineapple", "coconut", "almonds", "pecans", "walnuts", "cashews", "peanuts", "macadamia nuts", "pistachios",
        "hazelnuts", "beef", "pork", "chicken", "turkey", "duck", "lamb", "veal", "bacon", "sausage", "ham", "salmon", "tuna",
        "cod", "halibut", "shrimp", "crab", "lobster", "oysters", "clams", "mussels", "scallops"
        # fmt: on
    ]

    # # Read file contents into a string
    # with open('src/recommender/food_words.txt', 'r') as f:
    #     file_contents = f.read()

    # # Split string by spaces and commas, and convert to list
    # words_list = file_contents.replace(',', ' ').split()

    # # Add words from words_list to existing_list
    # words.extend(words_list)

    # for i in range(len(words)):
    #     words[i] = words[i].lower()
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
