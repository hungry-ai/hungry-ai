from ..graph import Graph
from .word_embeddings import PytorchWordEmbedding, WordEmbedding
from src.graph.vertex import Vertex


def generate_tags_graph(word_embedding: WordEmbedding, graph: Graph) -> dict[str, Vertex]:
    vertices = {word: graph.add_tag(word) for word in word_embedding}

    if len(word_embedding) <= 1:
        return

    for word, vertex in vertices.items():
        neighbors = [w for w in word_embedding if w != word]
        nearest_neighbor = min(
            neighbors,
            key=lambda w: word_embedding.distance(word, w),
        )
        distance = word_embedding.distance(word, nearest_neighbor)

        graph.add_edge(vertex, vertices[nearest_neighbor], distance)
    
    return vertices

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from ..graph import CSVGraph, visualize
    from ..recommender import RecommenderService

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=".")
    args = parser.parse_args()

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

    word_embedding = PytorchWordEmbedding(words, dimension=50)

    graph = CSVGraph(root=args.root)
    generate_tags(word_embedding, graph)
    print(len(graph.vertices))

    zozo = graph.add_user("Zozo")
    ramen = graph.add_image("Ramen Image")
    juice = graph.add_tag("juice")
    graph.add_edge(zozo, juice, 3.0)
    graph.add_edge(juice, ramen, 2.0)
    raise NotImplementedError
    # result = get_recommendations(graph, zozo.label, 1)
    # print(result)

    visualize(graph, "visualize_scaled.html")
    visualize(graph, "visualize_unscaled.html", scaled=False)
