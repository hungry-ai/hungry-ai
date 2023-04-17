import numpy as np
import pytest

from src.tags import WordEmbedding, PytorchWordEmbedding


def test_word_embedding() -> None:
    word_embedding_1 = WordEmbedding(dimension=5)
    assert 5 == word_embedding_1.dimension
    assert 0 == len(word_embedding_1)

    word_embedding_1["A"] = np.array([1, 0, 0, 0, 0])
    word_embedding_1["B"] = np.array([0, 1, 0, 0, 0])
    word_embedding_1["C"] = np.array([0, 0, 1, 0, 0])
    word_embedding_1["Hello"] = np.array([0, 0, 0, 1, 0])
    word_embedding_1["Bonjour"] = np.array([0, 0, 0, 0, 1])
    word_embedding_1["Nihao"] = np.array([1, 0, 0, 0, 1])
    word_embedding_1["Konichiwa"] = np.array([1, 0, 1, 0, 0])
    assert 7 == len(word_embedding_1)

    word_vector = word_embedding_1["Hello"]
    np.testing.assert_array_equal(word_vector, np.array([0, 0, 0, 1, 0]))
    with pytest.raises(KeyError):
        word_embedding_1["Wassup"]


def test_pytorch_word_embedding() -> None:
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
    # Change every word to lower case.
    for i in range(len(words)):
        words[i] = words[i].lower()
    # Get rid of duplicates.
    words = list(set(words))
    
    word_embedding_1 = PytorchWordEmbedding(words, dimension = 50)
    assert(50 == word_embedding_1.dimension)
    
    assert(50 == len(word_embedding_1["beef"]))
    assert(50 == len(word_embedding_1["turkey"]))
    assert(50 == len(word_embedding_1["broccoli"]))
    assert(50 == len(word_embedding_1["salmon"]))
    assert(50 == len(word_embedding_1["tuna"]))
    assert(50 == len(word_embedding_1["barbecue"]))
    
    with pytest.raises(KeyError):
        word_embedding_1["Wassup"]

