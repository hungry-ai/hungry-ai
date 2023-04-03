import torchtext
from word_embedding.word_embedding_basic import WordEmbeddingBasic
import topics_utils
import topics_graph.graph_txt as graph_txt
from graph_visualization import graph_visualization

def graph_visualization_example():
    '''
    This function runs graph_visualization on 4 different inputs. 
    It takes a hot minute to run...
    '''
    words = ["Food", "Cuisine", "Taste", "Delicious", "Meal", "Recipe", "Cooking", "Beverage", "Gourmet", "Flavor", 
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
         "cod", "halibut", "shrimp", "crab", "lobster", "oysters", "clams", "mussels", "scallops"]
    
    graph_visualization(words, 'from_words_list')

    words = list(set(words))
    # Load the pre-trained word embeddings
    glove = torchtext.vocab.GloVe(name='6B', dim=50)
    word_embedding_basic_1 = WordEmbeddingBasic(50)
    words_kept = 0
    for word in words:
        word = word.lower()
        if word in glove.stoi:
            word_embedding = glove.vectors[glove.stoi[word.lower()]]
            word_embedding_list = word_embedding.numpy().tolist()
            word_embedding_basic_1.add_word_vector((word, word_embedding_list))
            words_kept += 1
    print("type of word embedding: ", type(word_embedding_basic_1))
    print("Number of words kept: " + str(words_kept))

    graph_visualization(word_embedding_basic_1, 'from_word_embed')

    graph_txt_1 = graph_txt.GraphTXT()
    topics_utils.generate_topics_graph(word_embedding_basic_1, graph_txt_1)
    print(graph_txt_1.number_vertices)

    graph_visualization(graph_txt_1, 'from_graph')

    graph_txt_1.graph_to_file()

    graph_visualization('sample.txt', 'from_file')

graph_visualization_example()

