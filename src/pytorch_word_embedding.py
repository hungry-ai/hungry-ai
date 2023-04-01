import torch
import torchtext
from word_embedding.word_embedding_basic import WordEmbeddingBasic
import topics_utils
import topics_graph.graph_txt as graph_txt
from pyvis.network import Network
import networkx as nx

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

for i in range(len(words)):
    words[i] = words[i].lower()

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

print("Number of words kept: " + str(words_kept))
graph_txt_1 = graph_txt.GraphTXT()
topics_utils.generate_topics_graph(word_embedding_basic_1, graph_txt_1)
print(graph_txt_1.number_vertices)

graph_txt_1.graph_to_file()
graph_txt_2 = graph_txt.GraphTXT("sample.txt")

net = nx.DiGraph()

for i in range(graph_txt_1.number_of_vertices()):
    vertex = graph_txt_1.get_vertex(i)
    net.add_node(vertex.word)
for i in range(graph_txt_1.number_of_edges()):
    edge = graph_txt_1.get_edge(i)
    net.add_edge(edge.vertex_out, edge.vertex_in, arrows = "to")

scale = 10 # size ~= 10 * in-degree
d = dict(net.in_degree)
d.update((x,5+scale*y) for x,y in d.items()) # nodes with in-degree 0 have size 5
nx.set_node_attributes(net, d, 'size')

G = Network(notebook=True)
G.from_nx(net)




G.show("vis_scaled.html") # Node size scales with in-degree, label is outside circle

for n in G.nodes:
    n['shape'] = 'circle' # This forces pyvis to put label inside circle

G.show("vis_unscaled.html") # Label is inside circle, size overrided to fit label