import pandas as pd
from src.recommender.collaborative import train_item_similarity

ratings = pd.read_csv("data/ratings.csv")
sim = train_item_similarity(ratings)

sim.to_csv("models/item_similarity.csv")
print("Model retrained")
