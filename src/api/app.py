from fastapi import FastAPI
import pandas as pd

from src.recommender.collaborative import train_item_similarity, recommend_items

app = FastAPI()

ratings = pd.read_csv("data/ratings.csv")
sim_df = train_item_similarity(ratings)

@app.get("/recommend/{material_id}")
def recommend(material_id: int):
    recs = recommend_items(material_id, sim_df)
    return {"recommended_materials": recs}
