from fastapi import FastAPI, HTTPException
import pandas as pd

from src.recommender.collaborative import train_item_similarity, recommend_for_student
from src.recommender.hybrid import build_recommendation

app = FastAPI()

ratings = pd.read_csv("data/ratings.csv")
materials = pd.read_csv("data/materials.csv")
sim_df = train_item_similarity(ratings)

@app.get("/recommend/student/{student_id}")
def recommend_student(student_id: int):
    if student_id not in ratings["student_id"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    recs = recommend_for_student(student_id, ratings, sim_df)

    response = []
    for material_id, score in recs:
        material = materials[materials["material_id"] == material_id].iloc[0]
        response.append(
            build_recommendation(
                material,
                score,
                "Recommended based on similar students learning patterns"
            )
        )

    return {
        "student_id": student_id,
        "recommendations": response
    }
