import pandas as pd
from src.recommender.collaborative import train_item_similarity, recommend_items

def test_recommend_items():
    ratings = pd.DataFrame({
        "student_id": [1, 1, 2, 2],
        "material_id": [1, 2, 1, 3],
        "rating": [5, 4, 4, 5]
    })

    sim = train_item_similarity(ratings)
    recs = recommend_items(1, sim, top_k=2)

    assert isinstance(recs, list)
    assert len(recs) == 2
