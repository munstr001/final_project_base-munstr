import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def train_item_similarity(ratings: pd.DataFrame):
    pivot = ratings.pivot_table(
        index="material_id",
        columns="student_id",
        values="rating"
    ).fillna(0)

    similarity = cosine_similarity(pivot)
    sim_df = pd.DataFrame(
        similarity,
        index=pivot.index,
        columns=pivot.index
    )
    return sim_df


def recommend_items(material_id, sim_df, top_k=3):
    return (
        sim_df[material_id]
        .sort_values(ascending=False)
        .iloc[1:top_k+1]
        .index
        .tolist()
    )

def recommend_for_student(student_id, ratings, sim_df, top_k=3):
    student_items = ratings[ratings["student_id"] == student_id]["material_id"].tolist()

    scores = {}
    for item in student_items:
        for similar_item, score in sim_df[item].items():
            if similar_item not in student_items:
                scores[similar_item] = scores.get(similar_item, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
