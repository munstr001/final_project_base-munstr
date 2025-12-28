from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_content_model(materials):
    materials["text"] = materials["topic"] + " " + materials["difficulty"].astype(str)
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(materials["text"])
    similarity = cosine_similarity(matrix)
    return similarity


def recommend_content(material_id, materials, similarity, top_k=3):
    idx = materials.index[materials["material_id"] == material_id][0]
    scores = similarity[idx]
    indices = scores.argsort()[::-1][1:top_k+1]
    return materials.iloc[indices]["material_id"].tolist()
