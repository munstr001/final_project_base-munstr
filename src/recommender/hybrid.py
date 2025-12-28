def hybrid_recommend(collab, content):
    return list(dict.fromkeys(collab + content))

def build_recommendation(material, score, reason):
    return {
        "material_id": int(material["material_id"]),
        "title": material["title"],
        "topic": material["topic"],
        "difficulty": int(material["difficulty"]),
        "score": round(score, 3),
        "reason": reason
    }
