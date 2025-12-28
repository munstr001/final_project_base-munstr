def hybrid_recommend(collab, content):
    return list(dict.fromkeys(collab + content))
