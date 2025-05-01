from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

text1 = "I love reading books on machine learning and artificial intelligence."
text2 = "My favorite hobby is exploring topics like AI and ML through books."

embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2)

print(f"Cosine Similarity: {similarity.item():.4f}")
