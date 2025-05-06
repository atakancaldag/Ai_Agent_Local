from sentence_transformers import SentenceTransformer, util
import json

with open('data.json', 'r') as file:
    data = json.load(file)
questions = [item['question'] for item in data]
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions, convert_to_tensor=True)
user_question = input("Soru girin: ")
user_embedding = model.encode(user_question, convert_to_tensor=True)
cosine_scores = util.cos_sim(user_embedding, question_embeddings)[0]  # shape: [num_questions]
if max(cosine_scores) > 0.7:
    best_match_idx = cosine_scores.argmax()
    best_score = cosine_scores[best_match_idx].item()

    matched_question = questions[best_match_idx]
    matched_answer = data[best_match_idx]['answer']

    print(f"Cevap: {matched_answer} (Benzerlik skoru: {best_score:.4f})")
else:
    print('Buna cevap verebilecek kadar genis bir bilgi agim yok.')
    print(max(cosine_scores))