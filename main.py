from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import os
import torch

# Parametreler
SIMILARITY_THRESHOLD = 0.7
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_FILE = f"{MODEL_NAME.split('/')[-1]}_embeddings.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükle
model = SentenceTransformer(MODEL_NAME, device=device)

# JSON dosyasını oku
try:
    with open('data.json', 'r') as file:
        data = json.load(file)
    questions = [item['question'] for item in data]
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Veri dosyası hatalı: {e}")
    exit(1)

# Embedding yükle veya oluştur
if os.path.exists(EMBEDDING_FILE):
    question_embeddings_np = np.load(EMBEDDING_FILE)
else:
    print("İlk çalıştırma: embedding'ler hesaplanıyor, bu işlem sadece bir kez yapılır.")
    question_embeddings_np = model.encode(questions, convert_to_numpy=True)
    with open(EMBEDDING_FILE, 'wb') as f:
        np.save(f, question_embeddings_np)

# NumPy'den torch tensor'e ve cihaza gönder
question_embeddings = torch.tensor(question_embeddings_np).to(device)

# Soru cevap döngüsü
while True:
    print("Programdan çıkmak için 'cikis' yazınız.")
    user_input = input("Soru girin: ").strip()
    if user_input.lower() == "cikis":
        print("Programdan çıkılıyor...")
        break

    # Soruyu encode et
    user_embedding = model.encode(user_input, convert_to_tensor=True).to(device)
    cosine_scores = util.cos_sim(user_embedding, question_embeddings)[0]

    # En yüksek skor kontrolü
    max_score = torch.max(cosine_scores).item()
    if max_score > SIMILARITY_THRESHOLD:
        best_idx = torch.argmax(cosine_scores).item()
        matched_answer = data[best_idx]['answer']
        print(f"Cevap: {matched_answer} (Benzerlik skoru: {max_score:.4f})")
    else:
        print("Buna cevap verebilecek kadar geniş bir bilgi ağım yok. ")
        print(f"En yüksek skor: {max_score:.4f}")
