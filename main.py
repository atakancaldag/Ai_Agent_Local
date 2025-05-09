from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import os
<<<<<<< HEAD
import openai
from openai import AuthenticationError, OpenAIError


model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dosyasi = 'question_embeddings.npy'
chatgpt_model = "gpt-3.5-turbo"

def Embed():
    try:
        with open(embedding_dosyasi, "rb") as f:
            question_embeddings = np.load(f)
            return question_embeddings
    except FileNotFoundError:
        print("Ilk sefer calistirildiginden dolayi sürec biraz uzun sürebilir... Bu sürec tek seferliktir.")
        question_embeddings = model.encode(questions, convert_to_numpy=True)
        with open(embedding_dosyasi, "wb") as f:
            np.save(f, question_embeddings)
        return question_embeddings

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True



def load_api_key(path='openai_api.json'): 
    """
    Sorulari openaiye yollamak için openai_api.json diye dosyaya bakiyor, içinde key var mi diye bakiyor, dosya yoksa oluşturuyor
    Sonrasinda api key kontrolü yapiyor, yanlişsa dosyadakini siliyor, kullanicidan yenisini isteyip dosyaya kaydediyor
     """
    while True:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            key = input("OpenAI API anahtarinizi girin: ")
            with open(path, "w") as f:
                json.dump({"api_key": key.strip()}, f)

        with open(path, "r") as f:
            try:
                config = json.load(f)
                key = config.get("api_key") or config.get("api_id")
                openai.api_key = key.strip()
            except json.JSONDecodeError:
                key = None
        if not key:
            print("API anahtarı bulunamadı veya okunamadı. Tekrar deneyin.")
            os.remove(path)
            continue
        if check_openai_api_key(openai.api_key):
            print("API anahtari dogru, devam ediliyor...")
            break
        else:
            print("API anahtari yanlis, tekrar deneyin.")
            os.remove(path)
            continue


with open('data.json', 'r') as file:
    data = json.load(file)
questions = [item['question'] for item in data]

question_embeddings = Embed()

load_api_key()

def ask_openai(prompt, model_name=chatgpt_model):
    """
    Adi üstünde, benzerlik yetmediyse eger, soruyu chatgptye tükürüyor
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

while True:
    print("Programdan cikis yapmak için 'cikis' yaziniz")
    user_question = input("Soru girin: ")
    if user_question.lower() == 'cikis':
        print("Programdan cikiliyor...")
        break
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    if max(cosine_scores) > 0.7:
        best_match_idx = cosine_scores.argmax()
        best_score = cosine_scores[best_match_idx].item()

        matched_question = questions[best_match_idx]
        matched_answer = data[best_match_idx]['answer']

        print(f"Cevap: {matched_answer} (Benzerlik skoru: {best_score:.4f})")
    else:
        print("Bu sorunun cevabi bana ogretilmedi, ChatGPT'ye soruyorum...")
        ai_answer = ask_openai(user_question)
        print(ai_answer)
=======
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
>>>>>>> ecea86bc30638011358d74946ec73eaa2b0db2b8
