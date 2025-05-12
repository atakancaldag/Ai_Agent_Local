import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import openai
from openai.types.chat import ChatCompletionMessageParam

class QASystem:
    def __init__(self,
                 data_path='data.json',
                 embedding_file='all-MiniLM-L6-v2_embeddings.npy',
                 model_name="all-MiniLM-L6-v2",
                 similarity_threshold=0.7,
                 api_key_path='openai_api.json',
                 chatgpt_model="gpt-3.5-turbo"):
        """
        Soru-cevap sistemini başlatır ve gerekli tüm bileşenleri yükler.

        Args:
            data_path (str): JSON formatındaki soru-cevap veri dosyasının yolu.
            embedding_file (str): Embedding verilerinin kaydedileceği/yükleneceği dosya.
            model_name (str): Kullanılacak SentenceTransformer modeli.
            similarity_threshold (float): Benzerlik eşiği (0-1 arası).
            api_key_path (str): OpenAI API anahtarının saklandığı dosya yolu.
            chatgpt_model (str): Kullanılacak ChatGPT model adı.
        """
        self.data_path = data_path
        self.embedding_file = embedding_file
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.api_key_path = api_key_path
        self.chatgpt_model = chatgpt_model

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        self.data = []
        self.questions = []
        self.question_embeddings = None

        self.load_data()
        self.embed_questions()
        self.load_openai_key()

    def load_data(self):
        """
        data_path içerisindeki JSON dosyasını yükler ve soruları belleğe alır.
        """
        try:
            with open(self.data_path, 'r') as file:
                self.data = json.load(file)
                self.questions = [item['question'] for item in self.data]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Veri dosyası hatalı veya eksik: {e}")
            exit(1)

    def embed_questions(self):
        """
        Soruların embedding'lerini yükler veya oluşturur ve belleğe kaydeder.
        """
        if os.path.exists(self.embedding_file):
            self.question_embeddings = torch.tensor(np.load(self.embedding_file)).to(self.device)
        else:
            print("İlk çalıştırma: embedding'ler hesaplanıyor...")
            embeddings_np = self.model.encode(self.questions, convert_to_numpy=True)
            with open(self.embedding_file, 'wb') as f:
                np.save(f, embeddings_np)
            self.question_embeddings = torch.tensor(embeddings_np).to(self.device)

    def load_openai_key(self):
        """
        OpenAI API anahtarını dosyadan yükler veya kullanıcıdan ister. Anahtar geçersizse tekrar ister.
        """
        while True:
            if not os.path.exists(self.api_key_path) or os.path.getsize(self.api_key_path) == 0:
                key = input("OpenAI API anahtarınızı girin: ").strip()
                with open(self.api_key_path, "w") as f:
                    json.dump({"api_key": key}, f)

            with open(self.api_key_path, "r") as f:
                try:
                    config = json.load(f)
                    key = config.get("api_key")
                    openai.api_key = key
                except json.JSONDecodeError:
                    key = None

            if not key:
                print("API anahtarı okunamadı. Tekrar giriniz.")
                os.remove(self.api_key_path)
                continue

            if self.check_openai_api_key(key):
                print("API anahtarı doğru.")
                break
            else:
                print("API anahtarı geçersiz. Lütfen tekrar girin.")
                os.remove(self.api_key_path)

    @staticmethod
    def check_openai_api_key(api_key):
        """
        OpenAI API anahtarının geçerli olup olmadığını kontrol eder.

        Args:
            api_key (str): Test edilecek API anahtarı.

        Returns:
            bool: Anahtar geçerliyse True, değilse False.
        """
        try:
            client = OpenAI(api_key=api_key)
            client.models.list()
            return True
        except openai.AuthenticationError:
            return False

    def ask_openai(self, prompt):
        """
        OpenAI ChatGPT API'sini kullanarak kullanıcıdan gelen soruya yanıt alır.

        Args:
            prompt (str): Kullanıcıdan gelen soru.

        Returns:
            str: ChatGPT'den gelen yanıt veya hata mesajı.
        """
        try:
            client = OpenAI(api_key=openai.api_key)
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system",
                 "content": "You are a Turkish coding assistant specialized in machine learning and answering only machine learning related questions."},
                {"role": "user", "content": prompt}
            ] # type: ignore
            response = client.chat.completions.create(
                model=self.chatgpt_model,
                messages=messages,
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ChatGPT API hatası: {str(e)}"

    def find_best_match(self, user_question):
        """
        Kullanıcının sorduğu soruya en benzer soruyu veri kümesinde bulur.

        Args:
            user_question (str): Kullanıcının girdiği soru.

        Returns:
            tuple: (bulunan en iyi cevap veya None, benzerlik skoru)
        """
        user_embedding = self.model.encode(user_question, convert_to_tensor=True).to(self.device)
        cosine_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]

        max_score = torch.max(cosine_scores).item()
        if max_score >= self.similarity_threshold:
            best_idx = torch.argmax(cosine_scores).item()
            return self.data[best_idx]['answer'], max_score
        else:
            return None, max_score

    def run(self):
        """
        Ana uygulama döngüsünü başlatır. Kullanıcıdan soru alır, eşleşme varsa gösterir;
        yoksa OpenAI API'ye soruyu gönderir.
        """
        print("Soru cevap sistemine hoş geldiniz. Çıkmak için 'cikis' yazınız.")
        while True:
            user_input = input("Soru girin: ").strip()
            if user_input.lower() == "cikis":
                print("Programdan çıkılıyor...")
                break

            answer, score = self.find_best_match(user_input)
            if answer:
                print(f"Cevap: {answer} (Benzerlik skoru: {score:.4f})")
            else:
                print("Cevap verilemiyor, ChatGPT'den cevap alınıyor...")
                ai_answer = self.ask_openai(user_input)
                print(f"ChatGPT cevabı: {ai_answer} (Skor: {score:.4f})")


if __name__ == "__main__":
    qa_system = QASystem()
    qa_system.run()
