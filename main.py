import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import openai
from openai.types.chat import ChatCompletionMessageParam
from pyairtable import Api
from dotenv import load_dotenv
load_dotenv()

class QASystem:
    def __init__(self,
                 airtable_api_key,
                 airtable_base_id,
                 airtable_table_name,
                 model_name="all-MiniLM-L6-v2",
                 similarity_threshold=0.8,
                 api_key_path='.env',
                 keywords_path='keywords.json',
                 stopwords_file='keywords.json',
                 chatgpt_model="gpt-3.5-turbo"):

        self.airtable_api_key = airtable_api_key
        self.airtable_base_id = airtable_base_id
        self.airtable_table_name = airtable_table_name

        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.ml_keywords = keywords_path
        self.stopwords_file = stopwords_file
        self.chatgpt_model = chatgpt_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.api = Api(self.airtable_api_key)
        self.airtable = self.api.table(self.airtable_base_id, self.airtable_table_name)

        self.data = []
        self.questions = []
        self.answers = []
        self.embeddings = []
        self.load_openai_key()
        self.load_data_and_embeddings()

    def load_stopwords(self):
        with open(self.stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = json.load(f)
        return set([sw.lower() for sw in stopwords])

    def is_about_ml(self, text: str) -> bool:
        """
        Metni makine öğrenmesiyle ilgili olup olmadığını
        belirlemek için yerel json dosyasındaki anahtar kelimelerle kontrol eder.

        Args:
            text (str): Kontrol edilecek kullanıcı girişi.

        Returns:
            bool: Metin makine öğrenmesi konusuysa True, değilse False.
        """
        try:
            with open(self.ml_keywords, "r", encoding="utf-8") as f:
                ml_keywords = json.load(f)
        except Exception as e:
            print(f"ml_keywords.json dosyası okunurken hata oluştu: {e}")
            return False

        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in ml_keywords)

    def load_openai_key(self):
        """
        Loads the OpenAI API key from environment variables.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Environment variable 'OPENAI_API_KEY' is missing.")
        openai.api_key = api_key
        if not self.check_openai_api_key(api_key):
            raise ValueError("Invalid OpenAI API key.")

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
        except Exception:
            return False

    def load_data_and_embeddings(self):
        """
        Airtable'dan verileri çeker, eksik embedding varsa hesaplar, hem localde hem Airtable'da günceller.
        """
        print("Airtable'dan veriler çekiliyor...")
        records = self.airtable.all()
        self.data.clear()
        self.questions.clear()
        self.answers.clear()
        self.embeddings.clear()

        update_needed = []
        for record in records:
            fields = record.get("fields", {})
            question = fields.get("question", "").strip()
            answer = fields.get("answer", "").strip()
            embedding_str = fields.get("embedding", None)

            if not question or not answer:
                continue

            if embedding_str:
                try:
                    embedding = np.array(json.loads(embedding_str), dtype=np.float32)
                except Exception:
                    embedding = None
            else:
                embedding = None

            if embedding is None:
                update_needed.append((record["id"], question))
                embedding = None

            self.data.append({
                "id": record["id"],
                "question": question,
                "answer": answer,
                "embedding": embedding
            })
            self.questions.append(question)
            self.answers.append(answer)
            self.embeddings.append(embedding)

        # Hesaplanması gereken embeddingler varsa hesapla ve güncelle
        if update_needed:
            print(f"{len(update_needed)} embedding hesaplanacak ve Airtable'a kaydedilecek...")
            for record_id, question in update_needed:
                emb = self.model.encode(question, convert_to_numpy=True).tolist()
                self.airtable.update(record_id, {"embedding": json.dumps(emb)})
                # Güncellemeyi data listesine de yansıt
                for item in self.data:
                    if item["id"] == record_id:
                        item["embedding"] = np.array(emb, dtype=np.float32)

        # Embedding matrisi numpy array haline getir
        self.embeddings = [
            item["embedding"] if item["embedding"] is not None else np.zeros(self.model.get_sentence_embedding_dimension())
            for item in self.data
        ]
        self.question_embeddings = torch.tensor(np.array(self.embeddings)).to(self.device)
        print(f"{len(self.questions)} soru ve embedding yüklendi.")

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
            return self.answers[best_idx], max_score
        else:
            return None, max_score

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
            ]  # type: ignore
            response = client.chat.completions.create(
                model=self.chatgpt_model,
                messages=messages,
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ChatGPT API hatası: {str(e)}"

    def add_to_airtable(self, question, answer):
        """
        Yeni soru-cevap ve embedding'i Airtable'a ekler.

        Args:
            question (str): Soru metni.
            answer (str): Cevap metni.
        """
        embedding = self.model.encode(question, convert_to_numpy=True).tolist()
        record = {
            "question": question,
            "answer": answer,
            "embedding": json.dumps(embedding)
        }
        self.airtable.create(record)
        self.questions.append(question)
        self.answers.append(answer)
        self.embeddings.append(np.array(embedding, dtype=np.float32))
        self.question_embeddings = torch.tensor(np.array(self.embeddings)).to(self.device)

    def run(self):
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
                if not self.is_about_ml(user_input):
                    print("Üzgünüz, yalnızca makine öğrenmesiyle ilgili sorulara yanıt veriyorum.")
                    continue
                print("Cevap verilemiyor, ChatGPT'den cevap alınıyor...")
                ai_answer = self.ask_openai(user_input)
                print(f"ChatGPT cevabı: {ai_answer} (Benzerlik skoru: {score:.4f})")
                self.add_to_airtable(user_input, ai_answer)

def get_env_variable(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Environment variable '{key}' is missing.")
    return value

if __name__ == "__main__":
    try:
        api_key = get_env_variable("AIRTABLE_API")
        base_id = get_env_variable("AIRTABLE_BASE_ID")
        table_name = get_env_variable("AIRTABLE_TABLE_NAME")

        qa_system = QASystem(
            airtable_api_key=api_key,
            airtable_base_id=base_id,
            airtable_table_name=table_name
        )
        qa_system.run()

    except EnvironmentError as e:
        print(f"[ERROR] Configuration error: {e}")
        exit(1)