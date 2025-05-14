import os
import json
import numpy as np
import torch, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import openai
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Tuple

class QASystem:
    def __init__(self,
                 data_path='data.json',
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold=0.7,
                 api_key_path='openai_api.json',
                 chatgpt_model="gpt-3.5-turbo",
                 chroma_dir: str ='chroma_db_persistent'):
        """
        Soru-cevap sistemini başlatır ve gerekli tüm bileşenleri yükler.

        Args:
            data_path (str): JSON formatındaki soru-cevap veri dosyasının yolu.
            embedding_file (str): Embedding verilerinin kaydedileceği/yükleneceği dosya.
            model_name (str): Kullanılacak SentenceTransformer modeli.
            similarity_threshold (float): Benzerlik eşiği (0-1 arası).
            api_key_path (str): OpenAI API anahtarının saklandığı dosya yolu.
            chatgpt_model (str): Kullanılacak ChatGPT model adı.
            chroma_dir (str): ChromaDB veritabanının saklanacağı/yükleneceği dizin yolu. Varsayılan olarak 'chroma_db_persistent' kullanılır.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.api_key_path = api_key_path
        self.chatgpt_model = chatgpt_model
        self.chroma_dir = chroma_dir

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"ChromaDB verileri '{self.chroma_dir}' dizininde saklanacak/yüklenecek.")
        self.chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection_name: str = "qa_collection_persistent"
        self.collection: chromadb.Collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

        self.data = []
        self.questions = []

        self.load_data()
        self.embed_questions()
        self.load_openai_key()

    def load_data(self):
        """
        data_path içerisindeki JSON dosyasını self.question değişkenine yükler ve soruları belleğe alır.
        """
        try:
            with open(self.data_path, 'r') as file:
                self.data = json.load(file)
                self.questions = [item['question'] for item in self.data]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Veri dosyası hatalı veya eksik: {e}")
            exit(1)

    def add_to_data(self, new_question: str, gpt_answer: str):
        """
        Yeni bir soru ve GPT cevabını veri dosyasına ve ChromaDB'ye ekler.

        Args:
            new_question (str): Soru
            gpt_answer (str): ChatGPT'den alınan cevap
        """
        if any(q["question"].strip().lower() == new_question.strip().lower() for q in self.data):
            print("Bu soru zaten veri dosyasında mevcut.")
            return

        new_entry = {"question": new_question, "answer": gpt_answer}
        self.data.append(new_entry)
        self.questions.append(new_question)

        # JSON'a yaz
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            print("Yeni soru ve cevabı JSON dosyasına kaydedildi.")
        except Exception as e:
            print(f"Veri dosyasına yazma hatası: {e}")
            return

        # ChromaDB'ye ekle
        try:
            emb = self.model.encode(new_question, convert_to_tensor=False).tolist()
            doc_id = str(len(self.data) - 1)

            self.collection.add(
                ids=[doc_id],
                documents=[new_question],
                embeddings=[emb],
                metadatas=[{"answer": gpt_answer}]
            )
            print("Yeni embedding ChromaDB'ye eklendi.")
        except Exception as e:
            print(f"ChromaDB'ye embedding eklenirken hata: {e}")

    def is_question_about_ml(self, question: str) -> bool:
        try:
            client = OpenAI(api_key=openai.api_key)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that determines if the user's question is about machine learning or not. If the question is related to machine learning, answer with: YES: This question is about machine learning because it involves topics like [reason].   If the question is not related, answer with: NO: This question is not about machine learning because [reason]. Do not give long explanations, keep it concise but clear."
                    ),
                },
                {"role": "user", "content": question}
            ]

            response = client.chat.completions.create(
                model=self.chatgpt_model,
                messages=messages,
                temperature=0.2,
                max_tokens=10
            )
            reply = response.choices[0].message.content.strip().lower()
            print(f"[DEBUG] ML check reply: {reply}")  # Optional: to inspect what's coming back

            return reply.startswith("yes")
        except Exception as e:
            print(f"[ERROR] Makine öğrenmesi kontrolü başarısız: {e}")
            return False

    def embed_questions(self):
        """
        `self.questions` listesindeki soruların embedding'lerini oluşturur ve ChromaDB'ye kaydeder.

        Eğer ChromaDB'deki öğe sayısı `self.questions` listesindeki soru sayısından farklıysa,
        mevcut ChromaDB koleksiyonu silinir ve tüm sorular için yeniden embedding oluşturulup eklenir.
        Aksi takdirde, mevcut embedding'lerin güncel olduğu varsayılır.
        Embedding işlemi sırasında sorular `document`, cevaplar ise `metadata` olarak saklanır.
        """

        num_questions_in_data = len(self.questions)
        num_items_in_collection = self.collection.count()

        if not self.questions:
            print("Embedding için hiç soru bulunamadı. Lütfen önce veriyi yükleyin.")
            return

        if num_items_in_collection != num_questions_in_data:
            print(f"ChromaDB koleksiyonundaki öğe sayısı ({num_items_in_collection}) ile veri dosyasındaki soru sayısı ({num_questions_in_data}) eşleşmiyor.")
            print("Mevcut koleksiyon temizlenip yeniden embedding oluşturulacak.")

            if num_items_in_collection > 0:
                 all_ids = self.collection.get(include=[])['ids']
                 if all_ids:
                     self.collection.delete(ids=all_ids)
                 print(f"Koleksiyondaki {len(all_ids)} öğe silindi.")


            print("Sorular için embedding'ler oluşturuluyor...")
            embeddings: List[List[float]] = self.model.encode(self.questions, convert_to_tensor=False, show_progress_bar=True).tolist() # numpy array yerine list of lists
            ids: List[str] = [str(i) for i in range(num_questions_in_data)]

            metadatas = [{'answer': item['answer']} for item in self.data]

            try:
                self.collection.add(
                    ids=ids,
                    documents=self.questions,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"{len(ids)} adet soru embedding'i ChromaDB'ye başarıyla eklendi.")
            except Exception as e:
                print(f"ChromaDB'ye embedding eklenirken hata oluştu: {e}")
        else:
            print("Mevcut embedding'ler güncel. Yeniden oluşturmaya gerek yok.")

    def load_openai_key(self):
        """
        OpenAI API anahtarını `self.api_key_path` ile belirtilen dosyadan yükler.

        Dosya yoksa, boşsa veya geçersiz bir anahtar içeriyorsa, kullanıcıdan yeni bir
        API anahtarı girmesini ister ve dosyaya kaydeder. Anahtar doğrulanana kadar bu işlem tekrarlanır.
        Doğrulanan anahtar `openai.api_key` global değişkenine atanır.
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
                 "content": "You are an AI agent specialized in machine learning. You only respond to technical questions related to machine learning (such as algorithms, model training, evaluation metrics, frameworks, optimization, etc.). If your answer bigger than token do not cut it middle senteces cut it in point"},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=self.chatgpt_model,
                messages=messages,
                max_tokens=512,
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
        if self.collection.count() == 0:
            print("ChromaDB koleksiyonunda hiç öğe yok. Eşleşme yapılamaz.")
            return None, 0.0

        user_emb: List[float] = self.model.encode(user_question, convert_to_numpy=False).tolist() # Tek bir embedding için

        try:
            results = self.collection.query(
                query_embeddings=[user_emb],
                n_results=1,
                include=['documents', 'distances', 'metadatas']
            )
        except Exception as e:
            print(f"ChromaDB sorgusu sırasında hata: {e}")
            return None, 0.0



        if results and results['ids'] and results['ids'][0]:


            distance = results['distances'][0][0] if results['distances'] and results['distances'][0] else float('inf')
            similarity = 1 - distance


            if similarity >= self.similarity_threshold:

                answer = results['metadatas'][0][0].get('answer') if results['metadatas'] and results['metadatas'][0] and results['metadatas'][0][0] else None

                if answer:
                    return answer, similarity
                else:
                    return None, similarity
            else:
                return None, similarity

        return None, 0.0

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
                # Önce makine öğrenmesiyle ilgili mi kontrol et
                if self.is_question_about_ml(user_input):
                    print("Makine öğrenmesi ile ilgili. ChatGPT'den cevap alınıyor...")
                    ai_answer = self.ask_openai(user_input)
                    print(f"ChatGPT cevabı: {ai_answer} (Skor: {score:.4f})")

                    if score < self.similarity_threshold:
                        self.add_to_data(user_input, ai_answer)
                        print("Yeni soru ve cevabı veritabanına eklendi.")
                else:
                    print("Bu soru makine öğrenmesi ile ilgili değil. Cevap verilmeyecek.")



if __name__ == "__main__":
    qa_system = QASystem()
    qa_system.run()
