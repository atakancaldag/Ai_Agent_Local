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
import re 

class QASystem:
    def __init__(self,
                 data_path='data.json',
                 model_name="all-MiniLM-L6-v2",
                 similarity_threshold=0.7,
                 api_key_path='openai_api.json',
                 chatgpt_model="gpt-3.5-turbo",
                 keywords_path='keywords.json', 
                 chroma_dir: str ='chroma_db_persistent',
                 low_score_qa_path='low_score_qa.json'): 
        """
        Soru-cevap sistemini başlatır ve gerekli tüm bileşenleri yükler.
        """
        self.data_path = data_path
        self.low_score_qa_path = low_score_qa_path 
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.api_key_path = api_key_path
        self.chatgpt_model = chatgpt_model
        self.ml_keywords = keywords_path
        self.chroma_dir = chroma_dir

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"ChromaDB verileri '{self.chroma_dir}' dizininde saklanacak/yüklenecek.")
        self.chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection_name: str = "qa_collection_persistent"
        self.collection: chromadb.Collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
 
        self.data = [] 
        self.low_score_qa_data = [] 
        self.questions = [] 

        self._load_ml_keywords_and_stopwords()
        self.load_data() 
        self.embed_questions() 
        self.load_openai_key()

    def _load_json(self, path, default=None):
        """Yardımcı fonksiyon: JSON dosyasını yükler."""
        if default is None:
            default = {}
        if not os.path.exists(path): return default
        try:
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return default

    def load_data(self):
        """
        data_path ve low_score_qa_path içerisindeki JSON dosyalarını yükler.
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
                self.questions = [item['question'] for item in self.data]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Uyarı: '{self.data_path}' veri dosyası bulunamadı veya hatalı: {e}. Boş aktif liste ile devam ediliyor.")
            self.data = []
            self.questions = []

        try:
            with open(self.low_score_qa_path, 'r', encoding='utf-8') as file:
                self.low_score_qa_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Uyarı: '{self.low_score_qa_path}' düşük puanlı QA dosyası bulunamadı veya hatalı: {e}. Boş pasif liste ile devam ediliyor.")
            self.low_score_qa_data = []

    def _save_data(self):
        """Aktif QA verisini data.json'a kaydeder."""
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Hata: '{self.data_path}' dosyasına yazılırken sorun oluştu: {e}")

    def _save_low_score_qa_data(self):
        """Düşük puanlı QA verisini low_score_qa.json'a kaydeder."""
        try:
            with open(self.low_score_qa_path, 'w', encoding='utf-8') as f:
                json.dump(self.low_score_qa_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Hata: '{self.low_score_qa_path}' dosyasına yazılırken sorun oluştu: {e}")

    def embed_questions(self):
        """
        `self.questions` listesindeki (aktif data) soruların embedding'lerini oluşturur ve ChromaDB'ye kaydeder.
        """
        num_questions_in_data = len(self.questions)
        num_items_in_collection = self.collection.count()

        if not self.questions:
            print("Embedding için hiç aktif soru bulunamadı. Lütfen önce veriyi yükleyin.")
            if num_items_in_collection > 0:
                all_ids = self.collection.get(include=[])['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                    print(f"Koleksiyondaki {len(all_ids)} öğe silindi (aktif soru kalmadığı için).")
            return

        if num_items_in_collection != num_questions_in_data:
            print(f"ChromaDB koleksiyonundaki öğe sayısı ({num_items_in_collection}) ile aktif veri dosyasındaki soru sayısı ({num_questions_in_data}) eşleşmiyor.")
            print("Mevcut koleksiyon temizlenip yeniden embedding oluşturulacak.")
            
            if num_items_in_collection > 0:
                 all_ids = self.collection.get(include=[])['ids']
                 if all_ids:
                     self.collection.delete(ids=all_ids)
                 print(f"Koleksiyondaki {len(all_ids)} öğe silindi.")

            print("Aktif sorular için embedding'ler oluşturuluyor...")
            embeddings: List[List[float]] = self.model.encode(self.questions, convert_to_tensor=False, show_progress_bar=True).tolist()
            ids: List[str] = [str(i) for i in range(num_questions_in_data)]
            
            metadatas = [{'question': item['question'], 'answer': item['answer']} for item in self.data]

            try:
                self.collection.add(
                    ids=ids,
                    documents=self.questions, 
                    embeddings=embeddings,
                    metadatas=metadatas 
                )
                print(f"{len(ids)} adet aktif soru embedding'i ChromaDB'ye başarıyla eklendi.")
            except Exception as e:
                print(f"ChromaDB'ye embedding eklenirken hata oluştu: {e}")
        else:
            print("Mevcut aktif embedding'ler güncel. Yeniden oluşturmaya gerek yok.")

    def _load_ml_keywords_and_stopwords(self):
        """Yardımcı fonksiyon: Anahtar kelimeleri ve stop words'leri başlangıçta yükler."""
        try:
            with open(self.ml_keywords, "r", encoding="utf-8") as f:
                self.ml_keywords_set = set(keyword.lower() for keyword in json.load(f))
        except Exception as e:
            print(f"'{self.ml_keywords}' dosyası okunurken hata oluştu: {e}. Konu kontrolü devre dışı.")
            self.ml_keywords_set = set()

        try:
            with open('stopwords.json', 'r', encoding='utf-8') as f:
                self.stop_words = set(json.load(f))
        except FileNotFoundError:
            print("Uyarı: 'stopwords.json' bulunamadı. Basit bir stop words listesi kullanılacak.")
            self.stop_words = {'ve', 'veya', 'ile', 'ama', 'çünkü', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü', 'bu', 'şu', 'o', 'bir', 'için', 'ne', 'nasıl', 'nedir'}

    def load_openai_key(self):
        """
        OpenAI API anahtarını `self.api_key_path` ile belirtilen dosyadan yükler.
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

            if QASystem.check_openai_api_key(key): 
                print("API anahtarı doğru.")
                break
            else:
                print("API anahtarı geçersiz. Lütfen tekrar girin.")
                os.remove(self.api_key_path)

    @staticmethod
    def check_openai_api_key(api_key):
        """
        OpenAI API anahtarının geçerli olup olmadığını kontrol eder.
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
        """
        try:
            client = OpenAI(api_key=openai.api_key)
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system",
                 "content": "You are a Turkish coding assistant specialized in machine learning and answering only machine learning related questions."},
                {"role": "user", "content": prompt}
            ]
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
        Kullanıcının sorduğu soruya en benzer soruyu aktif veri kümesinde bulur.
        Sadece aktif (data.json) havuzdaki soruları dikkate alır.
        """
        print(f"DEBUG: find_best_match çağrıldı, user_question: '{user_question}'")

        if self.collection.count() == 0:
            print("DEBUG: ChromaDB koleksiyonunda hiç öğe yok. Eşleşme yapılamaz.")
            return None 

        user_emb: List[float] = self.model.encode(user_question, convert_to_numpy=False).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[user_emb], 
                n_results=1,
                include=['documents', 'distances', 'metadatas']
            )
            print(f"DEBUG: ChromaDB sonuçları: {results}")
        except Exception as e:
            print(f"DEBUG: ChromaDB sorgusu sırasında hata: {e}")
            return None 

        if results and results['ids'] and results['ids'][0]:
            distance = results['distances'][0][0] if results['distances'] and results['distances'][0] else float('inf')
            similarity = 1 - distance
            print(f"DEBUG: Eşleşme mesafesi: {distance}, Benzerlik skoru: {similarity:.4f}")

            if similarity >= self.similarity_threshold:
                matched_question_text = results['documents'][0][0] if isinstance(results['documents'][0], list) else results['documents'][0]
                print(f"DEBUG: Eşleşen soru metni (ChromaDB'den çıkarıldı): '{matched_question_text}'")
                
                # data.json içinde tam eşleşen item'ı bul
                for item in self.data:
                    if item['question'] == matched_question_text:
                        print(f"DEBUG: data.json içinde tam eşleşen soru bulundu: '{item['question']}'")
                        return item
                
                print(f"DEBUG: ChromaDB'de eşleşen soru metni bulundu ancak self.data içinde tam item bulunamadı. Bu bir senkronizasyon hatası olabilir.")
                return None 
            else:
                print(f"DEBUG: Benzerlik eşiğinin altında kaldı ({similarity:.4f} < {self.similarity_threshold}).")
                return None
        
        print("DEBUG: ChromaDB'den sonuç bulunamadı.")
        return None

    def add_new_qa_to_data(self, question: str, answer: str):
        """
        Yeni soruyu ve cevabını data.json dosyasına ve bellekteki verilere ekler,
        ardından embedding'leri günceller.
        Yeni eklenen sorulara başlangıç puanlama alanları eklenir.
        """
        if not question or not answer: 
            print("DEBUG: Soru veya cevap boş olamaz. Veriye eklenmedi.")
            return

        # Sadece soru metninin zaten var olup olmadığını kontrol et.
        # Eğer aynı soru metni zaten varsa, yeni bir girdi ekleme.
        for item in self.data:
            if item['question'] == question:
                print(f"DEBUG: Aynı soru metni zaten mevcut: '{question[:30]}...'. Yeni girdi eklenmedi.")
                return 

        new_entry = {
            "question": question, 
            "answer": answer,
            "sorulma_sayisi": 0,
            "ratings": [],
            "current_average": 0.0
        }
        
        self.data.append(new_entry)
        self.questions.append(question) 
        
        try:
            self._save_data() 
            print(f"DEBUG: Yeni soru-cevap '{question[:30]}...' başarıyla '{self.data_path}' dosyasına eklendi.")
            
            print("DEBUG: Veri güncellendi, embedding'ler yeniden oluşturulacak...")
            self.embed_questions() 

        except Exception as e:
            print(f"DEBUG: Yeni soru-cevap eklenirken beklenmedik bir hata oluştu: {e}")
            if self.data and self.data[-1] == new_entry: self.data.pop()
            if self.questions and self.questions[-1] == question: self.questions.pop()
    
    def update_answer_rating(self, question_text: str, answer_text: str, rating: int):
        """
        Belirli bir soru-cevap çiftinin puanını günceller, ortalamayı hesaplar
        ve duruma göre aktif/pasif havuzlar arasında taşır.
        """
        found_item = None
        item_index = -1

        # Aktif havuzda (data.json) ilgili soru-cevap çiftini bul
        for i, item in enumerate(self.data):
            if item['question'] == question_text and item['answer'] == answer_text:
                found_item = item
                item_index = i
                break
        
        if not found_item:
            print(f"DEBUG: Hata: Puanlanacak soru-cevap çifti aktif havuzda bulunamadı: Soru: '{question_text[:50]}...', Cevap: '{answer_text[:50]}...'")
            return {"status": "error", "message": "Puanlanacak soru-cevap bulunamadı."}

        # Puanı ekle ve sayaçları güncelle
        found_item['ratings'].append(rating)
        found_item['sorulma_sayisi'] += 1
        found_item['current_average'] = sum(found_item['ratings']) / len(found_item['ratings'])

        if found_item['sorulma_sayisi'] > 3 and found_item['current_average'] < 3.0:
            print(f"DEBUG: Soru-cevap çifti düşük puan aldı ({found_item['current_average']:.2f}). Pasif havuza taşınıyor.")
            
            self.data.pop(item_index)
            self.questions.remove(question_text)

            self.low_score_qa_data.append(found_item)

            self._save_data() 
            self._save_low_score_qa_data() 
            self.embed_questions() 
            
            return {"status": "success", "message": "Cevap puanlandı ve düşük puan nedeniyle pasif havuza taşındı."}
        else:
            self._save_data() 
            print(f"DEBUG: Cevap puanlandı. Yeni ortalama: {found_item['current_average']:.2f}, Sorulma Sayısı: {found_item['sorulma_sayisi']}")
            return {"status": "success", "message": "Cevap başarıyla puanlandı."}

