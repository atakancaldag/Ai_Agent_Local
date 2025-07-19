import os
import json
import numpy as np
import torch, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import openai
from openai.types.chat import ChatCompletionMessageParam
from typing import List, Tuple, Dict
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
                 low_score_qa_path='low_score_qa.json',
                 quiz_questions_path='quiz_questions.json'): # Quiz soruları dosyasını da ekledik
        """
        Soru-cevap sistemini başlatır ve gerekli tüm bileşenleri yükler.
        """
        self.data_path = data_path
        self.low_score_qa_path = low_score_qa_path 
        self.quiz_questions_path = quiz_questions_path # Quiz soruları yolu
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.api_key_path = api_key_path
        self.chatgpt_model = chatgpt_model
        self.ml_keywords = keywords_path
        self.chroma_dir = chroma_dir

        # Konu benzerliği için eşik
        self.TOPIC_SIMILARITY_THRESHOLD = 0.7 # Bu eşiğin altı yeni konu adayı

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"ChromaDB verileri '{self.chroma_dir}' dizininde saklanacak/yüklenecek.")
        self.chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(path=self.chroma_dir)
        
        # Ana QA koleksiyonu
        self.collection_name: str = "qa_collection_persistent"
        self.collection: chromadb.Collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        
        # Konu embedding'leri için yeni koleksiyon
        self.topic_collection_name: str = "qa_topic_collection_persistent"
        self.topic_collection: chromadb.Collection = self.chroma_client.get_or_create_collection(name=self.topic_collection_name)
        
        self.data = [] 
        self.low_score_qa_data = [] 
        self.quiz_questions_data = {} # quiz_questions.json içeriği
        self.questions = [] 
        self.canonical_topics = [] # Önceden tanımlanmış konu isimleri

        self._load_ml_keywords_and_stopwords()
        self.load_data() 
        self._load_and_embed_topics() # Konuları yükle ve embedding yap
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

    def _save_json(self, data, path):
        """Yardımcı fonksiyon: JSON dosyasını kaydeder."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Hata: '{path}' dosyasına yazılırken sorun oluştu: {e}")

    def load_data(self):
        """
        data_path, low_score_qa_path ve quiz_questions_path içerisindeki JSON dosyalarını yükler.
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
        
        self.quiz_questions_data = self._load_json(self.quiz_questions_path, default={})
        self.canonical_topics = list(self.quiz_questions_data.keys())


    def _save_data(self):
        """Aktif QA verisini data.json'a kaydeder."""
        self._save_json(self.data, self.data_path)

    def _save_low_score_qa_data(self):
        """Düşük puanlı QA verisini low_score_qa.json'a kaydeder."""
        self._save_json(self.low_score_qa_data, self.low_score_qa_path)

    def _save_quiz_questions_data(self):
        """Quiz soruları verisini quiz_questions.json'a kaydeder."""
        self._save_json(self.quiz_questions_data, self.quiz_questions_path)

    def _load_and_embed_topics(self):
        """
        quiz_questions.json'daki konuları yükler ve embedding'lerini ChromaDB'ye kaydeder.
        """
        current_topics_in_quiz_file = list(self.quiz_questions_data.keys())
        num_topics_in_collection = self.topic_collection.count()

        if not current_topics_in_quiz_file:
            print("DEBUG: Quiz soruları dosyasında hiç konu bulunamadı.")
            if num_topics_in_collection > 0:
                all_ids = self.topic_collection.get(include=[])['ids']
                if all_ids:
                    self.topic_collection.delete(ids=all_ids)
                    print(f"DEBUG: Konu koleksiyonundaki {len(all_ids)} öğe silindi.")
            return

        # Sadece eksik veya güncel olmayan konuları yeniden embedding yap
        if num_topics_in_collection != len(current_topics_in_quiz_file):
            print(f"DEBUG: Konu koleksiyonundaki öğe sayısı ({num_topics_in_collection}) ile quiz konuları sayısı ({len(current_topics_in_quiz_file)}) eşleşmiyor.")
            print("DEBUG: Konu koleksiyonu temizlenip yeniden embedding oluşturulacak/güncellenecek.")
            
            if num_topics_in_collection > 0:
                all_ids = self.topic_collection.get(include=[])['ids']
                if all_ids:
                    self.topic_collection.delete(ids=all_ids)
                    print(f"DEBUG: Konu koleksiyonundaki {len(all_ids)} öğe silindi.")

            print("DEBUG: Konular için embedding'ler oluşturuluyor...")
            embeddings: List[List[float]] = self.model.encode(current_topics_in_quiz_file, convert_to_tensor=False, show_progress_bar=True).tolist()
            ids: List[str] = [str(i) for i in range(len(current_topics_in_quiz_file))]
            
            try:
                self.topic_collection.add(
                    ids=ids,
                    documents=current_topics_in_quiz_file, 
                    embeddings=embeddings
                )
                print(f"DEBUG: {len(ids)} adet konu embedding'i ChromaDB'ye başarıyla eklendi.")
                self.canonical_topics = current_topics_in_quiz_file # In-memory listeyi güncelle
            except Exception as e:
                print(f"DEBUG: Konu embedding eklenirken hata oluştu: {e}")
        else:
            print("DEBUG: Mevcut konu embedding'leri güncel. Yeniden oluşturmaya gerek yok.")
            self.canonical_topics = current_topics_in_quiz_file # In-memory listeyi güncelle


    def embed_questions(self):
        """
        `self.questions` listesindeki (aktif data) soruların embedding'lerini oluşturur ve ChromaDB'ye kaydeder.
        """
        num_questions_in_data = len(self.questions)
        num_items_in_collection = self.collection.count()

        if not self.questions:
            print("DEBUG: Embedding için hiç aktif soru bulunamadı. Lütfen önce veriyi yükleyin.")
            if num_items_in_collection > 0:
                all_ids = self.collection.get(include=[])['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                    print(f"DEBUG: QA koleksiyonundaki {len(all_ids)} öğe silindi (aktif soru kalmadığı için).")
            return

        if num_items_in_collection != num_questions_in_data:
            print(f"DEBUG: QA koleksiyonundaki öğe sayısı ({num_items_in_collection}) ile aktif veri dosyasındaki soru sayısı ({num_questions_in_data}) eşleşmiyor.")
            print("DEBUG: QA koleksiyonu temizlenip yeniden embedding oluşturulacak.")
            
            if num_items_in_collection > 0:
                 all_ids = self.collection.get(include=[])['ids']
                 if all_ids:
                     self.collection.delete(ids=all_ids)
                 print(f"DEBUG: QA koleksiyonundaki {len(all_ids)} öğe silindi.")

            print("DEBUG: Aktif sorular için embedding'ler oluşturuluyor...")
            embeddings: List[List[float]] = self.model.encode(self.questions, convert_to_tensor=False, show_progress_bar=True).tolist()
            ids: List[str] = [str(i) for i in range(num_questions_in_data)]
            
            metadatas = [{'question': item['question'], 'answer': item['answer'], 'topic': item.get('topic', 'Genel')} for item in self.data]

            try:
                self.collection.add(
                    ids=ids,
                    documents=self.questions, 
                    embeddings=embeddings,
                    metadatas=metadatas 
                )
                print(f"DEBUG: {len(ids)} adet aktif soru embedding'i ChromaDB'ye başarıyla eklendi.")
            except Exception as e:
                print(f"DEBUG: ChromaDB'ye embedding eklenirken hata oluştu: {e}")
        else:
            print("DEBUG: Mevcut aktif embedding'ler güncel. Yeniden oluşturmaya gerek yok.")

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
            print(f"DEBUG: ChatGPT API hatası: {str(e)}")
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
                # BURADAKİ DÜZELTME: results['documents'][0] bir liste döndürüyorsa, ilk elemanı al.
                matched_question_text = results['documents'][0][0] if isinstance(results['documents'][0], list) else results['documents'][0]
                print(f"DEBUG: Eşleşen soru metni (ChromaDB'den çıkarıldı): '{matched_question_text}'")
                
                # data.json içinde tam eşleşen item'ı bul
                for item in self.data:
                    if item['question'] == matched_question_text:
                        print(f"DEBUG: data.json içinde tam eşleşen soru bulundu: '{item['question']}'")
                        return item # Tüm eşleşen öğeyi döndür
                
                print(f"DEBUG: ChromaDB'de eşleşen soru metni bulundu ancak self.data içinde tam item bulunamadı. Bu bir senkronizasyon hatası olabilir.")
                return None 
            else:
                print(f"DEBUG: Benzerlik eşiğinin altında kaldı ({similarity:.4f} < {self.similarity_threshold}).")
                return None
        
        print("DEBUG: ChromaDB'den sonuç bulunamadı.")
        return None

    def add_new_qa_to_data(self, question: str, answer: str, topic: str = "Genel Makine Öğrenmesi"):
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
            "answer2": "", 
            "sorulma_sayisi": 0,
            "ratings": [],
            "current_average": 0.0,
            "topic": topic # Konu bilgisini ekle
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
    
    def update_answer2(self, question_text: str, new_answer2: str):
        """
        Belirli bir soru için answer2 alanını günceller.
        """
        for item in self.data:
            if item['question'] == question_text:
                item['answer2'] = new_answer2
                self._save_data()
                print(f"DEBUG: Soru '{question_text[:30]}...' için answer2 güncellendi.")
                return True
        print(f"DEBUG: Soru '{question_text[:30]}...' için answer2 güncellenemedi, soru bulunamadı.")
        return False

    def update_answer_rating(self, question_text: str, answer_text: str, rating: int):
        """
        Belirli bir soru-cevap çiftinin puanını günceller, ortalamayı hesaplar
        ve duruma göre aktif/pasif havuzlar arasında taşır.
        Sadece birincil cevap (item['answer']) puanlanır.
        """
        found_item = None
        item_index = -1

        # Aktif havuzda (data.json) ilgili soru-cevap çiftini bul
        for i, item in enumerate(self.data):
            if item['question'] == question_text:
                # Puanlanan cevap, item'ın birincil cevabı mı?
                if item['answer'] == answer_text:
                    found_item = item
                    item_index = i
                    break
                # Eğer puanlanan cevap answer2 ise, bu senaryoda puanlama yapmıyoruz.
                # Kullanıcıdan gelen answer_text'in her zaman item['answer'] olması bekleniyor.
                elif item['answer2'] == answer_text:
                    print(f"DEBUG: answer2'ye puanlama denemesi algılandı, ancak answer2 puanlanmayacak. Soru: '{question_text[:50]}...'")
                    return {"status": "ignored", "message": "İkinci cevaba puanlama yapılamaz."}
        
        if not found_item:
            print(f"DEBUG: Hata: Puanlanacak soru-cevap çifti aktif havuzda bulunamadı (birincil cevap eşleşmedi): Soru: '{question_text[:50]}...', Cevap: '{answer_text[:50]}...'")
            return {"status": "error", "message": "Puanlanacak soru-cevap bulunamadı veya birincil cevap değil."}

        # Puanı ekle ve sayaçları güncelle (sadece birincil cevap için)
        found_item['ratings'].append(rating)
        found_item['sorulma_sayisi'] += 1
        found_item['current_average'] = sum(found_item['ratings']) / len(found_item['ratings'])

        # Taşıma mantığı: sorulma_sayisi 3'ten büyükse ve ortalama 3'ün altına düşerse
        if found_item['sorulma_sayisi'] > 3 and found_item['current_average'] < 3.0:
            print(f"DEBUG: Soru-cevap çifti düşük puan aldı ({found_item['current_average']:.2f}). Taşıma kontrolü yapılıyor.")
            
            # Pasif havuza taşınacak olan eski birincil cevap bilgileri
            failed_answer_entry = {
                "question": found_item['question'],
                "answer": found_item['answer'], # Eski birincil cevap
                "sorulma_sayisi": found_item['sorulma_sayisi'],
                "ratings": found_item['ratings'],
                "current_average": found_item['current_average'],
                "topic": found_item.get('topic', 'Genel Makine Öğrenmesi') # Konuyu da taşı
            }
            self.low_score_qa_data.append(failed_answer_entry)
            
            # Eğer answer2 boş değilse, onu birincil answer'a terfi ettir
            if found_item['answer2']:
                print(f"DEBUG: answer2 mevcut. answer2 birincil cevaba terfi ettiriliyor.")
                found_item['answer'] = found_item['answer2'] # answer2'yi answer'a taşı
                found_item['answer2'] = "" # answer2'yi boşalt

                # Yeni birincil cevabın istatistiklerini sıfırla
                found_item['sorulma_sayisi'] = 0
                found_item['ratings'] = []
                found_item['current_average'] = 0.0
                
                self._save_data() # data.json'ı kaydet
                self._save_low_score_qa_data() # low_score_qa.json'ı kaydet
                self.embed_questions() # Embedding'leri yeniden oluştur (soru metni değişmese de, cevabı değişti)
                
                return {"status": "success", "message": "Cevap düşük puan aldı, answer2 terfi ettirildi."}
            else:
                # answer2 boş ise, komple soru-cevap çiftini aktif havuzdan sil ve pasif havuza taşı
                print(f"DEBUG: answer2 boş. Komple soru-cevap çifti pasif havuza taşınıyor.")
                # data.json'dan sil
                self.data.pop(item_index)
                self.questions.remove(question_text) # Embedding listesinden de kaldır
                
                # low_score_qa.json'a ekle (failed_answer_entry'yi kullanmak daha güvenli)
                self.low_score_qa_data.append(failed_answer_entry) 

                self._save_data() 
                self._save_low_score_qa_data() 
                self.embed_questions() 
                
                return {"status": "success", "message": "Cevap düşük puan aldı ve pasif havuza taşındı."}
        else:
            self._save_data() 
            print(f"DEBUG: Cevap puanlandı. Yeni ortalama: {found_item['current_average']:.2f}, Sorulma Sayısı: {found_item['sorulma_sayisi']}")
            return {"status": "success", "message": "Cevap başarıyla puanlandı."}

    def get_qa_topic(self, user_question: str) -> str:
        """
        Kullanıcının sorusuna en uygun makine öğrenmesi konusunu belirler.
        Eğer mevcut konularla eşleşmezse, ChatGPT'den yeni bir konu tespit eder
        ve gerekirse bu konuyu ve ilgili quiz sorularını dinamik olarak oluşturur.
        """
        print(f"DEBUG: get_qa_topic çağrıldı, user_question: '{user_question}'")

        if not self.canonical_topics:
            print("DEBUG: Henüz yüklü kanonik konu yok. 'Genel Makine Öğrenmesi' döndürülüyor.")
            # Eğer hiç kanonik konu yoksa, önce quiz_questions.json'dan yüklemeyi dene
            self.quiz_questions_data = self._load_json(self.quiz_questions_path, default={})
            self.canonical_topics = list(self.quiz_questions_data.keys())
            if not self.canonical_topics: # Hala boşsa
                return "Genel Makine Öğrenmesi" # Varsayılan konu

        user_emb: List[float] = self.model.encode(user_question, convert_to_numpy=False).tolist()

        try:
            topic_results = self.topic_collection.query(
                query_embeddings=[user_emb],
                n_results=1,
                include=['documents', 'distances']
            )
            print(f"DEBUG: Konu arama sonuçları: {topic_results}")

            if topic_results and topic_results['ids'] and topic_results['ids'][0]:
                best_topic_text = topic_results['documents'][0][0] if isinstance(topic_results['documents'][0], list) else topic_results['documents'][0]
                similarity = 1 - topic_results['distances'][0][0]
                print(f"DEBUG: En benzer konu: '{best_topic_text}' (Benzerlik: {similarity:.4f})")

                if similarity >= self.TOPIC_SIMILARITY_THRESHOLD:
                    print(f"DEBUG: Eşik üzerinde benzerlik bulundu. Konu: '{best_topic_text}'")
                    return best_topic_text
        except Exception as e:
            print(f"DEBUG: Konu arama sırasında ChromaDB hatası: {e}")
            pass # Hata durumunda veya sonuç yoksa LLM'e düş

        print(f"DEBUG: Mevcut konularla eşik ({self.TOPIC_SIMILARITY_THRESHOLD}) altında benzerlik. ChatGPT'den konu tespiti yapılıyor.")
        
        # ChatGPT'den konu tespiti
        topic_prompt = f"Kullanıcının sorduğu soru '{user_question}' hangi makine öğrenmesi alt konusuyla ilgilidir? Sadece konunun adını yaz, başka hiçbir açıklama yapma. Eğer makine öğrenmesiyle ilgili değilse 'Genel Makine Öğrenmesi' yaz."
        detected_topic = self.ask_openai(topic_prompt)
        
        if detected_topic and not detected_topic.startswith("ChatGPT API hatası:"):
            detected_topic = detected_topic.strip()
            print(f"DEBUG: ChatGPT tarafından tespit edilen konu: '{detected_topic}'")

            # Tespit edilen konu kanonik konular arasında yoksa, yeni konu olarak ekle
            if detected_topic not in self.canonical_topics:
                print(f"DEBUG: Yeni konu tespit edildi: '{detected_topic}'. Quiz soruları oluşturuluyor ve ekleniyor.")
                
                self.quiz_questions_data[detected_topic] = [] # quiz_questions.json'a yeni konu ekle
                
                # Yeni konu için quiz soruları üret
                generated_quiz_questions = self.generate_quiz_questions_for_topic(detected_topic, num_questions=3)
                if generated_quiz_questions:
                    # Üretilen her soruya bir 'id' alanı ekle (eğer LLM eklemediyse)
                    for i, q in enumerate(generated_quiz_questions):
                        if 'id' not in q:
                            q['id'] = f"{detected_topic.lower().replace(' ', '_')}_gen_{i}"
                    self.quiz_questions_data[detected_topic].extend(generated_quiz_questions)
                
                self._save_quiz_questions_data() # quiz_questions.json'ı kaydet
                
                # Yeni konuyu topic collection'a ekle ve in-memory listeyi güncelle
                self.canonical_topics.append(detected_topic)
                new_topic_emb = self.model.encode([detected_topic], convert_to_tensor=False).tolist()
                # Yeni ID ataması: mevcut topic collection boyutuna göre
                new_topic_id = str(self.topic_collection.count()) 
                self.topic_collection.add(ids=[new_topic_id], documents=[detected_topic], embeddings=new_topic_emb)
                print(f"DEBUG: Yeni konu '{detected_topic}' ve quiz soruları eklendi, embedding oluşturuldu.")
            
            return detected_topic
        else:
            print(f"DEBUG: ChatGPT konu tespiti başarısız oldu veya hata döndürdü. 'Genel Makine Öğrenmesi' döndürülüyor.")
            return "Genel Makine Öğrenmesi" # ChatGPT hata verirse varsayılan

    def generate_quiz_questions_for_topic(self, topic_name: str, num_questions: int = 3) -> List[Dict]:
        """
        Belirtilen konu hakkında ChatGPT'den çoktan seçmeli quiz soruları üretir.
        """
        print(f"DEBUG: '{topic_name}' konusu için {num_questions} adet quiz sorusu üretiliyor.")
        quiz_prompt = f"""
        Makine öğrenmesi konusunda '{topic_name}' başlığı altında {num_questions} adet çoktan seçmeli quiz sorusu oluştur. Her sorunun 4 şıkkı (A, B, C, D) ve doğru cevabı olmalı. Yanıtını aşağıdaki JSON formatında ver:

        [
            {{
                "id": "unique_id_for_question_1",
                "soru": "Soru metni 1?",
                "siklar": {{
                    "A": "Şık A",
                    "B": "Şık B",
                    "C": "Şık C",
                    "D": "Şık D"
                }},
                "dogru_cevap": "C"
            }},
            {{
                "id": "unique_id_for_question_2",
                "soru": "Soru metni 2?",
                "siklar": {{
                    "A": "Şık A",
                    "B": "Şık B",
                    "C": "Şık C",
                    "D": "Şık D"
                }},
                "dogru_cevap": "A"
            }}
        ]
        Her soru için benzersiz bir 'id' alanı eklemeyi unutma. 'id' alanı, konuyu ve soruyu temsil eden küçük harfli, boşluksuz bir string olmalı (örneğin: 'konu_adi_soru_1').
        """
        
        try:
            client = OpenAI(api_key=openai.api_key)
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system",
                 "content": "You are a helpful assistant that generates quiz questions in specified JSON format."},
                {"role": "user", "content": quiz_prompt}
            ]
            response = client.chat.completions.create(
                model=self.chatgpt_model,
                messages=messages,
                max_tokens=1024, # Daha uzun yanıtlar için token sayısını artır
                temperature=0.7,
                response_format={"type": "json_object"} # JSON formatında yanıt iste
            )
            
            response_content = response.choices[0].message.content.strip()
            print(f"DEBUG: ChatGPT'den gelen ham quiz yanıtı: {response_content[:200]}...")
            
            # ChatGPT'den gelen JSON'ı ayrıştır
            parsed_json = json.loads(response_content)
            if isinstance(parsed_json, dict) and "questions" in parsed_json:
                return parsed_json["questions"]
            elif isinstance(parsed_json, list):
                return parsed_json
            else:
                print(f"DEBUG: ChatGPT'den beklenen quiz JSON formatı alınamadı. Ham: {response_content}")
                return []
        except Exception as e:
            print(f"DEBUG: Quiz sorusu üretilirken ChatGPT API hatası veya JSON ayrıştırma hatası: {str(e)}")
            return []

