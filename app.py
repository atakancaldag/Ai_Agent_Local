import os
import json
import random
import re
from flask import Flask, request, jsonify
from main import QASystem 

# --- Kullanıcı Yönetimi Sınıfı ---
class UserManager:
    def __init__(self, filepath='users.json'):
        self.filepath = filepath
        self.users = self._load_users()
    def _load_users(self):
        if not os.path.exists(self.filepath): return []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return []
    def _save_users(self):
        with open(self.filepath, 'w', encoding='utf-8') as f: json.dump(self.users, f, ensure_ascii=False, indent=2)
    def find_user_by_email(self, email):
        for user in self.users:
            if user.get('email') == email: return user
        return None
    def check_credentials(self, email, password):
        user = self.find_user_by_email(email)
        return bool(user and user.get('sifre') == password)
    def add_user(self, name, email, password):
        if self.find_user_by_email(email): return False
        self.users.append({"name": name, "email": email, "sifre": password})
        self._save_users()
        return True

# --- Quiz Yönetimi Sınıfı ---
class QuizManager:
    def __init__(self, questions_path='quiz_questions.json', topics_path='user_topics.json', keywords_path='keywords.json'):
        self.questions_path = questions_path
        self.topics_path = topics_path
        self.keywords_path = keywords_path # keywords.json dosyasının yolu
        self.quiz_questions = self._load_json(self.questions_path)
        self.user_topics = self._load_json(self.topics_path, default=[])
        self.ml_keywords = self._load_json(self.keywords_path, default=[]) # Anahtar kelimeleri yükle
        self.topic_keywords = self._map_topics_to_keywords() # Konu başlıkları için anahtar kelimeler

    def _load_json(self, path, default=None):
        if default is None:
            default = {}
        if not os.path.exists(path): return default
        try:
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return default

    def _save_user_topics(self):
        """Kullanıcı konularını ve çözülen soruları dosyaya kaydeder."""
        with open(self.topics_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_topics, f, ensure_ascii=False, indent=2)
            
    def _map_topics_to_keywords(self):
        """
        quiz_questions.json dosyasındaki konu başlıklarından dinamik olarak
        bir anahtar kelime eşlemesi oluşturur.
        """
        mapping = {}
        if not self.quiz_questions:
            return mapping

        for topic in self.quiz_questions.keys():
            topic_lower = topic.lower()
            keywords = [topic_lower]
            keywords.extend(topic_lower.split())
            
            mapping[topic] = list(set(keywords))
            
        print(f"Dinamik olarak oluşturulan konu eşlemesi: {mapping}")
        return mapping

    def is_about_ml(self, text):
        """
        Verilen metnin, keywords.json'daki anahtar kelimelerden birini
        bütün bir kelime olarak içerip içermediğini kontrol eder.
        """
        text_lower = text.lower()
        for keyword in self.ml_keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                return True
        return False

    def get_topic_from_question(self, question_text):
        """Sorunun metnine göre en uygun konuyu belirler."""
        question_lower = question_text.lower()
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, question_lower):
                    return topic
        
        return "Genel Makine Öğrenmesi"

    def get_user_data(self, email):
        """Kullanıcının tüm veri girişini (konular ve çözülmüş sorular) döndürür."""
        user_entry = next((user for user in self.user_topics if user.get('email') == email), None)
        if not user_entry:
            user_entry = {"email": email, "topics": [], "answered_questions": {}}
            self.user_topics.append(user_entry)
            self._save_user_topics()
        return user_entry

    def add_topic_for_user(self, email, topic):
        """Kullanıcının konu listesine yeni bir konu ekler."""
        if not topic: return

        user_entry = self.get_user_data(email)
        if topic not in user_entry['topics']:
            user_entry['topics'].append(topic)
            self._save_user_topics()

    def get_user_quiz_status(self, email):
        """Kullanıcının quiz'e girmesi için kaç konusu olduğunu döndürür."""
        user_data = self.get_user_data(email)
        return len(user_data.get('topics', []))

    def get_question_for_user(self, email):
        """Kullanıcının konularından rastgele bir soru seçer ve döndürür, tekrarları önler."""
        user_data = self.get_user_data(email)
        user_topics_list = user_data.get('topics', [])
        answered_questions = user_data.get('answered_questions', {})

        if not user_topics_list:
            return {"status": "no_topics", "message": "Tebrikler, zayıf olduğunuz konu kalmadı!"}

        available_topics = [t for t in user_topics_list if t in self.quiz_questions and self.quiz_questions[t]]
        
        all_questions_exhausted = True
        for topic in available_topics:
            all_questions_in_topic_ids = {q['id'] for q in self.quiz_questions[topic]}
            asked_question_ids_in_topic = set(answered_questions.get(topic, []))
            if len(all_questions_in_topic_ids) > len(asked_question_ids_in_topic):
                all_questions_exhausted = False
                break
        
        if all_questions_exhausted and available_topics:
            for topic in available_topics:
                if topic in answered_questions:
                    del answered_questions[topic]
            user_data['answered_questions'] = answered_questions 
            self._save_user_topics()
            return {"status": "reset_needed", "message": "Zayıf olduğunuz konulardaki tüm soruları tamamladınız. Soru havuzu sıfırlandı, yeni sorulara geçebilirsiniz."}


        random.shuffle(available_topics) 

        chosen_topic = None
        chosen_question = None

        for topic in available_topics:
            asked_question_ids_in_topic = set(answered_questions.get(topic, []))
            possible_questions = [q for q in self.quiz_questions[topic] if q['id'] not in asked_question_ids_in_topic]

            if possible_questions:
                chosen_topic = topic
                chosen_question = random.choice(possible_questions)
                break
        
        if not chosen_question:
            user_data['answered_questions'] = {} 
            self._save_user_topics()
            return {"status": "reset_needed", "message": "Zayıf olduğunuz konulardaki tüm soruları tamamladınız. Soru havuzu sıfırlandı, yeni sorulara geçebilirsiniz."}
        
        answered_questions.setdefault(chosen_topic, []).append(chosen_question['id'])
        user_data['answered_questions'] = answered_questions 
        self._save_user_topics()

        return {
            "status": "question_found",
            "topic": chosen_topic,
            "question_id": chosen_question['id'], 
            "question": chosen_question['soru'],
            "options": chosen_question['siklar']
        }

    def check_answer_and_update(self, email, topic, question_id, user_answer):
        """Cevabı kontrol eder ve doğruysa kullanıcının listesinden konuyu siler."""
        user_data = self.get_user_data(email)
        user_topics_list = user_data.get('topics', [])
        answered_questions = user_data.get('answered_questions', {})

        correct_answer_char = None
        correct_answer_text = ""
        
        target_question = None
        if topic in self.quiz_questions:
            for q in self.quiz_questions[topic]:
                if q['id'] == question_id: 
                    target_question = q
                    correct_answer_char = q['dogru_cevap']
                    correct_answer_text = q['siklar'][correct_answer_char]
                    break
        
        if not target_question: return {"result": "error", "message": "Soru bulunamadı."}

        if user_answer.strip().upper() == correct_answer_char:
            if topic in user_topics_list:
                user_topics_list.remove(topic)
                user_data['topics'] = user_topics_list
            
            if topic in answered_questions:
                del answered_questions[topic]
                user_data['answered_questions'] = answered_questions

            self._save_user_topics()
            return {"result": "correct", "message": "Doğru cevap!"}
        else:
            return {"result": "incorrect", "message": f"Yanlış cevap. Doğrusu: {correct_answer_char}) {correct_answer_text}"}

    def reset_user_quiz_progress(self, email):
        """
        Kullanıcının quiz ilerlemesini sıfırlar. Sadece cevaplanmış soruları temizler,
        öğrencinin çalıştığı konuları SİLMEZ.
        """
        user_data = self.get_user_data(email)
        user_data['answered_questions'] = {}
        self._save_user_topics()
        return {"status": "success", "message": "Quiz ilerlemesi sıfırlandı. Konularınız korundu."}


# --- Uygulama Kurulumu ve Webhook'lar ---
app = Flask(__name__)
print("Sistemler başlatılıyor...")
# QASystem'i low_score_qa_path ile başlat
qa_system = QASystem(low_score_qa_path='low_score_qa.json')
user_manager = UserManager()
quiz_manager = QuizManager() 
print("Sistemler başarıyla yüklendi.")

@app.route('/login', methods=['POST'])
def handle_login():
    data = request.get_json()
    email, password = data.get('email'), data.get('sifre')
    if not email or not password: return jsonify({"status": "error", "message": "E-posta ve şifre zorunlu."}), 400
    if user_manager.check_credentials(email, password):
        user = user_manager.find_user_by_email(email)
        return jsonify({"status": "success", "name": user.get('name', '')})
    return jsonify({"status": "error", "message": "Geçersiz e-posta veya şifre."})

@app.route('/register', methods=['POST'])
def handle_register():
    data = request.get_json()
    name, email, password = data.get('name'), data.get('email'), data.get('sifre')
    if not all([name, email, password]): return jsonify({"status": "error", "message": "Tüm alanlar zorunlu."}), 400
    if user_manager.add_user(name, email, password):
        return jsonify({"status": "success", "message": f"Hoş geldin, {name}!"})
    return jsonify({"status": "error", "message": "Bu e-posta zaten kayıtlı."})

@app.route('/ask', methods=['POST'])
def handle_ask():
    print("------------------------------------")
    print("'/ask' webhook'u çağrıldı.")
    print(f"Gelen İstek Data (Raw): {request.data}")

    try:
        data = request.get_json()
        print(f"Gelen İstek JSON: {data}")
    except Exception as e:
        print(f"ERROR: JSON ayrıştırma hatası: {e}")
        return jsonify({"error": "Geçersiz JSON formatı."}), 400

    if data is None:
        print("ERROR: request.get_json() None döndürdü.")
        return jsonify({"error": "İstek gövdesi boş veya geçerli JSON değil."}), 400

    user_question = data.get('question')
    request_type = data.get('request_type')
    email = data.get('email')

    print(f"Alınan 'user_question': {user_question} (Tipi: {type(user_question)})")
    print(f"Alınan 'request_type': {request_type} (Tipi: {type(request_type)})")
    print(f"Alınan 'email': {email} (Tipi: {type(email)})")

    if not user_question or not email:
        print("ERROR: 'question' veya 'email' alanı eksik veya boş.")
        return jsonify({"error": "Soru ve email alanları zorunludur."}), 400

    response_text, response_status = "", "success"
    question_to_rate = user_question 
    answer_to_rate = "" 

    if not quiz_manager.is_about_ml(user_question):
        response_text, response_status = "Üzgünüz, yalnızca makine öğrenmesiyle ilgili sorulara yanıt veriyorum.", "rejected"
        print(f"DEBUG: Konu dışı soru: '{user_question}'")
    elif request_type == 'regenerate':
        print(f"DEBUG: 'regenerate' isteği algılandı.")
        if user_question is None: 
            print("ERROR: user_question is unexpectedly None in regenerate block.")
            return jsonify({"error": "Soru alanı boş."}), 400

        topic = quiz_manager.get_topic_from_question(user_question)
        print(f"DEBUG: get_topic_from_question sonucu: {topic}")
        
        quiz_manager.add_topic_for_user(email, topic)
        
        regenerate_prompt = f"'{user_question}' konusunu daha basit veya farklı bir dille yeniden açıklar mısın?"
        print(f"DEBUG: regenerate_prompt: {regenerate_prompt}")
        
        response_text = qa_system.ask_openai(regenerate_prompt)
        print(f"DEBUG: qa_system.ask_openai sonucu: {response_text} (Tipi: {type(response_text)})")

        if response_text is None:
            print("ERROR: qa_system.ask_openai'den None döndü.")
            response_text = "Üzgünüm, şu anda bir cevap üretemiyorum."
            response_status = "error"
            return jsonify({"answer": response_text, "status": response_status})

        question_to_rate = user_question 
        answer_to_rate = response_text
        print(f"DEBUG: Yeniden oluşturma isteği işlendi. question_to_rate: '{question_to_rate}', answer_to_rate: '{answer_to_rate}'") 

    else: 
        print(f"DEBUG: Standart 'ask' isteği algılandı.")
        matched_item = qa_system.find_best_match(user_question)
        
        if matched_item: 
            response_text = matched_item['answer']
            question_to_rate = matched_item['question'] 
            answer_to_rate = response_text
            print(f"DEBUG: data.json'dan eşleşen cevap bulundu: '{response_text[:50]}...'")
        else:
            print("DEBUG: Veritabanında uygun cevap bulunamadı veya eşik altında kaldı, ChatGPT'den cevap alınıyor...")
            ai_answer = qa_system.ask_openai(user_question)
            
            if ai_answer and not ai_answer.startswith("ChatGPT API hatası:"):
                qa_system.add_new_qa_to_data(user_question, ai_answer)
                response_text = ai_answer
                question_to_rate = user_question 
                answer_to_rate = response_text
                print(f"DEBUG: ChatGPT'den yeni cevap alındı ve eklenmeye çalışıldı: '{response_text[:50]}...'")
            else:
                response_text = "Üzgünüm, şu anda cevap veremiyorum veya ChatGPT bir hata döndürdü."
                response_status = "error"
                print(f"DEBUG: ChatGPT hatası veya geçersiz cevap: {response_text}")
                return jsonify({"answer": response_text, "status": response_status})

    return jsonify({
        "answer": response_text,
        "status": response_status,
        "question_text_for_rating": question_to_rate, 
        "answer_text_for_rating": answer_to_rate   
    })


# Puanlama endpoint'i
@app.route('/rate_answer', methods=['POST'])
def handle_rate_answer():
    print("------------------------------------")
    print("'/rate_answer' webhook'u çağrıldı.")
    
    print(f"Gelen İstek Data (Raw): {request.data}")
    
    try:
        data = request.get_json()
        print(f"Gelen İstek JSON: {data}") 
    except Exception as e:
        print(f"JSON ayrıştırma hatası: {e}")
        print(f"Gelen İstek Headers: {request.headers}")
        return jsonify({"status": "error", "message": "Geçersiz JSON formatı."}), 400

    question_text = data.get('question')
    answer_text = data.get('answer')
    rating = data.get('rating')

    print(f"Alınan 'question': {question_text}")
    print(f"Alınan 'answer': {answer_text}")
    print(f"Alınan 'rating': {rating} (Tipi: {type(rating)})")

    if not all([question_text, answer_text, rating is not None]):
        print("Hata: Eksik veri (question, answer veya rating).")
        return jsonify({"status": "error", "message": "Soru, cevap ve puan zorunludur."}), 400

    try:
        rating_int = int(rating)
        print(f"Rating int'e çevrildi: {rating_int}")
    except ValueError:
        print(f"Hata: Puan geçerli bir sayıya dönüştürülemiyor: '{rating}'")
        return jsonify({"status": "error", "message": "Puan geçerli bir sayı olmalıdır."}), 400
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu (int dönüşümü): {e}")
        return jsonify({"status": "error", "message": f"Puan dönüşümü sırasında bir hata oluştu: {str(e)}"}), 500

    try:
        result = qa_system.update_answer_rating(question_text, answer_text, rating_int)
        print(f"Puanlama sonucu: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Puanlama sırasında QASystem metodunda hata oluştu: {e}")
        return jsonify({"status": "error", "message": f"Puanlama sırasında bir hata oluştu: {str(e)}"}), 500


@app.route('/get_quiz_status', methods=['POST'])
def get_quiz_status():
    data = request.get_json()
    email = data.get('email')
    if not email: return jsonify({"error": "Email zorunludur."}), 400
    topic_count = quiz_manager.get_user_quiz_status(email)
    return jsonify({"topic_count": topic_count})

@app.route('/get_quiz_question', methods=['POST'])
def get_quiz_question():
    data = request.get_json()
    email = data.get('email')
    if not email: return jsonify({"error": "Email zorunludur."}), 400
    question_data = quiz_manager.get_question_for_user(email)
    return jsonify(question_data)

@app.route('/check_quiz_answer', methods=['POST'])
def check_quiz_answer():
    data = request.get_json()
    email, topic, question_id, user_answer = data.get('email'), data.get('topic'), data.get('question_id'), data.get('user_answer')
    if not all([email, topic, question_id, user_answer]):
        return jsonify({"error": "Tüm alanlar zorunludur."}), 400
    result = quiz_manager.check_answer_and_update(email, topic, question_id, user_answer)
    return jsonify(result)

@app.route('/reset_quiz_progress', methods=['POST'])
def reset_quiz_progress():
    data = request.get_json()
    email = data.get('email')
    if not email: return jsonify({"error": "Email zorunludur."}), 400
    result = quiz_manager.reset_user_quiz_progress(email)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
