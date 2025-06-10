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
            # Konu adını küçük harfe çevir ve kelimelere ayır.
            topic_lower = topic.lower()
            keywords = [topic_lower]
            keywords.extend(topic_lower.split())
            
            # Tekrarları önlemek için seti kullan ve tekrar listeye çevir
            mapping[topic] = list(set(keywords))
            
            # Örnek: "Reinforcement Learning" için anahtar kelimeler:
            # ['reinforcement learning', 'reinforcement', 'learning']
        
        print(f"Dinamik olarak oluşturulan konu eşlemesi: {mapping}")
        return mapping

    def is_about_ml(self, text):
        """
        Verilen metnin, keywords.json'daki anahtar kelimelerden birini
        bütün bir kelime olarak içerip içermediğini kontrol eder.
        """
        text_lower = text.lower()
        for keyword in self.ml_keywords:
            # \b kelime sınırını belirtir, böylece "ai" kelimesi "train" ile eşleşmez.
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                return True
        return False

    def get_topic_from_question(self, question_text):
        """Sorunun metnine göre en uygun konuyu belirler."""
        question_lower = question_text.lower()
        
        # Daha spesifik konuları önce kontrol et
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, question_lower):
                    return topic
        
        # Eğer spesifik bir konu bulunamazsa, genel bir konu ata
        return "Genel Makine Öğrenmesi"

    def add_topic_for_user(self, email, topic):
        """Kullanıcının konu listesine yeni bir konu ekler."""
        if not topic: return
        
        user_entry = next((user for user in self.user_topics if user.get('email') == email), None)
        
        if user_entry:
            if topic not in user_entry['topics']:
                user_entry['topics'].append(topic)
        else:
            self.user_topics.append({"email": email, "topics": [topic]})
            
        self._save_user_topics()

    def get_user_quiz_status(self, email):
        """Kullanıcının quiz'e girmesi için kaç konusu olduğunu döndürür."""
        for user in self.user_topics:
            if user.get('email') == email:
                return len(user.get('topics', []))
        return 0

    def get_question_for_user(self, email):
        """Kullanıcının konularından rastgele bir soru seçer ve döndürür."""
        user_topics_list = next((user.get('topics', []) for user in self.user_topics if user.get('email') == email), [])
        
        if not user_topics_list:
            return {"status": "no_topics", "message": "Tebrikler, zayıf olduğunuz konu kalmadı!"}

        available_topics = [t for t in user_topics_list if t in self.quiz_questions and self.quiz_questions[t]]
        if not available_topics:
            return {"status": "no_question_for_topic", "message": "Zayıf olduğunuz konularla ilgili soru bulunamadı."}

        chosen_topic = random.choice(available_topics)
        chosen_question = random.choice(self.quiz_questions[chosen_topic])
        return {
            "status": "question_found",
            "topic": chosen_topic,
            "question": chosen_question['soru'],
            "options": chosen_question['siklar']
        }

    def check_answer_and_update(self, email, topic, question_text, user_answer):
        """Cevabı kontrol eder ve doğruysa kullanıcının listesinden konuyu siler."""
        correct_answer_char = None
        correct_answer_text = ""
        
        if topic in self.quiz_questions:
            for q in self.quiz_questions[topic]:
                if q['soru'] == question_text:
                    correct_answer_char = q['dogru_cevap']
                    correct_answer_text = q['siklar'][correct_answer_char]
                    break
        
        if not correct_answer_char: return {"result": "error", "message": "Soru bulunamadı."}

        if user_answer.strip().upper() == correct_answer_char:
            for user in self.user_topics:
                if user.get('email') == email and topic in user['topics']:
                    user['topics'].remove(topic)
                    self._save_user_topics()
                    break
            return {"result": "correct", "message": "Doğru cevap!"}
        else:
            return {"result": "incorrect", "message": f"Yanlış cevap. Doğrusu: {correct_answer_char}) {correct_answer_text}"}

# --- Uygulama Kurulumu ve Webhook'lar ---
app = Flask(__name__)
print("Sistemler başlatılıyor...")
qa_system = QASystem()
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
    data = request.get_json()
    user_question, request_type, email = data.get('question'), data.get('request_type'), data.get('email')
    if not user_question or not email:
        return jsonify({"error": "Soru ve email alanları zorunludur."}), 400

    response_text, response_status = "", "success"

    if not quiz_manager.is_about_ml(user_question):
        response_text, response_status = "Üzgünüz, yalnızca makine öğrenmesiyle ilgili sorulara yanıt veriyorum.", "rejected"
    elif request_type == 'regenerate':
        topic = quiz_manager.get_topic_from_question(user_question)
        quiz_manager.add_topic_for_user(email, topic)
        regenerate_prompt = f"'{user_question}' konusunu daha basit veya farklı bir dille yeniden açıklar mısın?"
        response_text = qa_system.ask_openai(regenerate_prompt)
    else:
        answer, score = qa_system.find_best_match(user_question)
        ai_answer = None
        if not answer:
            ai_answer = qa_system.ask_openai(user_question)
            qa_system.add_new_qa_to_data(user_question, ai_answer)
        response_text = answer or ai_answer

    return jsonify({"answer": response_text, "status": response_status})

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
    email, topic, question_text, user_answer = data.get('email'), data.get('topic'), data.get('question'), data.get('user_answer')
    if not all([email, topic, question_text, user_answer]):
        return jsonify({"error": "Tüm alanlar zorunludur."}), 400
    result = quiz_manager.check_answer_and_update(email, topic, question_text, user_answer)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 