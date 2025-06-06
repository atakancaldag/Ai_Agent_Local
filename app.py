import os
import json
from flask import Flask, request, jsonify
from main import QASystem 

# --- Kullanıcı Yönetimi Sınıfı ---
class UserManager:
    def __init__(self, filepath='users.json'):
        """Kullanıcı verilerini yönetir."""
        self.filepath = filepath
        self.users = self._load_users()

    def _load_users(self):
        """users.json dosyasını okur."""
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_users(self):
        """Kullanıcı listesini dosyaya yazar."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)

    def find_user_by_email(self, email):
        """Verilen e-postaya sahip kullanıcıyı bulur."""
        for user in self.users:
            if user.get('email') == email:
                return user
        return None

    def check_credentials(self, email, password):
        """E-posta ve şifrenin eşleşip eşleşmediğini kontrol eder."""
        user = self.find_user_by_email(email)
        if user and user.get('sifre') == password:
            return True
        return False

    def add_user(self, name, email, password):
        """Yeni bir kullanıcı ekler, e-posta zaten varsa False döndürür."""
        if self.find_user_by_email(email):
            return False
        
        new_user = {
            "name": name,
            "email": email,
            "sifre": password
        }
        self.users.append(new_user)
        self._save_users()
        return True

# --- Uygulama Kurulumu ---
app = Flask(__name__)

print("Sistemler başlatılıyor...")
try:
    qa_system = QASystem()
    user_manager = UserManager()
    print("Sistemler başarıyla yüklendi ve hazır.")
except Exception as e:
    print(f"Sistemler başlatılırken kritik bir hata oluştu: {e}")
    qa_system = None
    user_manager = None

# --- Login Endpoint'i ---
@app.route('/login', methods=['POST'])
def handle_login():
    """Giriş isteğini yönetir ve başarılı olursa kullanıcı ismini döndürür."""
    if not user_manager:
        return jsonify({"status": "error", "message": "Kullanıcı sistemi aktif değil."}), 503
    data = request.get_json()
    email = data.get('email')
    password = data.get('sifre')
    if not email or not password:
        return jsonify({"status": "error", "message": "E-posta ve şifre alanları zorunludur."}), 400
    
    print(f"Login denemesi: Email - {email}")
    
    if user_manager.check_credentials(email, password):
        user_details = user_manager.find_user_by_email(email)
        user_name = user_details.get('name', 'Kullanıcı') 
        response_payload = {
            "status": "success",
            "message": f"Hoş geldin, {user_name}!",
            "name": user_name
        }
    else:
        response_payload = {
            "status": "error", 
            "message": "Geçersiz e-posta veya şifre.",
            "name": ""
        }
    return jsonify(response_payload)

# --- Register Endpoint'i ---
@app.route('/register', methods=['POST'])
def handle_register():
    if not user_manager:
        return jsonify({"status": "error", "message": "Kullanıcı sistemi aktif değil."}), 503
    data = request.get_json()
    email = data.get('email')
    password = data.get('sifre')
    name = data.get('name')
    if not email or not password or not name:
        return jsonify({"status": "error", "message": "İsim, e-posta ve şifre alanları zorunludur."}), 400
    print(f"Yeni kayıt denemesi: İsim - {name}, Email - {email}")
    if user_manager.add_user(name, email, password):
        response_payload = {"status": "success", "message": f"Hoş geldin, {name}! Kaydın başarıyla oluşturuldu."}
    else:
        response_payload = {"status": "error", "message": "Bu e-posta adresi zaten kayıtlı. Lütfen farklı bir e-posta deneyin."}
    return jsonify(response_payload)

# --- Soru-Cevap Endpoint'i ---
@app.route('/ask', methods=['POST'])
def handle_ask():
    """Landbot'tan gelen soru-cevap isteklerini karşılar."""
    if not qa_system:
        return jsonify({"error": "Soru-Cevap sistemi şu anda aktif değil."}), 503

    data = request.get_json()
    if data is None or 'question' not in data:
        return jsonify({"error": "Geçersiz istek: JSON verisi içinde 'question' anahtarı bekleniyor."}), 400

    user_question = data.get('question')
    request_type = data.get('request_type') 

    print(f"Gelen Soru: {user_question} | İstek Tipi: {request_type}")

    response_text = ""
    response_status = "success" # Varsayılan durum

    if request_type == 'regenerate':
        print("Yeniden oluşturma isteği alındı. OpenAI'ye yönlendiriliyor...")
        regenerate_prompt = f"'{user_question}' konusunu daha basit veya farklı bir dille yeniden açıklar mısın?"
        response_text = qa_system.ask_openai(regenerate_prompt)
        
    else:
        # Konu kontrolü
        if not qa_system.is_about_ml(user_question):
            response_text = "Üzgünüz, yalnızca makine öğrenmesiyle ilgili sorulara yanıt veriyorum."
            response_status = "rejected" # reddetme kısmı
        else:
            answer, score = qa_system.find_best_match(user_question)
            if answer:
                print(f"Cevap yerel veriden bulundu (Skor: {score:.4f}).")
                response_text = answer
            else:
                print("Yerel cevap bulunamadı. OpenAI'ye soruluyor...")
                ai_answer = qa_system.ask_openai(user_question)
                is_error_message = "hata" in ai_answer.lower() or "error" in ai_answer.lower()
                if not is_error_message:
                    qa_system.add_new_qa_to_data(user_question, ai_answer)
                    response_text = ai_answer
                else:
                    response_text = "Bu soruya şu anda bir cevap bulamadım. Lütfen farklı bir soru deneyin."

    response_payload = {
        "answer": response_text,
        "status": response_status 
    }
    return jsonify(response_payload)

# --- Uygulamayı Başlatma ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
