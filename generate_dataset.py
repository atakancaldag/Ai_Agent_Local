import google.generativeai as genai
import os
import json
import time
import random 

GOOGLE_API_KEY = "geminiapikeyburayayayazilir" 


DATASET_PATH = 'data.json'


TARGET_QUESTION_COUNT = 200


TOPICS = [
    "Makine Öğrenmesi Nedir?", "Denetimli ve Denetimsiz Öğrenme", "Veri Seti (Eğitim, Test, Doğrulama)",
    "Aşırı Öğrenme (Overfitting) ve Nasıl Önlenir?", "Özellik (Feature) ve Etiket (Label)", "Sınıflandırma Problemi",
    "Regresyon Problemi", "Model Değerlendirme Metrikleri", "Doğruluk (Accuracy) ve Dezavantajları",
    "Karar Ağaçları (Decision Trees)", "K-En Yakın Komşu (K-NN) Algoritması", "Gradyan İnişi (Gradient Descent)",
    "Yapay Sinir Ağlarına Giriş", "Derin Öğrenme Nedir?", "Veri Ön İşlemenin Önemi",
    "Destek Vektör Makineleri (SVM)", "Rastgele Orman (Random Forest)", "Naive Bayes Sınıflandırıcısı",
    "Boyut Azaltma ve PCA", "Kümeleme (Clustering) ve K-Means", "ROC Eğrisi ve AUC Değeri"
]
# --- AYARLAR SONU ---


if "BURAYA_YENİ_VE_GÜVENLİ_API_ANAHTARINIZI_YAPIŞTIRIN" in GOOGLE_API_KEY:
    raise ValueError("Lütfen 1. ADIM'daki GOOGLE_API_KEY değişkenini YENİ ve GÜVENLİ anahtarınızla güncelleyin.")

# Gemini API'sini yapılandır
genai.configure(api_key=GOOGLE_API_KEY)

# Kullanılacak Gemini modelini seç
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def load_existing_data(filepath):
    """Mevcut JSON dosyasını okur ve içeriğini döndürür."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_data_to_json(filepath, data):
    """Veriyi JSON dosyasına güzel bir formatla (indent) yazar."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_new_qa_pair(existing_questions_normalized, topic):
    """Gemini API'sini kullanarak yeni ve benzersiz bir soru-cevap çifti oluşturur."""
    print(f"\n🌀 '{topic}' konusunda yeni bir soru-cevap çifti oluşturuluyor...")

    existing_questions_str = "\n- ".join(list(existing_questions_normalized)[:15])
    
    # --- ÇÖZÜM 2: Daha akıllı ve yaratıcı olmasını isteyen prompt ---
    prompt = f"""
    Sen, Makine Öğrenmesi (ML) konusuna yeni başlayanlar için bir SSS listesi hazırlayan bir uzmansın.
    Görevin, '{topic}' konusu hakkında, daha önce sorulmuş olanlardan FARKLI, konunun başka bir yönünü ele alan, temel seviyede ve yaratıcı bir soru sormaktır.
    Cevap, bu soruyu 2-3 cümlelik basit ve anlaşılır bir dille açıklamalıdır.

    Aşağıda daha önce sorulmuş olan sorular listelenmiştir. Lütfen bu listede OLMAYAN ve bu listeye BENZEMEYEN yeni bir soru üret.
    
    Mevcut Sorular:
    - {existing_questions_str}

    Lütfen cevabını SADECE aşağıdaki JSON formatında ver. Başka hiçbir açıklama veya metin ekleme.

    {{
      "question": "Buraya yeni ve yaratıcı giriş seviyesi soruyu yaz",
      "answer": "Buraya sorunun 2-3 cümlelik basit ve anlaşılır cevabını yaz"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        qa_pair = json.loads(cleaned_response_text)
        
        if "question" in qa_pair and "answer" in qa_pair:
            return qa_pair
        else:
            print("  ❌ HATA: Gemini'den gelen yanıt beklenen formatta değil.")
            return None
            
    except Exception as e:
        print(f"  ❌ HATA: Gemini API çağrısı veya JSON ayrıştırma sırasında bir hata oluştu: {e}")
        return None

def main():
    """Ana program döngüsü."""
    data = load_existing_data(DATASET_PATH)
    current_count = len(data)
    print(f"📚 Mevcut veri setinde {current_count} adet soru-cevap çifti bulunuyor.")
    
    if current_count >= TARGET_QUESTION_COUNT:
        print(f"🎉 Hedef olan {TARGET_QUESTION_COUNT} soruya zaten ulaşılmış veya geçilmiş. Program sonlandırılıyor.")
        return

    existing_questions_normalized = {item['question'].lower().strip() for item in data}
    
    while len(data) < TARGET_QUESTION_COUNT:
        # --- ÇÖZÜM 1: Konuyu rastgele seçiyoruz ---
        topic = random.choice(TOPICS) 
        
        new_qa = generate_new_qa_pair(existing_questions_normalized, topic)
        
        if new_qa and new_qa['question'].lower().strip() not in existing_questions_normalized:
            data.append(new_qa)
            existing_questions_normalized.add(new_qa['question'].lower().strip())
            
            save_data_to_json(DATASET_PATH, data)
            print(f"✅ Başarıyla eklendi! Toplam soru sayısı: {len(data)}. (Hedef: {TARGET_QUESTION_COUNT})")
        else:
            if new_qa:
                print(f"⚠️  Uyarı: Oluşturulan soru '{new_qa['question']}' zaten mevcut. Tekrar denenecek.")
            else:
                print("⚠️  Uyarı: Geçerli bir çift oluşturulamadı, tekrar denenecek.")
            
        print("...İstek limitini aşmamak için 5 saniye bekleniyor...")
        time.sleep(5) 

    print(f"\n🎉 İşlem tamamlandı! Hedeflenen {TARGET_QUESTION_COUNT} soruya ulaşıldı.")
    print(f"Toplam soru sayısı artık {len(data)}.")

if __name__ == "__main__":
    main()