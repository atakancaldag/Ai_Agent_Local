Yerel AI Agent Projesi

Bu proje, kullanıcıların sorularına en uygun cevapları sağlamak amacıyla yerel bilgisayarda embedding işlemi yaparak bir yapay zeka ajanı geliştirmektedir. Kullanıcıdan alınan soruyu önceden yüklenmiş veri tabanı ile karşılaştırarak en benzer soruyu bulur ve uygun cevabı döner.

---

Özellikler

- Veri Yükleme ve Ön İşleme: Sorular ve cevaplar JSON dosyasından yüklenir. <br> 
- Embedding Kullanımı: Sorular için Hugging Face'in SentenceTransformer modeli kullanılarak vektör temsilleri oluşturulur.<br> 
- Benzerlik Hesaplama: Kullanıcı sorusu ile veri tabanındaki sorular arasındaki benzerlik cosine similarity ile hesaplanır.<br> 
- Airtable Entegrasyonu: Veri tabanı olarak Airtable kullanılır, embeddingler de buraya kaydedilir.<br> 
- ChatGPT Entegrasyonu: Benzer soru bulunamazsa, OpenAI ChatGPT API'si kullanılarak cevap alınır ve Airtable'a eklenir.<br> 
- Türkçe Makine Öğrenmesi Odaklı: Sadece makine öğrenmesi ile ilgili sorulara yanıt verir.<br> 

---

Dosya Yapısı

all-MiniLM-L6-v2_embeddings.npy    # Önceden hesaplanmış embedding dosyası <br> 
api.json                          # OpenAI API anahtarı dosyası (JSON formatında)<br> 
keywords.json                     # Makine öğrenmesi anahtar kelimeleri<br> 
stopwords.json                   # Soru metinlerinden filtrelenecek kelimeler<br> 
main.py                          # Projenin ana Python dosyası<br> 
readme.md                        # Proje açıklaması (bu dosya)<br> 
requirements.txt                 # Proje bağımlılıkları<br> 

---

Kurulum

1. Gerekli kütüphaneleri yükleyin:

pip install -r requirements.txt

2. OpenAI API anahtarınızı api.json dosyasına JSON formatında ekleyin:

{
  "api_key": "YOUR_API_KEY_HERE"
}

---

Çalıştırma

python main.py

---

Notlar

- Proje sadece makine öğrenmesi ile ilgili sorulara yanıt vermek üzere tasarlanmıştır.
- Airtable tablonuzda soru, cevap ve embedding alanlarının doğru yapılandırıldığından emin olunuz.
- GPU destekliyorsa embedding işlemleri hızlanacaktır.
