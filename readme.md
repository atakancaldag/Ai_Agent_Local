Yerel AI Agent Projesi

Bu proje, yerel bilgisayarda embedding işlemi yaparak, kullanıcının sorusuna en uygun cevabı sunan bir yapay zeka ajanı geliştirmektedir. Sistem, kullanıcının sorusunu önceden tanımlanmış veri tabanındaki sorularla karşılaştırarak en benzerini bulur ve ilgili cevabı döner. Eğer uygun cevap bulunamazsa, OpenAI ChatGPT API'si ile cevap oluşturulur ve veri tabanına eklenir.

Özellikler
- Veri Yükleme ve Ön İşleme: Sorular ve cevaplar JSON dosyasından yüklenir.
- Embedding Kullanımı: Hugging Face’in SentenceTransformer modeli ile sorular vektörlere dönüştürülür.
- Benzerlik Hesaplama: Kullanıcı sorusu ile veri tabanındaki sorular cosine similarity ile karşılaştırılır.
- Airtable Entegrasyonu: Veri tabanı ve embeddingler Airtable üzerinde yönetilir.
- ChatGPT Entegrasyonu: Benzer soru bulunamazsa, OpenAI ChatGPT API kullanılarak cevap alınır ve Airtable’a kaydedilir.
- Türkçe Makine Öğrenmesi Odaklı: Yalnızca makine öğrenmesi alanındaki sorulara yanıt verir.

Dosya Yapısı
- all-MiniLM-L6-v2_embeddings.npy — Önceden hesaplanmış embedding dosyası
- api.json — OpenAI API anahtarı (JSON formatında)
- keywords.json — Makine öğrenmesi anahtar kelimeleri listesi
- stopwords.json — Soru metinlerinden filtrelenecek kelimeler
- main.py — Projenin ana Python dosyası
- requirements.txt — Proje bağımlılıkları listesi
- readme.txt — Proje açıklaması

Kurulum
1. Bağımlılıkları yükleyin:
   pip install -r requirements.txt

2. OpenAI API anahtarınızı api.json dosyasına aşağıdaki formatta ekleyin:
{
  "api_key": "YOUR_API_KEY_HERE"
}

Çalıştırma
Projeyi başlatmak için terminalde aşağıdaki komutu kullanınız:
python main.py

Önemli Notlar
- Proje sadece makine öğrenmesi ile ilgili sorulara yanıt vermek üzere tasarlanmıştır.
- Airtable tablonuzda soru, cevap ve embedding alanlarının doğru yapılandırıldığından emin olunuz.
- Eğer GPU desteğiniz varsa, embedding hesaplama işlemleri hızlanacaktır.
