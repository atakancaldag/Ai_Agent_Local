# Yerel Ai Agent Projesi

Bu proje, kullanıcıların sorularına en uygun cevapları sağlamak amacıyla yerel bilgisayarda embedding islemi yaparak bir yapay zeka ajanı geliştirmektedir. Proje, kullanıcıdan alınan soruyu, önceden yüklenmiş verilerle karşılaştırarak en benzer soruyu bulur ve uygun cevabı döner.

## Özellikler

- **Veri Yükleme ve Ön İşleme**: Sorular ve cevaplar bir JSON dosyasından yüklenir.
- **Embedding Kullanımı**: Sorular için **Hugging Face** kullanılarak vektör temsilleri oluşturulur.
- **Benzerlik Hesaplama**: Kullanıcıdan alınan sorgu ile veritabanındaki sorular arasındaki benzerlik, **cosine similarity** algoritmasıyla hesaplanır.

## Kullanım
1. **Kütüphanelerin Yüklenmesi**:\
   Proje, gerekli Python kütüphanelerini yüklemek için `requirements.txt` dosyasına sahiptir. Kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:
   
   ```bash
   pip install -r requirements.txt
   
2. **Projenin Çalıştırılması**:\
   Projeyi çalıştırmak için aşağıdaki komutu kullanabilirsiniz:
   
   ```bash
   python main.py