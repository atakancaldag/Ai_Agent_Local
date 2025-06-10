import json
import re
from zemberek import TurkishMorphology

# --- AYARLAR ---
INPUT_KEYWORDS_PATH = 'keywords.json'
OUTPUT_KEYWORDS_PATH = 'keywords_clean.json' # Çıktı dosyası bu olacak

TURKISH_STOP_WORDS = set([
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 
    'biz', 'bu', 'buna', 'bunda', 'bundan', 'bunlar', 'bunu', 'bunun', 'burada', 
    'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 
    'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 
    'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nedir', 'nerde', 'nerede', 'nereye', 
    'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 
    'yani', 'vb', 'olan', 'olarak', 'kadar', 'göre', 'ait', 'arasında', 'görevi',
    'fark', 'farkı', 'nedenlerinden', 'nelerdir', 'anlama', 'gelir', 'yarar', 'nasıl',
    'önlenir', 'olabilir', 'olur'
])
# --- AYARLAR SONU ---

def clean_and_normalize_keywords():
    print("Anahtar kelime temizleme işlemi başlatılıyor...")
    
    try:
        morphology = TurkishMorphology.create_with_defaults()
        print("✅ Zemberek morfoloji motoru başlatıldı.")
    except Exception as e:
        print(f"❌ HATA: Zemberek başlatılamadı. Hata: {e}")
        return

    try:
        with open(INPUT_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
            dirty_keywords = json.load(f)
        print(f"✅ '{INPUT_KEYWORDS_PATH}' dosyasından {len(dirty_keywords)} kelime okundu.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ HATA: '{INPUT_KEYWORDS_PATH}' dosyası okunamadı: {e}")
        return

    clean_keywords = set()
    for keyword in dirty_keywords:
        keyword = keyword.lower().strip()
        if len(keyword.split()) > 4 or keyword in TURKISH_STOP_WORDS: continue
        if len(keyword) < 3 and keyword not in ['ai', 'ml', 'k', 'svm', 'cnn', 'rnn', 'lstm', 'gru', 'gpt', 'gan', 'vae']: continue

        analysis = morphology.analyze(keyword)
        if analysis.analysis_results:
            root = analysis.analysis_results[0].get_stem()
            if len(root) > 2 and root not in TURKISH_STOP_WORDS:
                 clean_keywords.add(root)
        else:
            clean_keywords.add(keyword)

    sorted_clean_keywords = sorted(list(clean_keywords))
    
    try:
        with open(OUTPUT_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(sorted_clean_keywords, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 İşlem Tamamlandı! '{OUTPUT_KEYWORDS_PATH}' dosyası {len(sorted_clean_keywords)} temiz kelime ile oluşturuldu.")
    except Exception as e:
        print(f"❌ HATA: '{OUTPUT_KEYWORDS_PATH}' dosyası yazılırken bir sorun oluştu: {e}")

if __name__ == "__main__":
    clean_and_normalize_keywords()