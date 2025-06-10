import json

# Kontrol edilecek JSON dosyasının adı
DATASET_PATH = 'data.json'

def count_entries():
    """
    JSON dosyasındaki soru-cevap çiftlerinin sayısını sayar ve ekrana yazdırır.
    """
    try:
        # Dosyayı UTF-8 formatında oku
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 'data' bir liste olduğu için, uzunluğu (len) bize eleman sayısını verir.
        entry_count = len(data)
        
        print("\n" + "="*40)
        print(f"📊 Veri Seti Raporu 📊")
        print("="*40)
        print(f"'{DATASET_PATH}' dosyasında toplam {entry_count} adet soru-cevap çifti bulunmaktadır.")
        print("="*40 + "\n")

    except FileNotFoundError:
        print(f"\n❌ HATA: '{DATASET_PATH}' adında bir dosya bulunamadı. Lütfen dosya adını kontrol edin.\n")
    except json.JSONDecodeError:
        print(f"\n❌ HATA: '{DATASET_PATH}' dosyası boş veya bozuk. Lütfen içeriğini kontrol edin.\n")
    except Exception as e:
        print(f"\n❌ Beklenmedik bir hata oluştu: {e}\n")

if __name__ == "__main__":
    count_entries()