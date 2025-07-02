import json
import os

# Güncellenecek dosyanın adı
input_file_name = 'data.json'
# Güncellenmiş verinin kaydedileceği dosyanın adı
output_file_name = 'updated_data.json'

def update_json_data(input_path, output_path):
    """
    Belirtilen JSON dosyasındaki her soru objesine
    'sorulma_sayisi', 'ratings' ve 'current_average' alanlarını ekler.
    """
    if not os.path.exists(input_path):
        print(f"Hata: '{input_path}' dosyası bulunamadı.")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Hata: '{input_path}' dosyası geçerli bir JSON formatında değil.")
        return
    except Exception as e:
        print(f"Dosya okunurken bir hata oluştu: {e}")
        return

    updated_data = []
    for item in data:
        # Her bir soru objesine yeni alanları ekle
        # Eğer zaten varsa üzerine yazmaz, yoksa ekler.
        # Ancak sizin durumunuzda zaten yoklar, bu yüzden direkt ekleyecek.
        item['sorulma_sayisi'] = 0
        item['ratings'] = []
        item['current_average'] = 0.0 # Float olarak başlatmak daha iyi
        updated_data.append(item)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"'{input_file_name}' başarıyla güncellendi ve '{output_file_name}' olarak kaydedildi.")
        print(f"Toplam {len(updated_data)} soru güncellendi.")
    except Exception as e:
        print(f"Güncellenmiş veri '{output_file_name}' dosyasına yazılırken hata oluştu: {e}")

if __name__ == "__main__":
    update_json_data(input_file_name, output_file_name)

