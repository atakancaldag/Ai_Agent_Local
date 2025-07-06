import json
import os

# Güncellenecek dosyanın adı
input_file_name = 'data.json'
# Güncellenmiş verinin kaydedileceği dosyanın adı
output_file_name = 'data.json' # Orijinal dosyanın üzerine yazmak için aynı isim

def add_answer2_field(input_path, output_path):
    """
    Belirtilen JSON dosyasındaki her soru objesine 'answer2' alanını ekler.
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

    updated_count = 0
    for item in data:
        # Eğer 'answer2' alanı yoksa, boş string olarak ekle
        if 'answer2' not in item:
            item['answer2'] = ""
            updated_count += 1
        # Eğer varsa, zaten boş olabilir veya dolu olabilir, değiştirmemize gerek yok.
        # Bu betik sadece eksik olanları ekler.

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"'{input_file_name}' başarıyla güncellendi. {updated_count} soruya 'answer2' alanı eklendi.")
        print(f"Güncellenmiş dosya '{output_file_name}' olarak kaydedildi.")
    except Exception as e:
        print(f"Güncellenmiş veri '{output_file_name}' dosyasına yazılırken hata oluştu: {e}")

if __name__ == "__main__":
    add_answer2_field(input_file_name, output_file_name)

