# Tarımda Görüntü İşleme: Yaprak Hastalığı ve Bitki Sağlığı Analizi

Görüntü İşleme Dersi proje ödevi kapsamında geliştirilen bu uygulama, yaprak fotoğrafları üzerinden hastalık belirtilerini (sararma/kloroz, yanıklık ve koyu lekeler) otomatik olarak tespit eder ve hastalık şiddetini (%) hesaplar.

## Kullanılan Yöntemler ve İşleme Hattı
1. **Ön İşleme (Filtreleme):** Medyan (doğrusal olmayan) veya Gauss (doğrusal) filtreleme ile gürültü temizleme.
2. **Arka Plan Segmentasyonu:** CIELAB (`255 - a*` ve `b*`) kanalları üzerinde Otsu otomatik eşikleme, morfolojik açma/kapama ve en büyük bağlı bileşen seçimi ile yaprağı arka plandan ayırma.
3. **Hastalık Bölgesi Tespiti:** HSV renk uzayında sarı/kahverengi lekelerin ve uyarlamalı parlaklık eşiğiyle koyu nekroz/scab bölgelerinin maskelenmesi.
4. **Sayısal Değerlendirme:** Toplam yaprak piksel alanına göre hastalık oranının (%) hesaplanması ve durum sınıflandırması.

## Kullanılan Kütüphaneler
- **OpenCV (`cv2`):** Görüntü filtreleme, renk uzayı dönüşümleri (`BGR → LAB`, `BGR → HSV`), Otsu eşikleme, morfolojik işlemler ve kontur çizimi.
- **NumPy:** Piksel maskesi ve matris işlemleri.
- **Streamlit & Pillow:** Web tabanlı kullanıcı arayüzü, görsel yükleme ve canlı parametre kontrolü.

## Kurulum ve Çalıştırma
```bash
git clone https://github.com/erenaslandag/leaf-disease-analyzer.git
cd leaf-disease-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```