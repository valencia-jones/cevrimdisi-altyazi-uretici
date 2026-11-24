# Çevrimdışı Altyazı Üretici

Videoyu internete yüklemeden, bilgisayarınızda altyazı (SRT / VTT / TXT) üretir.
İki saat ve üzeri videolar için tasarlandı. Tamamen ücretsiz, kullanım sınırı yok.

## Kurulum (bir kez)

1. Python 3.9+ kurun: https://www.python.org/downloads/
   (Windows'ta kurulumda "Add Python to PATH" kutusunu işaretleyin.)
2. Klasörde terminal / komut istemi açın ve çalıştırın:

   ```
   pip install -r requirements.txt
   ```

3. Videolardan sesi okumak için faster-whisper kendi içinde ffmpeg bileşenini
   (PyAV) getirir; ayrıca ffmpeg kurmanız gerekmez.

## Çalıştırma

```
python app.py
```

Tarayıcı kendiliğinden http://127.0.0.1:5000 adresini açar.
Videoyu sürükleyin, dil ve modeli seçin, "Altyazıyı oluştur"a basın.

## Altyazıyı videoya gömme

Altyazı hazır olunca sayfada "Altyazıyı videoya göm" bölümü açılır:

- **Yakılmış (hard):** Yazı görüntünün üzerine işlenir, her oynatıcıda ve telefonda görünür,
  kapatılamaz. Video yeniden kodlanır (.mp4); 2 saatlik videoda uzun sürebilir.
  Kalite seçeneği: Hızlı / Yüksek (önerilen) / En yüksek. Kalite arttıkça süre ve dosya boyutu artar.
  Ses mümkünse (AAC, MP3, AC3) yeniden kodlanmadan aynen kopyalanır. 10-bit / HDR kaynaklar
  (çoğu yeni telefon videosu) renkleri korumak için HEVC 10-bit olarak kodlanır.
- **Ayrı iz (soft):** Altyazı videoya ayrı bir iz olarak eklenir (.mp4 ya da .mkv).
  Görüntü yeniden kodlanmaz, birkaç dakikada biter; oynatıcıdan açılıp kapatılır.
  Bazı oynatıcılar (ör. bazı telefon galerileri, sosyal medya) ayrı izi göstermez.

ffmpeg, `imageio-ffmpeg` paketiyle birlikte gelir; sisteminizde ffmpeg kuruluysa o kullanılır.
Özgün video, gömme için yüklenenler/ klasöründe tutulur; program her açıldığında bu klasör temizlenir.
Diskte videonuzun boyutu kadar ek yer gerekir (özgün kopya + çıktı).

## Türkçe altyazı nasıl üretilir?

1. Whisper konuşmayı kaynak dilde (ör. İngilizce) yazıya döker.
2. Argos Translate her satırı Türkçeye çevirir (tamamen yerel, ücretsiz).
   Doğrudan paket yoksa İngilizce üzerinden çevrilir (kaynak → İngilizce → Türkçe).

"Aynı dilde altyazı" seçeneği çeviri yapmadan yalnızca yazıya döker.
Video zaten Türkçeyse çeviri atlanır.

## İnternet ne zaman gerekir?

Yalnızca ilk kullanımda: Whisper modeli ve (kaynak dil için) çeviri paketi bir kez indirilir.
Sonrasında tamamen çevrimdışı çalışır.
Whisper dosyaları ~/.cache/huggingface, çeviri paketleri Argos'un kendi klasöründe saklanır.

Not: Makine çevirisi insan çevirisi kadar kusursuz değildir; önemli işlerde çıktıyı gözden geçirin.

## Model seçimi

| Model     | Boyut   | Hız       | Doğruluk |
|-----------|---------|-----------|----------|
| tiny      | ~75 MB  | çok hızlı | düşük    |
| base      | ~145 MB | hızlı     | orta     |
| small     | ~460 MB | dengeli   | iyi      |
| medium    | ~1.5 GB | yavaş     | çok iyi  |
| large-v3  | ~3 GB   | çok yavaş | en iyi   |

Türkçe için en az `small`, mümkünse `medium` önerilir.
Sadece işlemci (CPU) ile 2 saatlik video `small` modelle yaklaşık 40–90 dakika
sürebilir; NVIDIA ekran kartı varsa (CUDA kurulu) birkaç dakikaya iner.

## Notlar

- Aynı anda tek video işlenir, diğerleri sıraya girer.
- Yüklenen geçici dosya işlem bitince silinir.
- Sunucu yalnızca kendi bilgisayarınızdan (127.0.0.1) erişilebilir.
