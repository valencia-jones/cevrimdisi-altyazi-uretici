"""Çevrimdışı altyazı üretici - yerel web sitesi.

Video dosyaları bilgisayarınızdan çıkmaz. Model bir kez indirildikten sonra
internet bağlantısı gerekmez.
"""
import os
import threading
import time
import uuid
import webbrowser

from flask import Flask, jsonify, request, send_from_directory, Response

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "yuklenenler")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
app.config["MAX_CONTENT_LENGTH"] = None  # boyut sınırı yok (çok saatlik videolar)

JOBS = {}
LOCK = threading.Lock()
MODEL_CACHE = {}
# Aynı anda tek iş çalışsın (bellek/CPU için)
RUN_LOCK = threading.Lock()


def ts(seconds, sep=","):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def build(segments, fmt):
    if fmt == "srt":
        out = []
        for i, s in enumerate(segments, 1):
            out.append(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n")
        return "\n".join(out)
    if fmt == "vtt":
        out = ["WEBVTT\n"]
        for s in segments:
            out.append(f"{ts(s['start'], '.')} --> {ts(s['end'], '.')}\n{s['text']}\n")
        return "\n".join(out)
    return "\n".join(s["text"] for s in segments)


def get_model(name):
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    from faster_whisper import WhisperModel

    try:
        # Önce yalnızca yerel dosyalar (tamamen çevrimdışı)
        model = WhisperModel(name, device="auto", compute_type="auto", local_files_only=True)
    except Exception:
        # İlk kullanımda model indirilir (internet gerekir)
        model = WhisperModel(name, device="auto", compute_type="auto")
    MODEL_CACHE.clear()
    MODEL_CACHE[name] = model
    return model


def load_audio(path):
    """Videodan sesi ffmpeg ile 16 kHz mono float32 olarak çıkarır.

    Whisper'ın kendi PyAV tabanlı okuyucusu yerine bunu kullanırız; böylece
    eski/uyumsuz PyAV sürümlerinden kaynaklanan hatalar (ör. 'metadata_errors') oluşmaz.
    """
    import subprocess

    import numpy as np

    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", path, "-vn",
           "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        msg = p.stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError("Videodan ses çıkarılamadı: " + (msg or "ses akışı bulunamadı"))
    return np.frombuffer(p.stdout, dtype=np.float32)


def ensure_translator(src, job):
    """src -> Türkçe çeviri paketlerini hazırlar (yoksa bir kez indirir).

    Doğrudan paket yoksa İngilizce üzerinden (src->en, en->tr) çevrilir.
    """
    import argostranslate.package as pkg
    import argostranslate.translate as tr

    def ready(a, b):
        langs = {l.code: l for l in tr.get_installed_languages()}
        if a not in langs or b not in langs:
            return False
        return langs[a].get_translation(langs[b]) is not None

    if ready(src, "tr"):
        return
    job["message"] = "Çeviri paketleri indiriliyor (ilk seferde internet gerekir)..."
    pkg.update_package_index()
    avail = pkg.get_available_packages()

    def install(a, b):
        p = next((x for x in avail if x.from_code == a and x.to_code == b), None)
        if p is None:
            return False
        pkg.install_from_path(p.download())
        return True

    if not install(src, "tr"):
        if src != "en" and not ready(src, "en") and not install(src, "en"):
            raise RuntimeError(f"'{src}' dilinden Türkçeye çeviri paketi bulunamadı.")
        if not ready("en", "tr") and not install("en", "tr"):
            raise RuntimeError("İngilizce-Türkçe çeviri paketi bulunamadı.")
    if not ready(src, "tr"):
        raise RuntimeError("Çeviri paketi kuruldu ama kullanılamadı.")


def worker(job_id, path, model_name, language, task):
    job = JOBS[job_id]
    try:
        job["message"] = "Sıra bekleniyor..."
        with RUN_LOCK:
            job["state"] = "running"
            job["message"] = "Model yükleniyor (ilk seferde indirme gerekebilir)..."
            model = get_model(model_name)
            job["message"] = "Ses çözülüyor ve yazıya çevriliyor..."
            job["message"] = "Videodan ses çıkarılıyor..."
            audio = load_audio(path)
            job["message"] = "Ses çözülüyor ve yazıya çevriliyor..."
            # Whisper yalnızca kaynak dilde yazıya döker; Türkçe çeviriyi ayrıca yaparız.
            segs, info = model.transcribe(
                audio,
                language=None if language == "auto" else language,
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
            )
            src = info.language
            translate_to_tr = (task == "tr" and src != "tr")
            translate_fn = None
            if translate_to_tr:
                ensure_translator(src, job)
                import argostranslate.translate as argos

                def translate_fn(t, _src=src):
                    try:
                        return argos.translate(t, _src, "tr") or t
                    except Exception:  # noqa: BLE001
                        return t  # çeviri başarısızsa özgün metni koru

            job["message"] = (
                f"Ses çözülüyor ({src}) ve Türkçeye çevriliyor..."
                if translate_to_tr else "Ses çözülüyor ve yazıya çevriliyor..."
            )
            duration = info.duration or 0
            job["duration"] = duration
            start = time.time()
            for s in segs:
                if job.get("cancel"):
                    job["state"] = "cancelled"
                    job["message"] = "İptal edildi."
                    return
                text = s.text.strip()
                if not text:
                    continue
                if translate_fn:
                    text = translate_fn(text)
                job["segments"].append({"start": s.start, "end": s.end, "text": text})
                if duration:
                    job["progress"] = min(0.999, s.end / duration)
                elapsed = time.time() - start
                if job["progress"] > 0.01:
                    job["eta"] = int(elapsed / job["progress"] - elapsed)
            job["progress"] = 1.0
            job["state"] = "done"
            job["sub_lang"] = "tur" if (translate_to_tr or src == "tr") else None
            job["message"] = f"Tamamlandı. Algılanan dil: {src}" + (
                " → Türkçeye çevrildi." if translate_to_tr else "")
    except Exception as e:  # noqa: BLE001
        job["state"] = "error"
        job["message"] = f"Hata: {e}"
    finally:
        # Başarılıysa video, "videoya göm" adımı için saklanır; aksi halde silinir.
        if job.get("state") != "done":
            try:
                os.remove(path)
            except OSError:
                pass


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Dosya yok"), 400
    job_id = uuid.uuid4().hex[:12]
    safe = os.path.basename(f.filename or "video")
    path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe}")
    f.save(path)
    JOBS[job_id] = {
        "state": "queued", "progress": 0.0, "message": "Sıraya alındı",
        "segments": [], "duration": 0, "eta": None, "name": safe,
        "path": path, "export": {"state": "none", "progress": 0.0, "message": ""},
    }
    threading.Thread(
        target=worker,
        args=(job_id, path, request.form.get("model", "small"),
              request.form.get("language", "auto"),
              request.form.get("task", "transcribe")),
        daemon=True,
    ).start()
    return jsonify(id=job_id)


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Bulunamadı"), 404
    after = int(request.args.get("after", 0))
    return jsonify(
        state=job["state"], progress=job["progress"], message=job["message"],
        duration=job["duration"], eta=job["eta"], total=len(job["segments"]),
        new=job["segments"][after:],
    )


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return jsonify(ok=True)


@app.route("/api/download/<job_id>.<fmt>")
def download(job_id, fmt):
    job = JOBS.get(job_id)
    if not job or fmt not in ("srt", "vtt", "txt"):
        return "Bulunamadı", 404
    base = os.path.splitext(job["name"])[0]
    body = build(job["segments"], fmt)
    return Response(
        body, mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{base}.{fmt}"},
    )


def ffmpeg_exe():
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

# Kalite ön ayarları: (x264 preset, x264 CRF). CRF ne kadar düşükse kalite o kadar yüksek.
QUALITY = {
    "fast": ("veryfast", 21),   # hızlı, boyut küçük, hafif kalite kaybı
    "high": ("medium", 18),     # varsayılan: gözle fark edilmeyecek kayıp
    "max": ("slow", 16),        # en iyi kalite, en yavaş, en büyük dosya
}
X265_PRESET = {"fast": "fast", "high": "medium", "max": "slow"}
# mp4 içine yeniden kodlamadan kopyalanabilen ses türleri
COPYABLE_AUDIO = {"aac", "mp3", "ac3", "eac3"}

_ENCODERS = None


def has_encoder(name):
    """Kullanılan ffmpeg sürümünde bu kodlayıcı var mı?"""
    import re
    import subprocess

    global _ENCODERS
    if _ENCODERS is None:
        p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, encoding="utf-8", errors="replace")
        _ENCODERS = p.stdout or ""
    return re.search(rf"\b{re.escape(name)}\b", _ENCODERS) is not None


def probe_media(path):
    """Videonun piksel biçimi, renk bilgisi ve ses türlerini okur (ffprobe gerekmez)."""
    import re
    import subprocess

    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace")
    info = {"pix_fmt": "", "ten_bit": False, "color": None, "audio": []}
    got_video = False
    for line in p.stderr.splitlines():
        if "Stream #" not in line:
            continue
        if " Video:" in line and not got_video and "attached pic" not in line:
            got_video = True
            m = re.search(r"\b(yuv[0-9a-z]+|p01\w+|nv\d+)\b(?:\(([^)]*)\))?", line.split(" Video:", 1)[1])
            if m:
                info["pix_fmt"] = m.group(1)
                info["ten_bit"] = "10" in m.group(1)
                if m.group(2):
                    parts = [x.strip() for x in m.group(2).split(",")]
                    for part in parts:
                        if "/" in part:
                            sp, pr, trc = (part.split("/") + ["", ""])[:3]
                            info["color"] = (sp, pr, trc)
                        elif part not in ("tv", "pc", "progressive", "unknown") and not info["color"]:
                            info["color"] = (part, part, part)
        elif " Audio:" in line:
            codec = line.split(" Audio:", 1)[1].strip().split(" ", 1)[0].strip(",")
            info["audio"].append(codec.lower())
    return info


def hard_codec_args(info, quality):
    """Yakılmış altyazı için video/ses kodlama ayarları. (argümanlar, açıklama) döner."""
    preset, crf = QUALITY.get(quality, QUALITY["high"])
    key = quality if quality in QUALITY else "high"

    if info["ten_bit"] and has_encoder("libx265"):
        # 10-bit / HDR kaynak (ör. telefon videoları): 8-bit'e indirmek renkleri soldurur
        v = ["-c:v", "libx265", "-preset", X265_PRESET[key], "-crf", str(crf + 2),
             "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", "log-level=error"]
        if info["color"]:
            sp, pr, trc = info["color"]
            for flag, val in (("-colorspace", sp), ("-color_primaries", pr), ("-color_trc", trc)):
                if val and val != "unknown":
                    v += [flag, val]
        label = "HEVC 10-bit (özgün renk korunuyor)"
    else:
        v = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-profile:v", "high", "-pix_fmt", "yuv420p"]
        label = "H.264"

    if info["audio"] and all(a in COPYABLE_AUDIO for a in info["audio"]):
        a = ["-c:a", "copy"]          # ses aynen kopyalanır: kayıp yok
    else:
        a = ["-c:a", "aac", "-b:a", "256k"]
    return v + a, label


def export_worker(job_id, mode, size, quality="high"):
    """Altyazıyı videoya gömer. mode: 'hard' (yakılmış) veya 'soft' (ayrı iz)."""
    import subprocess

    job = JOBS[job_id]
    ex = job["export"]
    ex.update(state="queued", progress=0.0, message="Sıra bekleniyor...", file=None)
    try:
        with RUN_LOCK:
            ex.update(state="running", message="Video işleniyor...")
            src = job["path"]
            if not os.path.exists(src):
                raise RuntimeError("Özgün video dosyası bulunamadı; altyazıyı yeniden oluşturun.")
            base = os.path.splitext(job["name"])[0]
            ext = os.path.splitext(src)[1].lower()
            work = os.path.join(UPLOAD_DIR, f"exp_{job_id}")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "sub.srt"), "w", encoding="utf-8") as f:
                f.write(build(job["segments"], "srt"))
            exe = ffmpeg_exe()

            if mode == "hard":
                out_ext = "mp4"
                style = (f"FontName=Arial,FontSize={size},PrimaryColour=&H00FFFFFF,"
                         "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=24")
                codec_args, label = hard_codec_args(probe_media(src), quality)
                ex["message"] = f"Video işleniyor ({label})..."
                cmd = [exe, "-y", "-i", src, "-map", "0:v:0", "-map", "0:a?",
                       "-vf", f"subtitles=sub.srt:force_style='{style}'"]
                cmd += codec_args + ["-movflags", "+faststart"]
            else:
                if ext in (".mp4", ".m4v", ".mov"):
                    out_ext, scodec = "mp4", "mov_text"
                else:
                    out_ext, scodec = "mkv", "srt"
                cmd = [exe, "-y", "-i", src, "-i", "sub.srt", "-map", "0:v", "-map", "0:a?",
                       "-map", "1:0", "-c", "copy", "-c:s", scodec]
                if job.get("sub_lang"):
                    cmd += ["-metadata:s:s:0", f"language={job['sub_lang']}",
                            "-metadata:s:s:0", "title=Türkçe"]
            out_name = f"cikti.{out_ext}"
            cmd += ["-progress", "pipe:1", "-nostats", out_name]

            p = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            tail = []
            dur = job.get("duration") or 0
            for line in p.stdout:
                line = line.strip()
                if line.startswith(("out_time_us=", "out_time_ms=")) and dur:
                    try:
                        us = int(line.split("=", 1)[1])
                        ex["progress"] = max(0.0, min(0.999, us / 1e6 / dur))
                    except ValueError:
                        pass
                elif "=" not in line:
                    tail.append(line)
                    tail = tail[-12:]
            if p.wait() != 0:
                raise RuntimeError("ffmpeg hatası: " + " | ".join(tail[-4:]))
            ex.update(state="done", progress=1.0, message="Hazır.",
                      file=os.path.join(work, out_name),
                      name=f"{base}_altyazili.{out_ext}")
    except Exception as e:  # noqa: BLE001
        ex.update(state="error", message=f"Hata: {e}")


@app.route("/api/export/<job_id>", methods=["POST"])
def export_start(job_id):
    job = JOBS.get(job_id)
    if not job or job["state"] != "done":
        return jsonify(error="Önce altyazı oluşturulmalı"), 400
    if job["export"]["state"] in ("queued", "running"):
        return jsonify(ok=True)
    mode = request.form.get("mode", "hard")
    try:
        size = max(10, min(60, int(request.form.get("size", 24))))
    except ValueError:
        size = 24
    quality = request.form.get("quality", "high")
    threading.Thread(target=export_worker, args=(job_id, mode, size, quality), daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/export_status/<job_id>")
def export_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Bulunamadı"), 404
    ex = job["export"]
    return jsonify(state=ex["state"], progress=ex["progress"], message=ex["message"])


@app.route("/api/export_download/<job_id>")
def export_download(job_id):
    from flask import send_file

    job = JOBS.get(job_id)
    ex = job["export"] if job else None
    if not ex or ex.get("state") != "done" or not os.path.exists(ex.get("file", "")):
        return "Hazır değil", 404
    return send_file(ex["file"], as_attachment=True, download_name=ex["name"])


def clean_upload_dir():
    """Önceki çalıştırmalardan kalan geçici dosyaları temizler."""
    import shutil

    for n in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, n)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    clean_upload_dir()
    url = "http://127.0.0.1:5000"
    print(f"\nAltyazı sitesi çalışıyor: {url}\nKapatmak için Ctrl+C\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
