# 1pilot — Streamlit RAG

1pilot, 1PAVI kılavuzları için yerel çalışan bir soru-cevap uygulamasıdır. Streamlit arayüzü; Chroma vektör araması ve BM25'i birleştirir, adayları BGE reranker ile sıralar ve yanıtı yerel Ollama modeliyle üretir. Bu depo **standalone Streamlit sürümünü** içerir; 1PAVI içine gömülen API/Podman servisini içermez.

## Depoda neler var?

- `app/main.py`: sohbet arayüzü ve geri bildirim toplama.
- `app/pages/`: yerel geri bildirim analitiği.
- `app/rag_pipeline.py`, `app/rag_core/`, `app/rag_policies.py`: RAG akışı.
- `app/ingest.py`: kılavuzları yerel indekse dönüştürme.
- `requirements.txt`: mevcut Jetson ortamında doğrulanan Python bağımlılıkları.

Şirket kılavuzları, bunlardan türetilen Chroma/BM25/parent indeksleri, model ağırlıkları, konuşma kayıtları ve kullanıcı geri bildirimleri **bu depoda yoktur**. Bunları GitHub'a eklemeyin.

## Gereksinimler

- Python 3.10+; Jetson'da CUDA uyumlu NVIDIA PyTorch kurulmuş bir Python ortamı.
- Çalışan bir yerel Ollama sunucusu (`http://localhost:11434`).
- Ollama'da `gpt-oss:20b` ve `bge-m3:latest` modelleri (`ollama list` ile kontrol edin).
- BGE reranker ağırlıklarına yerel erişim: `BAAI/bge-reranker-v2-m3`. İlk indirme için ağ gerekebilir; çevrimdışı ortamda Hugging Face önbelleğini önceden hazırlayın.

Jetson'da genel amaçlı `torch` paketini körlemesine kurmayın; cihazın JetPack/CUDA sürümüne uygun PyTorch ortamını kullanın. Bağımlılıkları bu ortam aktifken yükleyin:

```bash
python3 -m pip install -r requirements.txt
```

## Özel kılavuzları yerelde hazırlama

`app/ingest.py` şu dosyaları **yerel** `data/` klasöründe bekler:

```text
data/user_manuel_dev.md
data/ui-user-guide.md
data/part_creation_bc_table_v2.tr.md
data/part_creation_bc_table_v2.en.md
```

Bu dosyaları yalnızca şirketin onayladığı güvenli kanaldan alın. Kılavuz kümesi değişirse `app/ingest.py` içindeki `SOURCE_PATHS` listesini yerel ortamınıza göre güncelleyin ve indeksi yeniden oluşturun. Kaynaklarla indeksin hash'leri eşleşmezse uygulama güvenli biçimde açılmaz; yalnız indeks dosyalarını başka cihazdan kopyalamak yeterli değildir.

## İndeks oluşturma ve çalıştırma

Depo kökünde, Ollama ve yukarıdaki modeller hazırken:

```bash
python3 app/ingest.py
python3 -m streamlit run app/main.py
```

İlk açılışta embedding, reranker ve yanıt modeli belleğe yüklendiği için bekleme olabilir. Tarayıcı adresini Streamlit terminal çıktısından alın. Arayüzde başka bir model seçerseniz o modelin de Ollama'da kurulu olması gerekir.

İndeksleme, `app/chroma_db/` ile `data/index_manifest.json`, `data/parent_store.pkl` ve `data/bm25_index.pkl` üretir. Sohbet ve geri bildirimler `log/` altında yalnız yerel olarak tutulur. Bu üç dizin `.gitignore` kapsamındadır.

## Sorun giderme

- **Eksik kılavuz:** Dört kaynak dosyasının adlarını ve `data/` konumunu kontrol edin.
- **İndeks uyuşmazlığı:** Kılavuzları ve indeksi aynı sürümde tutun; gerekirse `python3 app/ingest.py` ile yeniden oluşturun.
- **Model bulunamadı/bağlantı yok:** `ollama list` ve Ollama servis durumunu kontrol edin.
- **CUDA/PyTorch hatası:** Jetson'a uygun PyTorch kurulumunu ve GPU sürücüsünü doğrulayın. Sadece konteynerin veya web arayüzünün sağlıklı görünmesi gerçek model çıkarımını doğrulamaz.

Bu kod ve kılavuzlarla üretilen yanıtlar, üretim hattındaki kararların yerine geçmez; kaynakları ve iş kurallarını ayrıca doğrulayın.
