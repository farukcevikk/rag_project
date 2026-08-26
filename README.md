# Toyota RAG Asistanı

Jetson AGX Orin üzerinde, Toyota 1PAVI kullanım kılavuzlarından yanıt üreten yerel bir RAG uygulamasıdır. Sistem verileri dış servislere göndermeden Ollama, Chroma, BM25 ve cross-encoder reranking bileşenleriyle çalışır.

## Özellikler

- Streamlit tabanlı sohbet arayüzü
- Chroma ile vektör arama ve BM25 ile anahtar kelime araması
- Reciprocal Rank Fusion ile hibrit retrieval
- Cross-encoder ile yeniden sıralama
- Takip soruları için koşullu konuşma bağlamı seçimi
- Kanıt güvenilirliği kontrolü ve fail-closed yanıt akışı
- Doğrulanmış indeks manifestosu ile kaynak ve indeks tutarlılığı kontrolü
- Yerel kullanıcı geri bildirimi ve analitik paneli
- Jetson GPU çalışma ortamına uygun model warmup akışı

## Mimari

```text
Yerel kılavuzlar
       |
       v
app/ingest.py --> Chroma + BM25 + parent store + index manifest
       |
       v
Kullanıcı sorusu --> koşullu takip çözümleme
       |
       v
Vektör arama + BM25 --> hibrit sıralama --> cross-encoder reranking
       |
       v
Kanıt güvenilirliği kapısı --> Ollama --> doğrulanmış kaynaklı yanıt
```

## Dizin yapısı

```text
app/
  ingest.py                         Yerel kılavuzlardan indeks üretimi
  main.py                           Streamlit uygulama giriş noktası
  rag_pipeline.py                  Retrieval ve yanıt üretim pipeline'ı
  pages/                            Streamlit analitik paneli
tests/                              Çekirdek birim testleri
```

Kılavuzlar, indeksler, loglar, benchmark sonuçları, sanal ortam ve model binary'leri Git deposuna dahil edilmez. Bu dosyalar `.gitignore` ile yerel tutulur.

## Gereksinimler

- Python 3.10 veya üzeri
- NVIDIA Jetson AGX Orin ve CUDA destekli PyTorch kurulumu
- Ollama
- `bge-m3:latest` embedding modeli
- Kullanılacak üretim modellerinden en az biri: `llama3.1`, `qwen3:8b`, `gpt-oss:20b`, `qwen2.5:3b` veya `gemma3:12b`
- Hugging Face üzerinden indirilebilen `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` modeli

Python paketleri:

```text
numpy
ollama
onnxruntime
pandas
sentence-transformers
streamlit
langchain-community
langchain-core
langchain-text-splitters
rank-bm25
```

Jetson kurulumlarında PyTorch ve CUDA uyumlu paketler sistemin CUDA sürümüne göre kurulmalıdır.

## Yerel kurulum

1. Ollama servisini başlatın ve gerekli modelleri indirin:

   ```bash
   ollama serve
   ollama pull bge-m3:latest
   ollama pull qwen2.5:3b
   ```

2. Gizli kılavuz dosyalarını yerel olarak aşağıdaki konumlara koyun:

   ```text
   data/user_manuel_dev.md
   data/ui-user-guide.md
   ```

   Kılavuz içeriklerini GitHub'a veya başka bir dış servise yüklemeyin.

3. Proje sanal ortamını etkinleştirip Python paketlerini kurun:

   ```bash
   source .venv/bin/activate
   pip install numpy ollama onnxruntime pandas sentence-transformers streamlit \
     langchain-community langchain-core langchain-text-splitters rank-bm25
   ```

4. Yerel retrieval indekslerini oluşturun:

   ```bash
   python app/ingest.py
   ```

   Bu işlem `app/chroma_db/`, `data/parent_store.pkl`, `data/bm25_index.pkl` ve `data/index_manifest.json` üretir. Üretilen dosyalar yerel çalışma verisidir.

## Uygulamayı çalıştırma

```bash
streamlit run app/main.py
```

Ollama varsayılan olarak `http://localhost:11434` adresinde çalışmalıdır. Model seçimi uygulamanın sidebar bölümünden yapılır.

## Testler

Çekirdek testleri çalıştırmak için:

```bash
python -m unittest tests.test_condense_question \
  tests.test_index_manifest \
  tests.test_reranker_selection
```

Sözdizimi kontrolü:

```bash
python -m compileall -q app tests
```

Testlerin import edilebilmesi için Python bağımlılıklarının kurulmuş olması gerekir. Ollama servisi, gerçek kılavuz dosyaları veya üretilmiş indeksler test çalıştırmak için zorunlu değildir.

## Gizlilik ve güvenlik

- Kılavuz içerikleri yalnızca yerel retrieval kaynağı olarak kullanılır.
- Kullanıcı soruları ve geri bildirimler yerel `log/` klasörüne yazılır.
- `data/`, `evaluation/`, `log/`, `jetson_env/`, `app/chroma_db/` ve model dosyaları Git'e gönderilmemelidir.
- Paylaşım öncesinde staged dosya listesini kontrol edin:

  ```bash
  git diff --cached --name-only
  ```
