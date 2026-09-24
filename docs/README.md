# 1Pilot teknik devir dokümanı

Bu belge, 1PAVI içinde çalışan 1Pilot RAG sistemini devralacak geliştirici için
tek başlangıç noktasıdır. Mimariyi, kod sahipliğini, veri akışını, çalıştırma ve
bakım adımlarını birlikte açıklar. Tarihsel deney notları yerine çalışan kod esas
alınmıştır.

Son gözden geçirme: **24 Eylül 2026**

## 1. Sistem ne yapar?

1Pilot, 1PAVI kullanıcılarının kurum içi kullanım kılavuzları hakkında Türkçe
veya İngilizce soru sorabildiği, Jetson üzerinde tamamen yerel çalışan bir RAG
asistanıdır. Yanıtlar yerel kılavuz kanıtına dayanır; veri harici AI servislerine
gönderilmez; yeterli kanıt yoksa sistem tahmin yürütmek yerine güvenli biçimde
reddeder.

İki kullanıcı yüzü aynı RAG çekirdeğini kullanır:

- `app/main.py`: geliştirme, demo ve ayrıntılı tanılama için Streamlit.
- `app/copilot_api.py`: 1PAVI UI/BFF tarafından kullanılan production FastAPI.

## 2. Yüksek seviyeli mimari

```text
1PAVI kullanıcısı
       |
       v
1PAVI UI -- JWT --> 1PAVI BFF -- /api/copilot proxy --> 1Pilot FastAPI
                                                            |
                                                            v
                                                      RAGPipeline
                         +----------------------------------+----------------+
                         |                                                   |
                  sorgu planlama                                      yerel indeksler
             (rewrite/decomposition)                         (Chroma + BM25 + parent)
                         |                                                   |
                         +----------------> RRF adayları <--------------------+
                                                  |
                                                  v
                                    BGE cross-encoder reranking
                                                  |
                                                  v
                                      kanıt güvenilirliği kapısı
                                                  |
                                                  v
                                    Ollama / gpt-oss:20b generation
                                                  |
                                                  v
                                 NDJSON streaming + kaynak + metrik
```

Yerel model rolleri:

- `bge-m3:latest`: sorgu ve doküman embedding'leri (Ollama).
- `BAAI/bge-reranker-v2-m3`: adayları sıralayan Hugging Face cross-encoder.
  Ollama modeli değildir; yerel cache'den yüklenir.
- `gpt-oss:20b`: varsayılan rewrite, decomposition ve cevap üretim modeli.

## 3. Dizin yapısı ve sorumluluklar

```text
app/
  main.py                    Streamlit geliştirme arayüzü
  ingest.py                  Kılavuzlardan atomik indeks üretimi
  rag_pipeline.py            Sorgu aşamalarının orkestrasyonu
  rag_policies.py            Paylaşılan politika yardımcıları
  copilot_api.py             Production HTTP/NDJSON sınırı
  copilot_settings.py        Ortam değişkenleri
  copilot_auth.py            1PAVI RS256 JWT doğrulaması
  copilot_context.py         Kapalı sayfa bağlamı/açıklama prototipi
  copilot_chat_log.py        Güvenli append-only konuşma logu
  copilot_feedback.py        Kullanıcı oylarının JSONL kaydı
  rag_core/
    config.py                Runtime ayarlarının tek kaynağı
    routing.py               Sosyal yol, rewrite ve decomposition
    index_store.py           İndeks yükleme ve manifest doğrulama
    retrieval_engine.py      Dense + BM25 + RRF + reranking
    reranker_backends.py     BGE ve deneysel TensorRT adapter'ları
    evidence.py              Fail-closed kanıt kararı
    semantic_gate.py         Deneysel semantic gate adapter'ı
    answering.py             Prompt, generation ve streaming
    results.py               Tek turn sonuç/telemetry şeması
    models.py                Katmanlar arası veri sınıfları

data/
  user_manuel_dev.md         Salt okunur kurum içi ana kılavuz
  ui-user-guide.md           Salt okunur UI kılavuzu
  part_creation_*.md         Türetilmiş ek TR/EN kılavuzlar
  parent_store.pkl           Parent doküman deposu
  bm25_index.pkl             Sparse retrieval indeksi
  index_manifest.json        Kaynak hash ve indeks bütünlük sözleşmesi

app/chroma_db/               Dense vektör indeksi
tests/                       Çekirdek ve API contract testleri
deploy/                      Taşınabilir release araçları
evaluation/                  Onaylı değerlendirme varlıkları
log/                         Yerel loglar ve korunmuş final sonuçlar
rag_v9/, rag_v10/            Tarihsel snapshot; production değildir
1pavi-copilot/               Mentorun referans PoC submodule'ü; production değildir
documentation/               Draw.io mimari diyagramı ve üretim scripti
```

Yeni değişiklikler güncel `app/` koduna yapılır. `rag_v9`, `rag_v10` ve
`1pavi-copilot` karşılaştırma/geri inceleme içindir.

## 4. Offline ingestion

Ingestion yalnız kılavuz veya chunking değiştiğinde çalışır:

```text
Onaylı Markdown kılavuzlar
        -> başlık-temelli ayrıştırma
        -> 1500/100 parent belgeler
        -> 400/80 child parçalar -> BGE-M3 -> Chroma
        -> parent belgeler ----------------> BM25
        -> parent_store.pkl + index_manifest.json
```

`app/ingest.py` yeni indeksi staging alanında tamamlar. Chroma, BM25, parent
store ve manifest birlikte doğrulanıp yayımlanır; eski sürüm
`data/index_backups/` altına alınır. Yarım indeks production'a geçmez.

Kaynak kılavuzlar değiştirilemez girdilerdir. Ingestion bunları düzenlememeli,
yeniden adlandırmamalı veya silmemelidir.

```bash
cd /home/toyota/intern_faruk/toyota_rag_project
source jetson_env/bin/activate
python app/ingest.py
python -m unittest tests.test_index_manifest
```

## 5. Online soru akışı

### 5.1 Sosyal hızlı yol

Sınırlı selamlaşma/teşekkür mesajları `routing.py` içinde deterministik olarak
yanıtlanır; retrieval ve generation çalışmaz.

### 5.2 Query decomposition

Yapısal olarak birleşik olabilecek soruda planner iki bağımsız bilgi ihtiyacı
olup olmadığına karar verir. Geçerli plan tam iki farklı, bağımsız alt sorgu
üretmelidir; şema hatasında normal tek-sorgu yoluna dönülür. Her alt sorgu ayrı
retrieval'dan geçer, her birinden en fazla bir farklı parent alınır ve toplam
context iki parent'ı aşmaz.

### 5.3 Koşullu rewrite

Rewrite her soruda çalışmaz:

1. Ham kullanıcı sorusu önce doğrudan aranır.
2. Geçmiş yoksa rewrite yapılmaz.
3. Geçmiş varsa ve ham CrossEncoder skoru `0.30` veya üzerindeyse rewrite atlanır.
4. Geçmiş varsa ve skor `0.30` altındaysa son üç kullanıcı turuyla yalnız eksik
   özne/konuşma referansı tamamlanır.
5. Rewrite çıktısıyla en fazla bir ek retrieval geçişi yapılır.

`0.30` yalnız rewrite yönlendirme eşiğidir; kanıt yeterliliği veya cevap
doğruluğu değildir.

### 5.4 Hibrit retrieval ve reranking

- Chroma/BGE-M3 semantic search anlamca benzer bölümleri bulur.
- BM25 ekran adları, hata metinleri, kodlar ve kesin terimleri yakalar.
- Sonuçlar Reciprocal Rank Fusion ile birleştirilip tekilleştirilir.
- Parent belgeler başlık breadcrumb'larıyla tek batch halinde BGE
  cross-encoder'a verilir.

Varsayılan bütçeler `config.py` içindedir: dense `10`, BM25 `10`, rerank `15`,
generation context en fazla `2` parent.

### 5.5 Evidence gate

Production `legacy_ce` profili şu ayrımı yapar:

- BGE `0.03` relevance eşiği ilk yapısal uygunluk sinyalidir.
- Skor `0.30` altındaysa en iyi üç farklı parent üzerinde kısa sufficiency
  kararı alınır.
- `grounded` ise cevap üretilir; `none/off_topic` ise güvenli ret döner.

CrossEncoder skoru relevance ölçüsüdür; doğruluk/faithfulness olasılığı değildir.
Term hit, margin ve consensus yalnız tanılamadır, kabul kararını değiştirmez.

### 5.6 Generation, dil ve hız

Generation yalnız güncel retrieval context'ini teknik gerçeklik kaynağı sayar.
Önceki assistant yanıtı teknik kanıt değildir; yalnız “ilk konu/ikinci adım” gibi
söylem referansını çözebilir.

Yanıt dili ham kullanıcı mesajından TR/EN olarak seçilir; belirsiz takipte son
dili belirlenebilen kullanıcı turu kullanılır. UI adları ve kodlar korunabilir.

`gpt-oss:20b` için `thinking="low"`, `num_ctx=3072` ve sınırlı output bütçesi
kullanılır. Warmup, rewrite ve generation `num_ctx` değerleri hizalı kalmalıdır;
uyumsuzluk Ollama runner reload ve ciddi latency oluşturabilir.

## 6. Production API ve 1PAVI

| Yol | İşlev |
|---|---|
| `GET /health` | Servis ve bağımlılık durumunu gösterir |
| `GET /ready` | Pipeline hazır değilse HTTP 503 |
| `POST /api/v1/chat` | JWT korumalı NDJSON cevap akışı |
| `POST /api/v1/feedback` | Tamamlanan cevap için kullanıcı oyu |

Tarayıcı Copilot'a doğrudan gitmez. 1PAVI BFF aynı-origin proxy uygular:

```text
/api/copilot/* -> http://copilot:49159/api/v1/*
```

Eski host-servis kurulumunda hedef `host.containers.internal:49159` olabilir;
güncel Podman mikroservisinde `copilot` container adı tercih edilir. Hedef kodda
değil compose/environment ayarında tutulur.

NDJSON olayları:

- `status`: `queued` veya `processing`.
- `content`: görünen cevaba eklenecek parça.
- `content_replace`: görünen metni güvenli final cevapla değiştirir.
- `sources`: kullanılan kaynak başlıkları.
- `done`: kanonik cevap, turn id ve güvenli metrikler.
- `error`: iç ayrıntısı gizlenmiş hata.

Model reasoning'i API'den çıkmaz. Reverse-proxy buffering kapalı olmalıdır;
aksi halde tokenlar kullanıcıya toplu ulaşır. Pipeline mutable durum ve resident
modeller tuttuğundan Uvicorn **tek worker** ile çalışır. HTTP istekleri kabul
edilir fakat RAG turları lock ile sıraya alınır.

## 7. Güvenlik

Production API, 1PAVI manager'ın RS256 JWT'sini aynı public key ile doğrular;
beklenen payload `sub`, `role`, `exp` alanlarıdır.

- Production'da `COPILOT_AUTH_REQUIRED=true` kalır.
- Yalnız JWT public key bağlanır; private key kopyalanmaz.
- `COPILOT_AUTH_REQUIRED=false` sadece localhost smoke test içindir.
- Ham kılavuz pasajı, reasoning, JWT ve tam geçmiş loglanmaz.
- `data/`, indeks, log ve model cache'leri public Git'e gönderilmez.
- Kurum içi içerik harici AI/bulut/public repository'ye yüklenmez.

## 8. Runtime profilleri

`app/rag_core/config.py` tek kaynaktır.

### `production_bge` — varsayılan/onaylı

```text
reranker_backend          = bge
context_selection_policy  = legacy_threshold
evidence_gate_backend     = legacy_ce
generation_grounding_mode = legacy
query_decomposition       = enabled
```

### `qwen_candidate` — deneysel, production değildir

```text
reranker_backend          = qwen
context_selection_policy  = rank_only_top2
evidence_gate_backend     = single_call_structured
generation_grounding_mode = strict_context
model                     = qwen2.5:14b-instruct
```

Qwen profili geçmiş kalite sorunları nedeniyle production'a alınmamıştır.
Flag'leri tek tek karıştırmayın; yalnız sabit dataset karşılaştırmasında atomik
profil olarak kullanın.

## 9. Ortam değişkenleri

| Değişken | Production değeri/işlevi |
|---|---|
| `COPILOT_HOST` | Container içinde `0.0.0.0` |
| `COPILOT_PORT` | `49159` |
| `COPILOT_AUTH_REQUIRED` | `true` |
| `JWT_PUBLIC_KEY_PATH` | Volume içindeki public key |
| `COPILOT_RUNTIME_PROFILE` | `production_bge` |
| `COPILOT_MODEL` | `gpt-oss:20b` |
| `COPILOT_OLLAMA_HOST` | Deployment'a göre Ollama HTTP origin'i |
| `COPILOT_WARMUP` | `true` |
| `COPILOT_PAGE_CONTEXT_ENABLED` | `false` (prototip) |
| `COPILOT_CLARIFICATION_ENABLED` | `false` (prototip) |
| `COPILOT_CHAT_LOG_PATH` | Kalıcı log volume'u |
| `COPILOT_FEEDBACK_LOG_PATH` | Kalıcı log volume'u |

Cihaz IP'sini veya JWT yolunu Python koduna sabitlemeyin.

## 10. Çalıştırma

### Streamlit

```bash
cd /home/toyota/intern_faruk/toyota_rag_project
source jetson_env/bin/activate
python -m streamlit run app/main.py
```

### Yerel API smoke testi

```bash
COPILOT_AUTH_REQUIRED=false \
./jetson_env/bin/python -m uvicorn app.copilot_api:app \
  --host 127.0.0.1 --port 49159 --workers 1
```

Başka terminalden:

```bash
curl -s http://127.0.0.1:49159/health
curl -s http://127.0.0.1:49159/ready
curl -sN -X POST http://127.0.0.1:49159/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Kamera nasıl eklenir?","history":[]}'
```

### Production Podman kontrolü

```bash
./scripts/up.sh --status
podman ps --format '{{.Names}} | {{.Status}}'
curl -s http://127.0.0.1:49159/health
curl -s http://127.0.0.1:49159/ready
```

Yeni cihaza release hazırlama, private index paketini doğrulama, rsync ve geri
dönüş adımları çalıştırılabilir araçla birlikte
[`deploy/README.md`](../deploy/README.md) içindedir.

## 11. Güncelleme sınırları

### Yalnız Python/RAG kodu değiştiyse

Testlerden sonra yalnız Copilot image'ını build edip Copilot container'ını yeniden
oluşturun. UI ve diğer 1PAVI servislerini gereksiz yere yeniden başlatmayın.

### Yalnız widget/UI değiştiyse

Yalnız UI image/container'ını yenileyin. RAG indeksi ve Copilot image'ı değişmez.

### Kılavuz değiştiyse

1. Değişikliğin onaylı kaynak/türetilmiş dokümana uygulandığını doğrulayın.
2. `python app/ingest.py` ile dört indeks artifact'ını birlikte üretin.
3. Manifest/hash ve retrieval testlerini çalıştırın.
4. Yeni private-data release'i hazırlayıp atomik taşıyın.

Canlı Chroma SQLite dosyasını elle düzenlemeyin; indeks artifact'larından yalnız
birini tek başına değiştirmeyin.

## 12. Test ve kabul

```bash
./jetson_env/bin/python -m unittest discover -s tests -p 'test_*.py'
./jetson_env/bin/python -m compileall -q app tests deploy
```

Mimari/prompt değişikliğinde ayrıca aynı cihaz ve runtime profile üzerinde onaylı
dataset ile önce/sonra ölçümü yapılmalıdır. Retrieval, doğru cevap, güvenli ret,
dil uyumu, TTFT, toplam süre ve p95 birlikte değerlendirilir. Jetson'da başka
benchmark veya model dönüşümü çalışırken latency sonucu geçerli sayılmaz.

👍/👎 oyları hata keşfi içindir; ground truth değildir.

## 13. Loglar

Geliştirme projesi:

```text
log/chat_logs.log
log/user_feedback_logs.jsonl
```

İkinci Jetson Podman kurulumu:

```text
/mnt/ssd/1pavi-copilot/log/chat_logs.log
/mnt/ssd/1pavi-copilot/log/user_feedback_logs.jsonl
```

```bash
podman logs --tail 200 copilot
```

Chat log; soru/cevap, başlıklar, profile, rewrite, decomposition, retrieval,
rerank, TTFT, generation, token ve toplam süre alanlarını içerir. Loglar kullanıcı
içeriği barındırır; commit edilmez veya haricen paylaşılmaz.

## 14. Sorun giderme

| Belirti | İlk kontrol | Muhtemel neden |
|---|---|---|
| Uzun süre `Queued` | `podman logs copilot`, çalışan turn | Tek pipeline başka isteği işliyor |
| `Searching the guide` bitmiyor | `/ready`, Ollama | Cold start, bağlantı veya pipeline hatası |
| `/health` degraded | response `error` alanı | JWT, indeks, model veya startup bağımlılığı |
| `/ready` 503 | Copilot logu | Pipeline initialize edilemedi |
| İlk soru yavaş | warmup, `ollama ps` | Model resident değil |
| Her soru yavaş | power mode, GPU/RAM, num_ctx | Güç/memory baskısı veya context uyumsuzluğu |
| Cevap toplu geliyor | BFF proxy buffering | NDJSON biriktiriliyor |
| Index hash hatası | `index_manifest.json` | Kılavuz ve indeks farklı sürüm |
| Markdown yıldızı görünüyor | UI image/cache/adres | Eski UI build'i açık |
| UI var, cevap yok | `COPILOT_URL`, network, `/ready` | BFF Copilot'a ulaşamıyor |

Container'ın `healthy` görünmesi gerçek soru turunun başarılı olduğunu tek başına
kanıtlamaz; `/ready` ve UI smoke testi de gerekir.

## 15. Korunacak kurallar

1. Kaynak kılavuzları değiştirmeyin veya silmeyin.
2. Bağımsız soruya konuşma geçmişini eklemeyin.
3. Önce ham sorguyu arayın; rewrite yalnız zayıf geçmiş-bağımlı takipte çalışsın.
4. Dense ve BM25 aynı retrieval katmanında birleşsin.
5. Reranker skorunu cevap doğruluğu olasılığı saymayın.
6. Generation context'i en fazla iki parent olarak kalsın.
7. Assistant geçmişini teknik kanıt saymayın.
8. Warmup/rewrite/generation `num_ctx` değerlerini hizalı tutun.
9. Production'da tek Uvicorn worker kullanın.
10. Reasoning'i UI, API veya loglara göndermeyin.
11. Deney profilini kısmi flag'lerle production'a taşımayın.
12. Tek örnek için özel regex/eşik/bypass eklemeyin; önce sabit setle RCA yapın.

## 16. Bilinen sınırlar

- Yüksek relevance yolunda bağımsız answer-faithfulness modeli yoktur.
- Düşük güven sufficiency çağrısı bazı doğru paraphrase'leri reddedebilir.
- TR/EN dil seçimi deterministiktir; çıktı dili ikinci modelle doğrulanmaz.
- Sayfa bağlamı ve clarification/chip prototipleri varsayılan kapalıdır; açılmadan
  önce kalite ve latency A/B testi gerekir.
- Turlar serialize edilir. Yüksek trafik için aynı pipeline'a worker eklemek yerine
  ayrı servis/memory kapasitesi tasarlanmalıdır.
- Model veya JetPack değişince CUDA/PyTorch/sentence-transformers uyumu yeniden
  doğrulanmalıdır.

## 17. Devir kontrol listesi

1. Bu belgeyi ve `AGENTS.md` güvenlik kurallarını okuyun.
2. `git status` ile kullanıcı değişikliklerini kontrol edin.
3. Ollama'da `gpt-oss:20b` ve `bge-m3:latest` modellerini doğrulayın.
4. BGE reranker cache'ini doğrulayın.
5. İndeks manifest/hash testini çalıştırın.
6. Tüm unit/contract testlerini çalıştırın.
7. Streamlit'te bağımsız, takip ve kanıt-yetersiz soru deneyin.
8. Production'da `/health`, `/ready`, streaming, kaynak ve feedback'i doğrulayın.
9. Değişiklikten önce ilgili kararın tek sahibi olan modülü bulun; aynı kararı
   başka katmanda tekrar uygulamayın.

Bu belge kodla çelişirse çalışan kod ve testler incelenmeli, ardından belge de
kodla birlikte güncellenmelidir.
