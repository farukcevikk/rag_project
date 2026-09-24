# 1Pilot — Yerel ve Kaynak Odaklı RAG Asistanı

**1Pilot**, 1PAVI kullanıcı kılavuzlarını operatörlerin ve mühendislerin doğal dille sorgulayabilmesini sağlayan, tamamen yerel çalışan bir yapay zekâ asistanıdır.

Uzun teknik dokümanlarda doğru bölümü elle aramak yerine kullanıcı sorusunu Türkçe veya İngilizce sorar; 1Pilot ilgili kılavuz bölümlerini bulur, kanıtları yeniden sıralar ve cevabı kaynak göstererek üretir. Tüm arama ve üretim süreci yerel cihazda çalışabildiği için kurumsal dokümanların harici bir yapay zekâ servisine gönderilmesi gerekmez.

> Bu depo, projenin **standalone Streamlit sürümünü** içerir. 1PAVI arayüzüne gömülü FastAPI/Podman dağıtımı bu deponun kapsamında değildir.

## Neden 1Pilot?

1PAVI gibi çok sayıda ekran, cihaz ve iş kuralı içeren sistemlerde doğru bilgiye hızlı ulaşmak kritik öneme sahiptir. 1Pilot bu ihtiyacı şu kazanımlarla karşılar:

- **Kılavuzla sınırlandırılmış cevaplar:** Yanıtlar bulunan doküman kanıtlarına dayandırılır; yeterli kanıt yoksa sistem bunu açıkça belirtir.
- **Türkçe ve İngilizce kullanım:** Soru dili algılanır ve cevap aynı dilde oluşturulur.
- **Hibrit arama:** Anlamsal benzerlik ile tam teknik terim eşleşmesi birlikte kullanılır.
- **Kaynak gösterimi:** Kullanıcı, cevabın hangi kılavuz başlığına dayandığını görebilir.
- **Yerel çalışma:** Ollama, Chroma ve reranker modelleri cihaz üzerinde çalışır; hassas kılavuzların buluta gönderilmesi gerekmez.
- **İzlenebilir performans:** Retrieval, reranking, ilk token süresi, toplam süre ve token metrikleri arayüzden incelenebilir.
- **Geri bildirim döngüsü:** Kullanıcıların olumlu/olumsuz değerlendirmeleri yerel olarak kaydedilerek sonraki iyileştirmelere veri sağlar.

## Sistem nasıl çalışır?

```text
Kullanıcı sorusu
      │
      ▼
Dil ve konuşma bağlamı çözümleme
      │
      ▼
Chroma anlamsal arama ──┐
                        ├─► Birleştirme ve tekrar temizleme
BM25 anahtar kelime ────┘
                                  │
                                  ▼
                         BGE cross-encoder reranking
                                  │
                                  ▼
                         Kanıt yeterliliği kontrolü
                                  │
                        ┌─────────┴─────────┐
                        ▼                   ▼
                Kaynaklı LLM cevabı   Güvenli yetersiz-kanıt yanıtı
```

Akışın önemli aşamaları:

1. Kılavuzlar başlık yapısı korunarak parçalara ayrılır ve metadata ile indekslenir.
2. Kullanıcı sorusu hem BGE-M3 tabanlı vektör aramasına hem BM25 aramasına gönderilir.
3. İki aramadan gelen adaylar birleştirilir ve `BAAI/bge-reranker-v2-m3` ile yeniden sıralanır.
4. Konuşma geçmişine bağlı veya zayıf eşleşen sorularda koşullu query rewrite uygulanabilir.
5. Evidence gate, bulunan içeriğin cevap üretmek için yeterli olup olmadığını kontrol eder.
6. Yerel Ollama modeli yalnız seçilen bağlamı kullanarak cevabı ve kaynak bilgisini üretir.

Bu tasarım, yalnızca “en benzer metni bulup LLM'e gönderme” yaklaşımından daha kontrollü ve açıklanabilir bir RAG akışı sunar.

## Kullanılan teknoloji

| Katman | Teknoloji | Görevi |
|---|---|---|
| Kullanıcı arayüzü | Streamlit | Sohbet, kaynaklar, performans metrikleri ve geri bildirim |
| Embedding | `bge-m3:latest` (Ollama) | Çok dilli anlamsal temsil |
| Vektör veritabanı | Chroma | Anlamsal aday bulma |
| Sözcüksel arama | BM25 | Teknik terim ve birebir ifade yakalama |
| Reranker | `BAAI/bge-reranker-v2-m3` | En ilgili kılavuz bölümlerini öne çıkarma |
| Yanıt modeli | `gpt-oss:20b` (Ollama) | Yerel ve kaynaklandırılmış cevap üretme |
| Uygulama dili | Python 3.10+ | Ingestion, retrieval ve orchestration |

Varsayılan çalışma profili, yanıt kalitesi ile Jetson üzerindeki gecikme arasında denge kuracak şekilde hazırlanmıştır. Model, context ve eşik ayarları [app/rag_core/config.py](app/rag_core/config.py) içinde görülebilir.

## Depo yapısı

```text
app/
├── main.py                 # Streamlit sohbet arayüzü
├── ingest.py               # Kılavuzları indeksleme işlemi
├── rag_pipeline.py         # Uçtan uca RAG orkestrasyonu
├── rag_policies.py         # Prompt ve davranış politikaları
├── rag_core/               # Retrieval, reranking, evidence ve model katmanları
└── pages/                  # Yerel geri bildirim analitiği

docs/README.md              # Mimari ve devralma dokümantasyonu
requirements.txt            # Python bağımlılıkları
```

Kapsamlı mimari açıklama ve bakım rehberi için [docs/README.md](docs/README.md) dosyasına bakın.

## Hızlı başlangıç

### 1. Gereksinimler

- Python 3.10+
- Çalışan bir Ollama sunucusu: `http://localhost:11434`
- Ollama üzerinde `gpt-oss:20b` ve `bge-m3:latest`
- Yerel veya önceden indirilmiş `BAAI/bge-reranker-v2-m3`
- Jetson kullanılıyorsa JetPack/CUDA sürümüyle uyumlu NVIDIA PyTorch ortamı

Model durumunu kontrol edin:

```bash
ollama list
curl http://localhost:11434/api/tags
```

Eksik Ollama modellerini ağ erişimi olan bir ortamda indirin:

```bash
ollama pull gpt-oss:20b
ollama pull bge-m3:latest
```

> Jetson üzerinde PyPI'daki genel `torch` paketini körlemesine kurmayın. JetPack ve CUDA sürümünüzle uyumlu NVIDIA PyTorch kurulumunu kullanın.

### 2. Python ortamını hazırlayın

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Jetson'da zaten doğrulanmış bir CUDA/PyTorch ortamınız varsa yeni sanal ortam yerine onu etkinleştirin.

### 3. Kılavuzları yerleştirin

`app/ingest.py` varsayılan olarak aşağıdaki özel kaynakları yerel `data/` klasöründe bekler:

```text
data/user_manuel_dev.md
data/ui-user-guide.md
data/part_creation_bc_table_v2.tr.md
data/part_creation_bc_table_v2.en.md
```

Bu dosyalar şirket içi içerik oldukları için GitHub deposuna dahil edilmemiştir. Yalnızca onaylı ve güvenli bir kanaldan temin edin. Farklı dosyalar kullanacaksanız `app/ingest.py` içindeki `SOURCE_PATHS` listesini güncelleyin.

### 4. Arama indeksini oluşturun

Depo kökünde:

```bash
python3 app/ingest.py
```

Bu işlem şu yerel çıktıları üretir:

```text
app/chroma_db/
data/index_manifest.json
data/parent_store.pkl
data/bm25_index.pkl
```

Manifest, kaynak dosyalarla indeksin aynı sürüme ait olduğunu doğrular. Kılavuzlar değiştiğinde ingestion işlemini yeniden çalıştırın.

### 5. Uygulamayı başlatın

```bash
python3 -m streamlit run app/main.py
```

Terminalde gösterilen yerel adresi tarayıcıda açın. İlk çalıştırmada embedding, reranker ve üretim modeli belleğe alınacağı için sonraki sorulara göre daha uzun bekleme görülebilir.

## Kayıtlar ve geri bildirim

Çalışma sırasında sohbet kayıtları, performans metrikleri ve kullanıcı geri bildirimleri yerel `log/` dizinine yazılır. Streamlit içindeki analitik sayfası geri bildirimleri incelemek için kullanılabilir.

`data/`, `app/chroma_db/` ve `log/` dizinleri gizlilik nedeniyle `.gitignore` kapsamındadır. Bu dizinleri, model ağırlıklarını veya şirket dokümanlarını GitHub'a eklemeyin.

## Sık karşılaşılan sorunlar

| Belirti | Kontrol |
|---|---|
| Kılavuz bulunamıyor | Dört kaynak dosyasının adını ve `data/` konumunu kontrol edin. |
| İndeks uyuşmazlığı | Kaynak değiştiyse `python3 app/ingest.py` komutunu yeniden çalıştırın. |
| Ollama bağlantı hatası | Ollama servis durumunu, `ollama list` ve `http://localhost:11434` erişimini kontrol edin. |
| Reranker indirilemiyor | Hugging Face model önbelleğini çevrimdışı cihaza önceden aktarın. |
| CUDA/PyTorch hatası | JetPack, CUDA ve NVIDIA PyTorch sürümlerinin uyumunu doğrulayın. |
| İlk soru çok yavaş | Model warm-up süresini ve cihazın güç/bellek durumunu kontrol edin. |
| Yanıt için kanıt yetersiz | İlgili bilginin kılavuzda bulunduğunu ve doğru biçimde indekslendiğini doğrulayın. |

## Gizlilik ve kullanım sınırı

Bu kod deposu yalnızca uygulama kaynaklarını içerir. Şirket kılavuzları, üretim verileri, kullanıcı konuşmaları, geri bildirimler, indeksler, model ağırlıkları ve erişim bilgileri depoya konulmamalıdır.

1Pilot bir karar destek aracıdır. Üretilen cevaplar üretim hattındaki resmî prosedürlerin, güvenlik kurallarının veya yetkili personel kararlarının yerine geçmez. Kritik işlemlerde gösterilen kaynağı ve güncel iş kuralını ayrıca doğrulayın.
