# Snapshots & Resume

`Sim`, zamanı, state’leri, parametreleri, stepper workspace’ini ve runtime metadatasını kapsayan canlı bir `SessionState` tutar. Snapshot’lar bu state’i belirli bir anda yakalar; böylece geri sarabilir, dallanabilir (branch) veya bir simulation’ı serialize edebilirsin. `run(resume=True)` ise modeli baştan kurmadan kaydedilmiş segment’leri uzatmanı sağlar.

## Snapshot temelleri

- **Başlangıç snapshot’ı**: `"initial"` snapshot’ı ilk `run()` öncesinde otomatik oluşturulur; böylece her zaman geri dönebileceğin bilinen bir başlangıç noktası vardır.
- **`create_snapshot(name, description="…")`**: Mevcut `SessionState`’i kopyalar, mevcut `time_shift`/`dt` değerlerini kaydeder, snapshot’a `name`/`description` yazar ve tam workspace + stepper config’i saklar. Aynı isim ikinci kez verilirse hata verir; açıklayıcı isimler seç.
- **`list_snapshots()`** `name`, simulation zamanı `t`, step sayısı, oluşturma zamanı ve verdiysen açıklamayı döndürür.
- **`compat_check(snapshot)`** `SessionPins` (spec hash, stepper, workspace imzası, dtype, dynlib sürümü) karşılaştırması yapar ve snapshot’ın uyumlu bir build’den geldiğini garanti eder. `reset()` aynı kontrolü kullanır; model, stepper veya dtype değiştiyse hızlıca başarısız olur.

Snapshot oluşturmak hafiftir ve saklamak ucuzdur; bu yüzden bir dönüm noktasına geldiğinde bir tane al (ör. uyarı/stimulus uyguladıktan sonra, bir parametre sweep’ini bitirince vb.).

## Resetleme ve yeniden başlatma

- **`reset(name="initial")`** oturumu isimli snapshot’a geri alır ve kaydedilmiş geçmişi, segment’leri, bekleyen run tag’lerini ve resume state’ini temizler. Snapshot’tan `_time_shift` ve `_nominal_dt` değerlerini geri yükler; böylece sonraki `run()` çağrıları tam o andan başlar.
- `reset` sonrası recorder temizlenir; bu aynı zamanda saklanan `record_vars` seçimini de sıfırlar. Sonraki run’dan önce farklı bir değişken alt kümesi seçebilirsin.
- `session_state_summary()` `can_resume`/`reason` raporlar; böylece `resume=True` mümkün mü diye, run mantığını tetiklemeden bakabilirsin.

## Çalıştırma ve devam ettirme (resume)

`Sim.run(resume=True)`, `t0`’dan yeniden başlamak yerine en güncel `SessionState`’ten devam eder. Ana davranışlar:

1. **Session sürekliliği**: Önceki run’daki workspace, stepper konfigürasyonu ve `time_shift` korunur. Böylece sonraki segment hem deterministic hem de adaptive stepper’larda kesintisiz bir uzantı gibi davranır.
2. **Kayıt kısıtları**: `ic`, `params`, `t0` veya `dt` override edemezsin; `resume` her zaman oturumun kaldığı yerden başlar. `transient > 0` (warm-up) yasaktır; çünkü resume segment’i anında devam etmelidir. `record_vars` da yeniden verilemez; `reset` sonrası ilk recorded run değişken listesini sabitler ve sonraki tüm `resume` run’ları bu listeyi otomatik yeniden kullanır.
3. **Segment takibi**: Her recorded run bir `Segment` girdisi ekler: `t_start`, `t_end`, `step_start`, `step_end` ve parçanın resume ile üretilip üretilmediği. `run()` çağrısına `tag="label"` verirsen okunur bir isim olur; aksi halde `run#0`, `run#1` gibi isimler üretilir. `ResultsView` için okunur etiketler gerektiğinde `name_segment()` veya `name_last_segment()` ile yeniden adlandır.
4. **Sonuçların birleştirilmesi**: Resume aynı `_ResultAccumulator`’ı kullanır; bu yüzden `raw_results()`/`results()` tüm segment’leri kapsayan tek parça bir time-series görür. `run(resume=True)`, istenen ufuk mevcut zamanın ilerisinde değilse hata verir; böylece üst üste binme (overlap) olmaz.
5. **Uyumluluk kontrolü**: Resume’den önce `can_resume()`, mevcut pin’leri `SessionState` içinde yakalanmış pin’lerle karşılaştırır. `(False, reason)` dönerse `reset()` ile yeniden başlat veya uyumlu bir `FullModel` ile yeniden kur.
6. **Resume içinde parametre override yok**: Resume segment’inde yeni `ic`, `params`, `t0` veya `dt` veremezsin. Run; önceki segmentin parametrelerini, stepper workspace’ini ve zamanlamasını korur. Değer değiştirmek için reset/snapshot kullanman veya `resume=True` olmadan ayrı bir `run()` yapman gerekir.

Tipik kullanım:

```python
sim.run(T=2.0, record=True, tag="phase-1")
sim.create_snapshot("phase-1", "after the first stimulus")

# Yeniden kurmadan devam et; ikinci run eklenir
sim.run(T=5.0, resume=True, tag="phase-2")

# Kaydedilmiş snapshot’a resetleyerek farklı bir dal başlat
sim.reset("phase-1")
sim.run(T=3.0, record=True, tag="phase-1-replay")
```

Segment’ler arasında parametre değiştirmek gerekiyorsa, resumed run’dan önce yap: daha erken bir snapshot’a resetle, yeni parametre/state değerlerini `assign()` ile ver (ya da zaten o değerleri içeren bir snapshot içe aktar), sonra `resume` olmadan run et veya yeni değerler hazır olduktan sonra `run(resume=True)` çağır. Resume hiçbir zaman `ic`, `params`, `t0` veya `dt` override kabul etmez; yeni konfigürasyon mutlaka resumed segment başlamadan önce snapshot/assignment ile ayarlanmalıdır.

### Segment’ler arasında parametre değiştirme örnekleri

**Örnek 1: Parametre değiştirerek dallanma (resume yok)**

```python
# İlk segmenti çalıştır
sim.run(T=2.0, record=True, tag="baseline")

# İlk segment sonunda snapshot oluştur
sim.create_snapshot("after-baseline", "End of baseline run")

# Snapshot’a dön ve bir parametre değiştir
sim.reset("after-baseline")
sim.assign(I=15.0)  # Input current parametresini değiştir

# Değişmiş parametreyle yeni segment çalıştır (kayıt yeniden başlar)
sim.run(T=3.0, record=True, tag="modified-current")
```

**Örnek 2: Parametre değiştirerek devam etme (resume kullanarak)**

```python
# İlk segment
sim.run(T=2.0, record=True, tag="phase-1")

# Snapshot oluştur
sim.create_snapshot("phase-1-end", "End of phase 1")

# Resetle ve parametreleri değiştir
sim.reset("phase-1-end")
sim.assign(a=0.02, b=0.25)  # Izhikevich parametrelerini değiştir

# Reset noktasından yeni parametrelerle devam et
sim.run(T=5.0, resume=True, tag="phase-2-modified")
```

**Örnek 3: `assign()` + `clear_history=True` ile yeni kayıt başlatma**

`assign()` metodunun `clear_history` adında opsiyonel bir parametresi vardır. `clear_history=True` olunca, yeni atanan değerlerle mevcut session state (zaman, workspace vb.) korunur ama önceki sonuçlar ve segment’ler temizlenir. Bu, snapshot’a dönmeden yeni bir segment başlatmana imkân verir:

```python
# İlk segment
sim.run(T=2.0, record=True, tag="initial")

# Yeni parametreleri ata ve geçmişi temizle
sim.assign(I=20.0, clear_history=True)

# Bu run yeni bir segment üretir (çünkü geçmiş temizlendi)
sim.run(T=3.0, record=True, tag="new-segment")
```

Not: `clear_history=True` simulation zamanını veya workspace state’ini değiştirmez — sadece kaydedilmiş sonuçları temizler ve sonraki `run()` çağrısının mevcut session state’ten yeni bir kayıt başlatmasını sağlar.

## Kalıcılık ve taşınabilirlik

- **`export_snapshot(path, source="current" | "snapshot", name=...)`** bir `.npz` dosyasına şunları yazar:
  - `meta.json` (şema sürümü, pin’ler, isimler, zaman/adım sayaçları, `time_shift`, `nominal_dt`, stepper config isim/değerleri)
  - `y` ve `params` vektörleri
  - Workspace klasörleri (`workspace/runtime/<name>`, `workspace/stepper/<name>`) ve varsa `stepper_config`
  - Yazma işlemi geçici dosya üzerinden atomik yapılır; böylece yarım yazmalar mevcut snapshot’ları bozmaz.
- **`inspect_snapshot(path)`** session’ı değiştirmeden `meta.json` okur; içe aktarmadan önce uyumluluk kontrolü yapmak için idealdir.
- **`import_snapshot(path)`** session state’i dosyadaki snapshot ile değiştirir, results/segment’leri temizler, `_time_shift`/`_nominal_dt` değerlerini resetler ve pin’leri aktif modelle uyuşmayan dosyaları reddeder.

Uzun bir hesaplamayı checkpoint almak, bir resume noktasını ekip arkadaşınla paylaşmak veya CI run’ları arasında workflow state’i saklamak için export/import kullan.

## Segment adlandırma ve resume metadatası

- Her `Segment`, bir `cfg_hash` (stepper config özeti) ve `resume` bayrağı taşır; böylece aşağı akış araçları bir parçanın taze bir run ile mi yoksa `resume` ile mi üretildiğini anlayabilir.
- `run()` çağrısında `tag` kullan veya sonradan `name_segment()` ile değiştir; `segments` listesini okunur tutar. `ResultsView` bu isimleri gösterir; böylece istediğin kısmı hızlıca seçebilirsin.
- Resetlediğinde segment listesi boşalır, ama snapshot’lar kalır. Segment eklemeyi yalnızca `resume=True` ile sürdür — aksi halde `run()` accumulator’ı temizler ve yeni bir kayıt turu başlatır.

Snapshot ve resume kontrolleri deneylerini tekrarlanabilir tutar: dallanma noktalarında snapshot al, varyasyonları denemek için onlara resetle ve uzun trajectory’leri geçmişi kaybetmeden büyütmek için resume kullan.
