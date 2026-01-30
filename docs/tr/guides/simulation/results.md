# Simülasyon Sonuçları

Bu rehber, dynlib'in ne kaydettiğini, sonuçları nasıl dilimleyeceğinizi/filtreleyeceğinizi/dışa aktaracağınızı ve büyük simülasyonları nasıl yönetilebilir tutacağınızı anlamanız için `Sim.results()` / `Sim.raw_results()` konularına derinlemesine iner.

## 1. `ResultsView` ile isimlendirilmiş erişim

`Sim.results()`, model spesifikasyonundan türetilen isimlerle simülasyon sonuçlarına ergonomik erişim sağlayan bir `ResultsView` döndürür:

- `res.t`, `res.step`, `res.flags`; zaman eksenini, adım indekslerini ve durum bayraklarını NumPy görünümleri (views) olarak verir.
- `res["x"]`, `res["aux.energy"]` veya `res[["x","y"]]`; durumlar ve aux değişkenleri için kaydedilen serileri döndürür ve çoklu değişken isteklerinde gerektiğinde bunları kompakt kopyalar halinde yığar.
- `res.analyze(...)` hızlı istatistikler (maks/min/kesişmeler) için bir `TrajectoryAnalyzer` / `MultiVarAnalyzer` oluşturur ve `res.observers`, çalışma zamanı observer çıktılarını ergonomik `ObserverResult` sarmalayıcısı aracılığıyla yüzeye çıkarır.
- `res.segment`, ana API'yi yansıtırken tek bir çalıştırmaya (otomatik `run#N` isimleri veya manuel etiketler) odaklanmanızı sağlar. Her `SegmentView`, kopyalama yapmadan o parçaya ait `t`, `step`, `flags` ve hatta `events()` verilerini dilimler.

## 2. Şununla ham erişim: `Results`

Alttaki tamponlara (buffers) doğrudan erişmesi gereken ileri düzey kullanıcılar için `Sim.raw_results()`, kopyalama yapmadan runner tamponlarını yansıtan bir `Results` veri sınıfı (dataclass) sunar. Temel alanlar şunlardır: destekleyici diziler (zaman `T`, durumlar `Y`, isteğe bağlı aux `AUX`, `STEP`, `FLAGS`), olay günlüğü (`EVT_CODE`, `EVT_INDEX`, `EVT_LOG_DATA`), dolu sayılar `n`/`m`, çıkış `status` ve son durum/parametreler/çalışma alanlarının snapshot'ları. Her erişimci (accessor), dolu bölgeyle sınırlı bir görünüm sağlar, böylece her zaman bitişik kayıtları görürsünüz ve `Results.to_pandas()`, sütunları sonraki NumPy/Pandas iş akışları için düzenli (tidy) bir `DataFrame` olarak somutlaştırabilir.

Tamponun tamamına ihtiyacınız olduğunda `Sim.raw_results()` kullanın. Çoğu kullanıcı için `Sim.results()`, bu düşük seviyeli nesneyi isimler, yardımcılar ve segmentlerle sarar.

## 3. Dilimleme, filtreleme ve dışa aktarma

- `res["var"]`'ı birincil dilimleme kancanız olarak kullanın; birden fazla seriyi yığmak ve doğal sıralamayı korumak için `res[["x","y"]]` kullanın.
- Segment başına yörünge (trajectory) dilimleri için `res.segment[0]` veya `res.segment["run#1"]` indekslemesi yapın. Her segment kendi kayıt penceresine saygı duyar ve sarılmış kısım için `events()` işlevini açar.
- Tablo şeklinde dışa aktarmaları tercih ettiğinizde, `Results.to_pandas()` size `t`, `step`, `flag`, her durum sütunu ve önekli aux sütunlarını verir, böylece çerçeveyi doğrudan Pandas/NumPy'ye verebilirsiniz.

### Olay (Event) günlüğü sonuçlarına erişim

Olaylar yörüngeyle birlikte saklanır ve her olay satırı bir kod, sahip kayıt indeksi ve günlüğe kaydedilmiş veri bloğu taşır. `ResultsView`, DSL tanımlı isimleri/etiketleri çözer, böylece `res.event("threshold")` hangi kodun, alanların ve etiketlerin kullanılacağını bilir. NumPy'nin keyfi satır görünümleri üzerindeki sınırlamaları nedeniyle, filtreleme (zaman aralıkları, baş/kuyruk, sıralama) kompakt diziler ayırır (allocate), ancak API ayırmaları izole tutar, böylece sonuçların geri kalanı salt okunur (view-only) kalır.

- Bir `EventView` almak için `res.event("spike")` çağırın, ardından `.time(t0, t1)`, `.head(k)`, `.tail(k)` veya `.sort(by="t")` zincirleyin ve `ev["id"]` veya `ev[["t","id"]]` ile bireysel alanları alın. `ev.table()` kaydedilen tüm sütunları sırayla somutlaştırır.
- Birden fazla olay türü üzerinde gruplandırılmış bir görünüm için `res.event(tag="group")` kullanın; `group.select(...)` alanları birleştirmenize veya kesiştirmenize izin verirken, `group.table(...)` birleştirilmiş satırları sıralayabilir.
- `res.event.summary()` olay türü başına hızlı sayımlar verir ve `res.event.names()/fields()/tags()` neyin kaydedildiğini keşfetmenize yardımcı olur.

## 4. Büyük veri setleri ve harici araçlarla çalışma

- `Sim.config()` ve `run()` kancaları aracılığıyla günlüğe kaydetmeyi (logging) kontrol edin: `record` özelliğini açıp kapatın, her `record_interval` adımda bir atlayın veya sadece ihtiyacınız olanı yakalamak için `record_vars`/`[]` geçin.
- Tamponları önceden ayırmak için `cap_rec`/`cap_evt` değerlerini artırın, alt örnekleme (downsampling) için `record_interval` değerini düşürün veya zaman/adım/bayrakları kaydederken durum/aux kaydını tamamen devre dışı bırakın.
- Tamponları aşırı yüklemeden aşamalı deneyleri yönetmek için `transient`, `resume` ve snapshot'ları (`Sim.reset()`, `Sim.create_snapshot()`) kullanın.
- Halihazırda açığa çıkarılmış diziler (`res.t`, `res["x"]`, `.events()`, `.table()`) aracılığıyla veya tutarlı sütun isimlerine sahip bir `DataFrame`'e ihtiyaç duyduğunuzda `Results.to_pandas()` ile NumPy/Pandas'a dışa aktarın.
- `res.segment[...]`, `res.event(...)` ve `res.observers`, ilgilendiğiniz dilimleri tamponun geri kalanından bağımsız tutar, böylece bunları gereğinden fazla kopyalamadan sonraki analizörlere aktarabilirsiniz.

## Özet

- Ergonomik isimler ve yardımcılar istediğinizde `Sim.results()` kullanın, ham tampona ihtiyacınız olduğunda `Sim.raw_results()`'a geri dönün.
- Segmentleri, olayları ve observer'ları dilimler çıkarmak için `ResultsView` aracılığıyla keşfedin ve yörüngeleri NumPy/Pandas'a vermek için `Results.to_pandas()` veya `res[["x","y"]]` ile yığmayı kullanın.
- Bellek kullanımını kararlı tutmak için uzun çalıştırmalardan önce `record`, `record_interval`, `record_vars` ve tampon sınırlarını (buffer caps) ayarlayın, ardından ilgilendiğiniz olayları/segmentleri tekrar oynatın veya dışa aktarın.