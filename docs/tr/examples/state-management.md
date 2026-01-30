# Durum, URI ve Kaynak Yönetimi Demoları

## Genel Bakış

Bu betikler, dynlib'de simülasyon durumlarının (state) anlık görüntüler seviyesinde yönetilmesi, derlenmiş kaynak kodlarının dışa aktarılması, ön ayarların kaydedilip geri yüklenmesi ve URI/şema sisteminin farklı yollarla çözümlenmesi gibi sık kullanılan iş akışlarını gösterir. Burada anlatılanlar, bir çalışmayı checkpoint'e almak, workspace/sonuçları yeniden kullanmak ve farklı kaynaklardan gelen modelleri tutarlı biçimde yüklemek istediğiniz senaryolarda başvuracağınız örneklerdir.

## Örnek betikler

### Anlık görüntüler (Snapshots) ve içe/dışa aktarma

Bu demo, basit bir üstel bozunma simülasyonu kurar, birkaç zaman noktasına kadar çalıştırır ve bir etiketli snapshot oluşturur. Ardından geçici dizin kullanarak hem kayıtlı hem de geçerli durumları `sim.export_snapshot` ile diske aktarır, meta verilerini inceler, `sim.import_snapshot` ile eski durumu geri yükler, sonuç verilerinin sıfırlandığını doğrular ve workspace bilgisini başka bir simülasyona aktarırken yeni oturuma aynı durumu geri yüklemenin mümkün olduğunu gösterir.

```python
--8<-- "examples/snapshot_demo.py"
```

### Derlenmiş kaynakları dışa aktarma

`tests/data/models/decay.toml` dosyasından JIT etkin bir simülasyon oluşturulur, önbellekleme (disk_cache) devre dışı bırakılır ve `sim.model.export_sources()` yardımıyla RHS, event ve stepper kaynakları geçici bir dizine yazılır. Demo, dosya boyutlarını ve içeriklerini raporlayarak hangi bileşenlerin üretildiğini doğrular ve özellikle RHS ile stepper fonksiyonlarının sağlanan kaynak kodlarını gösterir.

```python
--8<-- "examples/export_sources_demo.py"
```

### Ön ayar (Preset) iş akışları

Satır içi Izhikevich nöronu tanımlaması, ortamda yer alan `regular_spiking`, `fast_spiking` ve `bursting` ön ayarlarının listelenmesi ile başlar. Örnek, her bir ön ayarı sırasıyla uygulayıp spike sayılarını `results().event.summary()` üzerinden rapor eder, daha sonra geçici bir `.toml` dosyasına ön ayarları kaydeder, başka bir `Sim` nesnesinde bunları geri yükler ve glob desteğiyle ön ayarları filtreledikten sonra tekrar çalıştırarak davranışların eşleştiğini kanıtlar.

```python
--8<-- "examples/presets_demo.py"
```

### URI ve yol çözümleme

Birinci bölüm satır içi TOML ile `inline:` şemasını, ikinci bölüm mutlak ve göreceli dosya yollarını, üçüncü bölüm ise desteklenen URI şemalarını (`proj://`, `TAG://`, uzantısız isimler, `#mod=` parçaları gibi) ve bunların nereden çözümlendiğini gösterir. Betik ayrıca Linux/macOS/Windows üzerindeki `dynlib` konfigürasyon dosyalarının yerlerini ve önemli ortam değişkenlerini hatırlatarak belirsiz model yollarında hata ayıklamayı kolaylaştırır.

```python
--8<-- "examples/uri_demo.py"
```
