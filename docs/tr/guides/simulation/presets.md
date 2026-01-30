# Runtime Preset'leri

Bu rehber, bir simülasyon çalışırken preset (hazır ayar) bankasıyla nasıl çalışılacağına odaklanır. Preset tanımlarının DSL tarafı `docs/guides/modeling/presets.md` içinde yer alır, ancak bir `FullModel` derlendiğinde, `Sim` örneği bu preset'lerin kendi bellek içi önbelleğini, artı sonradan eklediğiniz veya içe aktardığınız her şeyi tutar.

## `Sim` içindeki Preset bankası

Her `Sim`, model spesifikasyonundaki satır içi `[presets.<isim>]` tabloları ile başlatma sırasında doldurulan bir preset bankası tutar. Şu anda nelerin mevcut olduğunu incelemek için `list_presets(pattern="*")` kullanın; bu, alfabetik olarak sıralanmış isimleri döndürür ve `glob` tarzı filtreleri (`*`, `?`, `[]`) destekler.

```python
sim = Sim(model)
print(sim.list_presets())  # ['bursting', 'regular_spiking', ...]
```

Banka; çalıştırmalar (runs), snapshot'lar ve resume segmentleri arasında paylaşılır, böylece modeli yeniden oluşturmadan preset'ler arasında geçiş yapabilirsiniz.

## Çalıştırma öncesi bir preset uygulama

Bir preset'in durumlarını (states) ve/veya parametrelerini mevcut oturuma göndermek için `sim.apply_preset(isim)` çağrısı yapın. Yalnızca preset içinde listelenen anahtarlar güncellenir—diğer her şey (zaman, `dt`, stepper çalışma alanı, kaydedilen geçmiş) dokunulmadan bırakılır—bu nedenle bu, `run()` öncesinde veya segmentler arasında oturumu yeniden yapılandırmanın güvenli ve artımlı bir yoludur.

```python
sim.apply_preset("bursting")
sim.run(T=2.0, record=True)
```

Preset'in parçası olmayan bir parametreyi değiştirmeniz gerekirse, preset'i uyguladıktan sonra `sim.assign(...)` kullanın veya ek anahtarı içeren yeni bir preset oluşturun. Bir preset uygulamak `reset()`/`import_snapshot()` sonrasında da çalışır, böylece kaydedilmiş bir durumdan yeni bir değer kombinasyonuyla dallanabilirsiniz.

## Anında yeni preset'ler yakalama

`sim.add_preset(isim, *, states=None, params=None, overwrite=False)`, preset bankasına yeni bir giriş kaydeder.

- Hem `states` hem de `params` atlandığında, yöntem mevcut oturum değerlerinin anlık görüntüsünü (snapshot) alır. Aksi takdirde, saklamak istediğiniz değişkenler için eşlemeler (mappings) veya 1-D NumPy dizileri sağlayın.
- Mevcut bir preset'i değiştirmek için `overwrite=True` geçin, aksi takdirde çakışmayı önlemek için bir `ValueError` yükseltilir.

```python
sim.assign(I=15.0)
sim.run(T=1.0)
sim.add_preset("after_stim", overwrite=True)  # en son durumları ve parametreleri yakalar
```

Ayrıca anahtar kelime argümanlarından birini geçirerek kısmi preset'ler (örneğin, durumların sadece bir alt kümesini saklayan) oluşturabilirsiniz.

## Preset'leri içe ve dışa aktarma

`dynlib-presets-v1` şemasını (`[__presets__].schema = "dynlib-presets-v1"`) izleyen bir TOML dosyasından preset okumak için `sim.load_preset(isim_veya_desen, yol, *, on_conflict="error")` kullanın. Tek bir preset ismini eşleştirebilir veya `"fast_*"` gibi bir glob deseni geçebilirsiniz. Yükleyici dosyayı doğrular, sayısal tabloları zorunlu kılar ve referans verilen her durumun/parametrenin aktif modelde mevcut olduğundan emin olur.

- `on_conflict="error"` (varsayılan), banka zaten preset'i içeriyorsa hata verir.
- `"keep"`, dosya preset'ini atlar ve bankayı dokunmadan bırakır (uyarı verilir).
- `"replace"`, banka girişini dosya sürümüyle üzerine yazar (uyarı verilir).

```python
sim.load_preset("fast_*", "presets.toml", on_conflict="replace")
sim.apply_preset("fast_spiking")
sim.run(T=5.0)
```

Tersine, `sim.save_preset(isim, yol, *, overwrite=False)` bir banka girişini diske geri yazar. Yardımcı, bir `[__presets__]` başlığının var olduğundan emin olur, dosyadaki ilgisiz preset'leri dokunmadan bırakır ve ya yeni bir `[presets.<isim>]` tablosu ekler ya da zaten varsa (eğer `overwrite=True` ise) günceller.

```python
sim.save_preset("after_stim", "presets.toml", overwrite=True)
```

Preset'leri kalıcı hale getirmek; deney müfredatları oluşturmanıza, yapılandırmaları iş arkadaşlarınızla paylaşmanıza veya bir sonuç üreten tam durumları/parametre setlerini sürüm kontrolüne almanıza olanak tanır.

## Runtime preset'leri ile çalışma ipuçları

- Tek bir iş akışı içinde preset'ler arasında geçiş yaparken, oturumu yeniden yapılandırmak için genellikle `apply_preset()` yeterlidir; yalnızca zamanı geri sarmak veya kaydediciyi temizlemek istiyorsanız `reset()` veya snapshot'lara ihtiyaç duyarsınız.
- Tekrar oynatılabilir bir durumu yakalamak ve diğer projeler için dışa aktarmak amacıyla `add_preset()` ve `save_preset()` komutlarını birleştirin.
- Preset'lerin yalnızca listeledikleri değişkenlere dokunduğunu unutmayın—her parametreyi geçersiz kılmanız veya sabit bir zaman hikayesi tutmanız gerekiyorsa, çalıştırmadan önce bunları snapshot'lar veya `assign()` çağrıları ile eşleştirin.