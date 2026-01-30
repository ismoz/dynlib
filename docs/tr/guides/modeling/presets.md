# Presets (Ön Tanımlı Ayarlar)

Preset'ler (ön tanımlı ayarlar), bir modelin "modları" (örn. hızlı vs. yavaş dinamikler, dinlenme vs. aktif) arasında hızlıca geçiş yapabilmeniz için yeniden kullanılabilir durum (state) ve parametre değer setlerini yakalamanızı sağlar. Tanımlandıktan sonra preset'ler, bir Sim'in bellek içi bankasında yaşar ve çalışma zamanında uygulanabilir, listelenebilir, kaydedilebilir, yüklenebilir veya oluşturulabilir.

## DSL'de Preset Tanımlama

Satır içi (inline) preset'ler, `[presets.<isim>]` tabloları kullanılarak model TOML dosyası içinde bildirilir. Her preset şunları sağlayabilir:

- Parametre geçersiz kılmaları (override) için `[presets.<isim>.params]`
- Durum başlangıç değerleri için `[presets.<isim>.states]`

İki bölümden en az biri bulunmalıdır ve her değer bir sayı (tam sayılar ve ondalıklı sayılar kabul edilir) olmalıdır. Bir preset, durumları atlayabilir (sadece parametre), parametreleri atlayabilir (sadece durum) veya her ikisini de sağlayabilir. Bildirilen isimler, modelin `states` ve `params` değerleriyle eşleşmelidir; geçersiz isimler DSL doğrulandığında yakalanır.

```toml
[presets.fast.params]
alpha = 2.5
beta = 0.1

[presets.fast.states]
x = 5.0
y = -1.0

[presets.rest.params]
alpha = 0.2
beta = 0.01
```

Satır içi preset'ler, başlatma sırasında her `Sim` örneğine (instance) otomatik olarak yüklenir. Bir preset adı birden fazla kez görünürse, ilk tanım kazanır ve bir uyarı verilir.

## Preset Bankası ile Çalışma

Her `Sim` örneği, satır içi tanımlardan ve çalışma zamanında eklenen/yüklenenlerden doldurulan bir preset bankası tutar.

- `list_presets(pattern="*")`: Eşleşen tüm isimleri (`*`, `?`, `[]` desteklenir) alfabetik olarak sıralanmış şekilde döndürür.
- `apply_preset(name)`: Yalnızca preset içinde listelenen parametreleri/durumları günceller; zaman, dt, stepper çalışma alanı, adım sayısı ve kaydedilen geçmiş dokunulmadan kalır. Uygulamadan önce Dynlib, her anahtarın var olduğunu doğrular ve sayısal değerleri model veri tipine (dtype) dönüştürür (hassasiyet kaybolabilirse uyarı verilir).

### Anında Yeni Preset Ekleme

Mevcut oturumun anlık görüntüsünü almak veya özel değerleri kaydetmek için `add_preset(name, *, states=None, params=None, overwrite=False)` kullanın:

- Hem `states` hem de `params` `None` ise, preset mevcut oturumun değerlerini yakalar.
- Her argüman bir eşleme (`{"x": 1.0}`) veya 1-B NumPy dizisi (bildirim sırasına göre yorumlanır) olabilir ve kısmi olabilir (örn. durumların sadece bir alt kümesi).
- Metot, `overwrite=True` olmadığı sürece isim zaten mevcutsa veya saklanacak hiçbir şey yoksa `ValueError` verir.

### Preset'leri Diske Kaydetme (Persisting)

Dynlib, `dynlib-presets-v1` formatını izleyen TOML dosyalarını kullanarak preset'leri okuyabilir/yazabilir. Dosya şunları içermelidir:

```toml
[__presets__]
schema = "dynlib-presets-v1"

[presets.example.params]
a = 1.0
b = 2.0

[presets.example.states]
x = 0.0
```

- `load_preset(name_or_pattern, path, *, on_conflict="error")`: Dosyadaki preset'leri bankaya aktarır. Tam bir isim veya bir glob deseni (örn. `"fast_*"`) geçebilirsiniz. Varsayılan olarak, mevcut bir banka girişiyle çakışma bir hata oluşturur, ancak `"keep"`/`"replace"` banka girişini atlamanıza veya üzerine yazmanıza izin verir (uyarılar eylemi vurgular). Yükleyici, şema başlığını doğrular, sayısal tabloları zorunlu kılar ve referans verilen tüm isimlerin aktif modelde var olduğundan emin olur.
- `save_preset(name, path, *, overwrite=False)`: Bankadan bir preset'i diske ekler veya yazar. `[__presets__]` başlığını oluşturur veya günceller, mevcut ilgisiz preset'lere dokunmaz ve dosya içindeki isim çakışmaları için `overwrite` parametresine saygı duyar.

Birlikte bu yardımcılar, parametre/durum setlerinden müfredatlar oluşturmayı, bunları projeler arasında paylaşmayı veya sayısal bir deneyin durumunu daha sonra yeniden kullanmak üzere dışa aktarmayı kolaylaştırır.

## Örnek

```toml
[model]
type = "ode"

[states]
v = -65.0
u = -13.0

[params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0
I = 10.0
v_th = 30.0

[equations]
expr = """
dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
du = a * (b * v - u)
"""

[events.reset]
cond = "v >= v_th"
phase = "post"
action = """
v = c
u = u + d
"""

# PRESETS:
[presets.regular_spiking.params]
a = 0.02
b = 0.2
c = -65.0
d = 8.0

[presets.intrinsic_bursting.params]
a = 0.02
b = 0.2
c = -55.0
d = 4.0

[presets.bursting.params]
a = 0.02
b = 0.2
c = -50
d = 2

[presets.fast_spiking.params]
a = 0.1
b = 0.2
c = -65
d = 2

[presets.low_threshold.params]
a = 0.02
b = 0.25
c = -65
d = 2

[presets.resonator.params]
a = 0.1
b = 0.26
c = -65
d = 2
```

Simülasyon dosyasında mevcut preset'lerden birini seçebilirsiniz. TOML preset'leri, preset bankasına otomatik olarak eklenir.

Preset'lerin çalışma zamanı kullanımı için simülasyona bakınız.