# Komut satırı rehberi

Dynlib, iki giriş noktası olarak hafif bir komut satırı arayüzü sunar: paketle birlikte yüklenen `dynlib` konsol betiği ve `python -m dynlib.cli`. CLI (Komut Satırı Arayüzü), bir Python betiği yazmadan modelleri doğrulayabilmeniz, stepper kaydını inceleyebilmeniz ve disk üzerindeki JIT önbelleğini yönetebilmeniz için bir avuç çalışma zamanı aracını aynalar. Her fiil/isim ayrıştırıcı katmanı `--help` seçeneğini destekler, bu nedenle mevcut seçenekleri hatırlamanız gerektiğinde `dynlib <komut> --help` veya `python -m dynlib.cli <komut> --help` komutunu çalıştırın.

## Genel bayraklar

- `--version`
  Şu anda yüklü olan dynlib sürümünü yazdırır. CLI, sürümü önce `importlib.metadata` aracılığıyla keşfeder, bu başarısız olduğunda ise (düzenlenebilir kurulumlar, kaynak kod kontrolleri vb.) `pyproject.toml` dosyasını okumaya geçer.

## Model araçları

`dynlib model validate <uri>`

- **Amaç:** Dynlib'in TOML tabanlı DSL'inde tanımlanmış bir modeli ayrıştırır ve doğrular. CLI, işlemi `load_model_from_uri` işlevine devreder, bu nedenle projenin geri kalanında kullanılan `builtin://` URI'lerini, dosya sistemi yollarını ve diğer kayıtlı yükleyicileri destekler.
- **Başarı mesajı:** DSL geçerli olduğunda, komut `Model OK` mesajıyla birlikte model türünü (`ode` veya `map`), veri tipini (dtype), durum (state) sayısını ve `spec.sim.stepper` içinde kayıtlı varsayılan stepper'ı yazdırır.
- **Hata yönetimi:** Sözdizimi ihlalleri, eksik alanlar veya çalışma zamanı doğrulama sorunları, `stderr` üzerinde açıklayıcı bir mesaj ve sıfır olmayan bir çıkış kodu ile `DynlibError` aracılığıyla gösterilir.

Bu komutu, bir simülasyonu çalıştırmadan, bir modeli paylaşmadan veya spesifikasyonu başka bir araç zincirine dahil etmeden önce hızlı bir sağlamlık kontrolü olarak kullanın.

## Stepper kaydı

`dynlib steppers list [--kind <tür>] [--<kapasite>] [--jacobian <politika>]`

- **Amaç:** Kayıtlı her `StepperMeta`/`StepperCaps` çiftini inceleyin. CLI, yan etki olarak `dynlib.steppers` modülünü içe aktardığı için, tüm yerleşik ve kayıtlı üçüncü taraf stepper'lar listede görünür.
- **Görüntülenen sütunlar:** Her satır; stepper adını, `kind` (tür), `scheme` (şema), `order` (derece), `stiff` (sertlik) ipucunu ve her `StepperCaps` alanını gösterir, böylece kaynak kodunu okumadan özellikleri hızlıca karşılaştırabilirsiniz.
- **Tür (Kind) filtresi:** Listeyi sadece ODE çözücüler veya ayrık haritalarla (çalışma zamanı tarafından kullanılan aynı `Kind` numaralandırması) sınırlamak için `--kind ode` veya `--kind map` kullanın.
- **Kapasite filtreleri:** CLI, her `StepperCaps` alanı için dinamik olarak bir bayrak sunar.
  - Boolean (Mantıksal) bayraklar (`--dense_output`, `--jit_capable`, `--requires_scipy`, `--variational_stepping`) birer _gereksinimdir_. Sağlandığında, yalnızca `StepperCaps` ayarı o alanı `True` (Doğru) olarak ayarlayan stepper'lar çıktıda kalır.
  - Değer bayrakları şu anda `--jacobian` seçeneğini içerir (`JacobianPolicy` değişmeziyle eşleşir: `none`, `internal`, `optional`, `required`). O Jacobian davranışını bildiren stepper'ları filtrelemek için tam politika dizesini sağlayın.
- **Kullanım durumları:** Bu komut, hangi stepper'ların yoğun çıktıyı (dense output) desteklediğini doğrulamak (örneğin, animasyon veya değişken adımlı enterpolasyon için temel oluşturup oluşturmadığı), JIT ile derlenebilen alt kümeyi belirlemek veya üçüncü taraf bir stepper'ın beklediğiniz yetenekleri kaydettiğini hızlıca kontrol etmek için kullanışlıdır.

## Önbellek (Cache) yönetimi

Tüm önbellek komutları `resolve_cache_root()` işlevine devreder, böylece [yapılandırma dosyasında](../modeling/config-file.md) açıklanan `[cache]` geçersiz kılmalarınıza, `DYN_MODEL_PATH` etiket haritası uzantılarınıza veya `DYNLIB_CONFIG` ortam değişkeninize saygı duyarlar.

### `dynlib cache path`

- Önbellek kök dizinini yazdırır. Diskteki dosyaları incelemek, dizini bir konteyner içine bağlamak (mount) veya izin sorunlarını gidermek istediğinizde kullanışlıdır.

### `dynlib cache list [--stepper <isim>] [--dtype <belirteç>] [--hash <önek>]`

- **Amaç:** `cache_root/jit/{triplets,steppers,runners}` altındaki her girdiyi numaralandırır.
- **Çıktı formatı:** Her girdi; aileyi (`triplets`, `steppers` veya `runners`), stepper adını, dtype'ı, spec özetini (hash), digest'i, boyutu (okunabilir formatta) ve dosya yolu bilgisini yazdırır. Derleme zamanı bileşenlerini kaydeden girdiler ayrıca `components=...` bilgisini de ekler.
- **Filtreler:**
  - `--stepper` stepper adıyla eşleşir (büyük/küçük harf duyarsız).
  - `--dtype` dtype belirteciyle eşleşir (büyük/küçük harf duyarsız).
  - `--hash`, belirli bir spesifikasyonla ilişkili yapıları çekmek için model spec özetinin (hash) bir önekiyle eşleşir.
- **Sıralama:** Sonuçlar aile, stepper, dtype ve digest'e göre sıralanır, böylece ilişkili yapılar bir arada görünür.
- **Kullanım durumları:** Dtype veya stepper değiştirdikten sonra önbelleğe alınmış çekirdeklerin (kernels) var olup olmadığını doğrulamak veya `disk_cache` bayrağının geride hangi çalıştırıcıları (runners) bıraktığını kontrol etmek için bunu çalıştırın. `--hash` ile filtreleme, bir model özeti (hash) için derlenmiş yapıyı bulmanın en hızlı yoludur ve `--dtype` karışık hassasiyet kullandığınızda yardımcı olur.

### `dynlib cache clear (--all | --stepper <isim> | --dtype <belirteç> | --hash <önek>) [--dry_run]`

- **Amaç:** Artık ihtiyaç duymadığınız önbelleğe alınmış JIT yapılarını silmek veya kod değişikliklerinden sonra önbellek bozulmasını düzeltmek.
- **Güvenlik önlemi:** `--all` veya filtre bayraklarından en az birini belirtmelisiniz. Bir filtre olmadan CLI, bir mesaj ve `2` hata kodu ile çıkar.
- **`--all`:** `shutil.rmtree` aracılığıyla tüm önbellek kök dizinini kaldırır. Temiz bir sayfa istediğinizde (örneğin, dynlib'i güncelledikten veya `cache_root` değiştirdikten sonra) bunu kullanın. Dizin eksikse komut hiçbir şey yapmaz.
- **Seçici silme:** Yalnızca eşleşen önbellek girdilerini silmek için `--stepper`, `--dtype` ve/veya `--hash` seçeneklerini birleştirin. Eşleştirme büyük/küçük harfe duyarlı değildir ve `--hash` önek eşleşmeleri üzerinde çalışır, böylece özetin (hash) yalnızca bir kısmını hatırlasanız bile bir commit'i veya spec sürümünü hedefleyebilirsiniz.
- **`--dry_run`:** Diske dokunmadan silinecek dosya ve dizinleri yazdırır. Yıkıcı bir işlemden önce hedef listesini iki kez kontrol etmek için bunu çalıştırın.
- **Geri bildirim:** Silinen her önbellek bir onay satırı yazdırır ve herhangi bir silme işlemi başarısız olursa komut sıfır olmayan bir çıkış kodu döndürür.

## Örnekler

```bash
dynlib model validate docs/models/lorenz.toml
dynlib steppers list --kind ode --jit_capable --variational_stepping
dynlib cache list --hash 9c8a --dtype float64
dynlib cache clear --stepper rk4 --dry_run
```

## Sorun Giderme

- Önbellek sorunlarını incelerken, hangi dizini kontrol ettiğinizi bilmek için `dynlib cache list` komutunu `dynlib cache path` ile eşleştirin.
- CLI beklediğiniz bir stepper'ı bulamazsa, onu tanımlayan modülün içe aktarıldığından emin olun (çalışma zamanı başlangıçta `dynlib.steppers` modülünü otomatik olarak içe aktarır, ancak üçüncü taraf stepper'lar CLI çalıştırılmadan önce kendilerini kaydettirmelidir).
- `model validate` komutu herhangi bir DSL ayrıştırma sorunu için `DynlibError` yükseltir; hata çıktısını başka bir araçla işlemek istediğinizde bunu bir kabuk (shell) boru hattında çalıştırın (`dynlib model validate ... || true`).