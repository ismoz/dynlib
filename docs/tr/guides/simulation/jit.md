# Simülasyonlarda JIT Derlemesi

dynlib şu anda tek just-in-time (JIT) arka ucu (backend) olarak **Numba**'ya dayanmaktadır. `build()` ve `setup()` varsayılan olarak `jit=False` ve `disk_cache=False` ayarlarına sahip olduğundan, JIT'i açmak bilinçli bir tercihtir: derlenmiş çekirdeklere (kernels) ihtiyacınız olduğunda `jit=True` parametresini geçin. Eğer Numba yüklü değilse, bu bayrak `JITUnavailableError` hatası verir, bu yüzden önce numba'yı yükleyin. Runtime tasarımı, derlenmiş çekirdeğin GIL-free (Global Interpreter Lock'tan bağımsız) olduğunu varsayar, bu nedenle diğer JIT motorları (PyPy, Cython, LLVM sarmalayıcıları vb.) desteklenmez.

## JIT'i Açma

- Düz bir `build(model)` veya `setup(model)` çağrısı, simülasyonu tamamen Python içinde çalıştırır.
- dynlib'i RHS, events, auxiliary updater, stepper, runner ve guard'ları Numba ile derlemeye zorlamak için `jit=True` parametresini kullanın.
- `jit=False` hızlı deneyler veya kısa partiler (batches) için güvenliyken, `jit=True`, ön derleme maliyetinin amorti edildiği uzun süreli simülasyonlar için avantajlıdır.
- Derlenmiş çıktıları işlemler (processes) arasında kalıcı hale getirmek için `jit=True` ile birlikte `disk_cache=True` kullanın; `disk_cache=False` bırakmak her şeyi bellekte tutar ve önbellek kök dizinine (cache root) yazılmasını engeller.

```python
from dynlib import setup

sim = setup("builtin://ode/vanderpol", stepper="rk4", jit=True, disk_cache=True)
sim.run(T=20.0)
```

`FullModel`'i incelemeniz gerektiğinde doğrudan `build()` fonksiyonunu da çağırabilirsiniz. Derlenmiş nitelikleri (`model.rhs`, `model.stepper`, `model.runner` vb.), JIT işleminden sonra olağan Numba `signatures` özelliklerini açığa çıkarır.

## Neler derlenir

JIT yolu, sayısal olarak yoğun döngü (hot loop) içinde çalışan parçaları kapsar:

- **Triplet fonksiyonları**: `rhs`, `events_pre`, `events_post` ve `update_aux`, stepper/runner tarafından çağrıldıklarında performanslı kalmaları için birlikte derlenir.
- **Stepper**: `stepper.emit()`, entegrasyon çekirdeğini üretir; bu da stepper JIT uyumlu olarak işaretlendiğinde JIT ile derlenir.
- **Runner**: Runner şablonu (ordinary, fast-path veya analysis varyantı), tüm wrapper → runner yığınının Python'a geri dönmeden çalışması için aynı `jit` bayrağıyla derlenir.
- **Guard'lar**: JIT etkinleştirildiğinde, durumları/parametreleri doğrulayan guard yardımcıları bir kez derlenir ve `nopython` sözleşmesini sağlam tutmak için yeniden kullanılır.

## JIT çıktılarını önbelleğe alma (Caching)

Önbelleğe alma, aynı modeli yeniden çalıştırdığınızda veya başka bir işlemde (process) yeniden oluşturduğunuzda gereksiz derlemeyi ortadan kaldırır.

- `disk_cache=True`: Derlenmiş triplet'leri, stepper'ları ve runner'ları şu yollarda kalıcı hale getirir:
    - `~/.cache/dynlib/jit/...` (Linux),
    - `~/Library/Caches/dynlib/jit/...` (macOS),
    - `%LOCALAPPDATA%/dynlib/Cache/jit/...` (Windows).

- `disk_cache=False`: Her şeyi bellekte tutar; önbellek kök dizinine yazamadığınızda veya her çalıştırma için özellikle temiz bir derleme istediğinizde kullanışlıdır.

- Kaynak görünürlüğü: Üretilen kaynak kodu, `disk_cache` durumuna bakılmaksızın `model` nesnesi üzerinde (`model.rhs_source`, `model.stepper_source`, …) erişilebilir kalır.

### Önbellek anahtarı (cache key) nasıl oluşturulur

- `runner_variants.get_runner` ve triplet/stepper önbellek oluşturucuları, önbellek anahtarlarını deterministik girdilerden türetir: model hash'i, stepper adı, dtype, guard yapı imzası, runner varyantı, analiz imzası, `cache_token`, JIT bayrağı ve şablon sürümü.
- Triplet/stepper önbellekleri, derlenmiş modülleri `cache_root/jit/triplets|steppers/.../<digest>` altında saklar, böylece aynı hash'e sahip iki çalıştırma derlenmiş çıktıyı hemen yeniden kullanır.
- Runner'lar `analysis` kancalarını (hooks) enjekte eder ve bu nedenle yalnızca analiz dışı varyantları önbelleğe alır (analiz uyumlu runner'lar için `njit(cache=True)` bayrakları atlanır çünkü kancalar çalışma zamanında çözümlenir). Varyantlar yine de işlem içinde (in-process) ve disk üzerinde `runner_cache` aracılığıyla önbelleğe alınır ve `cache_token` (yapılandırılmış önbellek bağlamına dayalı olarak), çalışma alanı düzeni veya dtype değiştiğinde önbelleklerin geçersiz kılınmasını sağlar.

### Önbellek kök dizinini yapılandırma

Önbellek kök dizinini yapılandırma dosyaları veya ortam değişkenleri aracılığıyla geçersiz kılabilirsiniz:

1. `DYNLIB_CONFIG`, `cache_root = "/ozel/kok"` veya `root = "/ozel/kok"` içeren bir `[cache]` tablosu bulundurabilen bir TOML dosyasına işaret eder.
2. `load_config()` ayrıca `DYN_MODEL_PATH`'i de dikkate alır, böylece `cache_root` geçersiz kılmalarını özel TAG kökleriyle birleştirebilirsiniz.
3. Yapılandırılan kök yazılabilir değilse, dynlib `/tmp/dynlib-cache` yoluna geri döner (ve bir kez uyarır). Bu da başarısız olursa, runtime uyarır ve JIT önbelleğini tamamen bellekte tutar (dosya yazılmaz).

### Önbellek dayanıklılığı

- Disk üzerindeki önbellek, bozulma tespit ettiğinde kendini yeniden oluşturur: bozuk modülü siler, oluşturulan kaynağı yeniden işler (re-render) ve derlemeyi yeniden dener.
- `CacheLock` korumaları, birden fazla işlem aynı özeti (digest) aynı anda doldurmaya çalıştığında yarış durumlarını (races) önler.
- Önbellekler ayrıca `cache_token`'a dokunarak manuel bir geçersiz kılma yolunu da destekler (runner oluşturucu mevcut yapı imzasını ve dtype'ı alır), böylece ABI'yi etkileyen değişiklikler otomatik olarak yeni girişler oluşturur.
- Runner veya stepper kaynak kodundaki değişiklikler izlenmez veya hash'lenmez. Kaynak kodundaki herhangi bir değişiklik önbelleği bozabilir veya önbelleğe alınmış çıktılar nedeniyle eski davranışlarla karşılaşabilirsiniz. Bu gibi durumlarda önbelleği silin. Önbelleği silmek için CLI kullanılabilir (örneğin: `dynlib cache clear --all`).

## Derleme durumunu kontrol etme

Bir çağrılabilirin (callable) gerçekten JIT ile derlenip derlenmediğini bilmek istiyorsanız:

```python
from dynlib import build

def is_jitted(fn):
    return hasattr(fn, "signatures") and bool(fn.signatures)

model = build("model.toml", stepper="euler", jit=True)
print("RHS jitted", is_jitted(model.rhs))
print("Runner jitted", is_jitted(model.runner))
```

`model.export_sources(...)` fonksiyonu `disk_cache=True` olduğunda bile çalışır; dışa aktarılan dizin, dynlib'in arka planda ne derlediğini incelemeyi kolaylaştıran her derlenmiş bileşeni (`rhs.py`, `stepper.py`, `runner.py` vb.) içerir.

## En iyi uygulamalar

- `jit=True` seçeneğini, ön derleme maliyetini amorti edecek kadar uzun çalışan simülasyonlar için saklayın.
- Aynı modelleri tekrar tekrar oluşturduğunuz geliştirme döngüleri için `disk_cache=True` kullanın; temiz bir başlangıca ihtiyaç duyduğunuzda (örneğin, taze bir derleme sağlayan CI veya testlerde) bunu kapatın.
- Bir çalıştırma Numba hatasıyla başarısız olursa, traceback'i inceleyin, desteklenmeyen Python yapılarını düzeltin ve yeniden çalıştırın - dynlib, `jit=True` olduğunda sessizce Python'a geri dönmez.
- Yeni observer kancaları eklediğinizde veya dtype/stepper değiştirdiğinizde, önbellek anahtarı otomatik olarak değişir, bu nedenle önbellek dizinlerini manuel olarak silmenize gerek yoktur.

Bu ayarlarla, kalıcı önbellekler aracılığıyla izlenebilirliği ve tekrarlanabilirliği korurken neredeyse yerel (native) performans elde edebilirsiniz.