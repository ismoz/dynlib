# Çalışma zamanı gözlemcileri (Runtime observers)

Dynlib, her `Sim.run()` çağrısıyla birlikte çalışan küçük bir analiz alt sistemi içerir. Observer (gözlemci) modülleri **adım öncesi/sonrası kancaları (hooks)** enjekte eder, çalışma alanı tamponlarını (workspace buffers) taşır ve isteğe bağlı olarak izleri (traces) paylaşılan bir `Results` nesnesine kaydeder; böylece kaydedilen yörüngeleri sonradan işlemeye gerek kalmadan tanılamaları (Lyapunov üsleri, spektrumlar, yakınsama izleri vb.) hesaplayabilirsiniz.

## Observer'ları `Sim.run` ile kaydetme

- `Sim.run(..., observers=…)` şunları kabul eder:
  - Bir `ObserverModule` örneği.
  - Bir **Observer factory** (gözlemci fabrikası) (`(model, sim, record_interval)` imzasına ve `__observer_factory__` bayrağına sahip çağrılabilir nesne). `lyapunov_mle_observer` gibi Dynlib yapımı factory'ler bu bayrağı ayarlar, böylece `Sim` derlenmiş modeli otomatik olarak enjekte edebilir.
  - Bir `ObserverModule` dizisi. `Sim`, birden fazla modülü tek bir geçişte çalışacak şekilde bir `CombinedObserver` (Birleşik Gözlemci) içinde sarmalar.
- `record_interval` (kayıt aralığı) factory'lere iletilir; izleri adım boyutundan daha kaba bir tempoda örneklemek istiyorsanız bunu `Sim.run` çağrısında sağlayın.
- Tüm observer'lar, çalışma başlamadan önce mevcut stepper'a karşı doğrulanır (`ObserverModule.validate_stepper`); uyumsuz kombinasyonlar (örneğin, uyarlamalı bir stepper üzerinde sabit adımlı bir observer) hemen hata verir.
- `CombinedObserver`, benzersiz anahtarları zorunlu kılar, durumu (state) değiştiren observer'ları reddeder ve adım başına en fazla bir varyasyonel entegratöre izin verir; bu nedenle ya tek bir modül ya da dikkatlice oluşturulmuş bir kombinasyon seçin.
- Çalıştırıcı (runner) geçiş başına yalnızca bir varyasyonel stepper yayabildiğinden ve observer'lar aynı iz tamponlarını/sayaçlarını paylaştığından, birden fazla varyasyonel observer'ı karıştıramaz veya tek bir çalışmada birden fazla observer'ın model durumunu değiştirmesine izin veremezsiniz. Her observer ayrıca izleme planı (trace plan) üzerinde anlaşmalıdır (aynı `TracePlan` veya `record_interval`), böylece kaydedilen izler hizalı kalır; hızlı yol (fast-path) ile uyumlu olmayan özellikler (olay günlükleri, kabul/ret kancaları, sabit adım zorunluluğu) gerektiren herhangi bir observer, hızlı çalıştırıcıyı talep etmiş olsanız bile çalışmayı sarmalayıcı (wrapper) yoluna geri zorlayabilir.

## Temel observer yapı taşları

- `ObserverModule` şunları taşır:
  - Sabit adım yürütme, Jacobian-vektör ürünleri (`need_jvp`), yoğun Jacobianlar, olay günlükleri veya varyasyonel adımlama kancaları gibi ihtiyaçları beyan eden `requirements` (`ObserverRequirements`).
  - Çalıştırıcının ne kadar çalışma zamanı depolama alanı ayıracağını belirleyen `workspace_size`, `output_size`, `output_names` ve isteğe bağlı `trace` meta verileri (`TraceSpec`).
  - Her entegrasyon adımında çalışan `pre_step`/`post_step` geri çağrılarına sahip `hooks` (`ObserverHooks`); bu kancalar geçerli `t`, `dt`, durumlar, parametreler, çalışma zamanı çalışma alanı, analiz çalışma alanı, çıktı tamponları ve iz tamponlarını alır.
- `TraceSpec`, kaydedilen iz düzenini tanımlar ve genişlik > 0 olduğunda bir `TracePlan` gerektirir. Çoğu observer için bir `FixedTracePlan(record_interval=K)` geçirebilir veya sadece factory'ye sağladığınız `record_interval`'a güvenebilirsiniz.
- Observer'lar `needs_trace`, `trace_stride` ve `trace_capacity(total_steps=...)` gibi yardımcı yöntemleri dışa açar, böylece `Sim` tampon boyutlarını ayarlayabilir ve taşmayı algılayabilir; `build_observer_metadata(...)` bu bilgileri nihai `Results` yükünde toplar.
- Adım başına kancaları Numba ile derlemeyi planlıyorsanız (hızlı yol çalıştırıcıları), `ObserverModule.resolve_hooks(jit=True, dtype=...)` bunları talep üzerine derlerken, `observer_noop_hook()` kanca kurulu olmadığında çalışma zamanlarını tip kararlı (type-stable) tutar.

## Referans Lyapunov observer'ları

Yerleşik `dynlib.runtime.observers` paketi iki observer factory sağlar:

1. `lyapunov_mle_observer(...)`
2. `lyapunov_spectrum_observer(...)`

Her iki factory de sağlanan modelden gerekli `jvp` ve `n_state`'i otomatik olarak algılar (veya bunları açıkça geçirmenizi gerektirir). Stepper'ın birleşik varyasyonel adımına dayanan veya manuel olarak tanjant entegratörünü çağıran varyasyonel kancalar oluştururlar, bu nedenle:

- **Akış ve Harita modu**: `mode="flow"` paydaları zaman birimlerinde tutar, `mode="map"` iterasyon sayılarını toplar ve `mode="auto"`, `model.spec.kind`'dan doğru davranışı çıkarır.
- **Trace örneklemesi**: Yakınsama izlerini yakalamak için `record_interval` (veya `trace_plan=FixedTracePlan(record_interval=K)`) belirtin; bir trace planı olmadan observer yalnızca çıktı kayıtlarını günceller.
- **Varyasyonel adımlama**: Akış modu observer'ları, `caps.variational_stepping` özelliği etkinleştirilmiş bir stepper gerektirir. `prefer_variational_combined=True`, stepper'ın birleşik durum+tanjant entegratörünü yeniden kullanmaya çalışır; aksi takdirde observer `stepper_spec.emit_tangent_step(...)` tarafından oluşturulan yalnızca tanjant geri çağrısına geri döner.
- **Analiz türü**: Modülü aşağı akış meta verileri veya önbellekleme için etiketlemek üzere `analysis_kind=1` veya başka bir tamsayı geçirin; değer `build_observer_metadata` ve çalıştırıcı önbellekleri (runner caches) üzerinden taşınır.

Factory kullanımı deyimseldir:

```python
from dynlib.runtime.observers import lyapunov_mle_observer, lyapunov_spectrum_observer

sim.run(
    N=5000,
    dt=1.0,
    record_interval=1,
    observers=[
        lyapunov_mle_observer(model=sim.model, record_interval=1),
        lyapunov_spectrum_observer(model=sim.model, k=1, record_interval=1),
    ],
)
res = sim.results()
```

`examples/analysis/lyapunov_logistic_map_demo.py` içindeki lojistik harita demosu, daha sonra `res.observers`'ın nasıl okunacağı da dahil olmak üzere tam olarak bu deseni gösterir.

## Observer çıktılarını inceleme

- `ResultsView.observers`, her observer'ın `key`'i (anahtarı) ile anahtarlanmış `ObserverResult` nesnelerinden oluşan bir sözlük döndürür. Her `ObserverResult` şunları sunar:
  - Geriye dönük uyumluluk için eşleme erişimi (`result["out"]`, `result["trace"]`).
  - `output_names` ve `trace_names`'den türetilen otomatik oluşturulmuş nitelik erişimi (`result.log_growth`, `result.steps` vb.).
  - Keşif yardımcıları (`result.output_names`, `result.trace_names`, `list(result)`).
  - Çalıştırıcı bunları her kaydettiğinde iz satırlarını adım/zaman indeksleriyle hizalayan iz yardımcıları (`result.trace`, `result.trace_steps`, `result.trace_time`).
- Örnek:

```python
lyap = res.observers["lyapunov_mle"]
mle = lyap.mle          # nihai yakınsanmış üs
log_growth = lyap.log_growth
n_steps = int(lyap.steps)
trace = lyap["mle"]     # tam yakınsama izi (record_interval aralıklı)
```

- Çalışma zamanı meta verileri `res.observer_metadata` (veya ham `Results.observer_metadata`) aracılığıyla son işlemden sonra da korunur, böylece çalışma alanı boyutlarını, iz adımlarını veya bir izin taşıp taşmadığını inceleyebilirsiniz.

## Hızlı yol (Fast-path) ve pratik notlar

- Observer'lar mevcut olduğunda, çalıştırıcı analize duyarlı varyantlara (`RunnerVariant.ANALYSIS` / `FASTPATH_ANALYSIS`) geçer. `CombinedObserver.supports_fastpath(...)`, bir hızlı yol çalıştırıcısının uyumlu olup olmadığını kontrol eder; örneğin, olay günlükleri, kabul/ret kancaları veya durum mutasyonu gerektiren observer'lar hızlı yol yürütmeyi devre dışı bırakır.
- `trace` verisi olmayan observer'lar yine de çıktı yuvalarına katkıda bulunur, bu nedenle `Sim.results().observers`, `trace_plan` `None` olsa bile `ObserverResult`'ları döndürecektir.
- Özel observer'lar oluşturursanız aynı deseni izleyin: gereksinimleri bildirin, bir `ObserverModule` döndürün ve `workspace_size`/`output_names`/`trace_names` değerlerini ayarlayın, böylece aşağı akış kodu (çalıştırıcı önbellekleri, meta veri oluşturucuları) tamponları otomatik olarak hizalayabilir.

Çalışma zamanı tanılamalarını (Lyapunov üsleri, spektrumlar, büyüme oranları) ekstra son işlem olmadan izlemek için observer altyapısını kullanın ve tanılamaların simülasyon çalışmanızla anahtarlanmış ve zaman hizalı kalması için `ResultsView.observers`'a güvenin.