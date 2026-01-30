# Tarama (Sweep) analizi araçları

Dynlib, önyükleme mantığını kopyalamadan veya çalıştırmaları manuel olarak yinelemeden sistem davranışının bir parametre ızgarası (grid) boyunca nasıl değiştiğini keşfedebilmeniz için özel bir `dynlib.analysis.sweep` modülü barındırır. Her yardımcı (helper), *mevcut simülasyon oturumunu* (`sim.state_vector(source="session")` ve `sim.param_vector(source="session")`) temel olarak okur, bir parametreyi değiştirir ve içeriği kaydettiklerinizle eşleşen bir `SweepResult` döndürür.

## Temel sonuç yardımcıları

### `SweepResult`
`SweepResult`, beklediğiniz meta verileri (`param_name`, `values`, `kind`, `meta`) normalleştirir ve asıl veriyi eşleme ve öznitelik erişimi yoluyla sunar. Skaler taramalar `outputs` kısmını doldurur, Lyapunov taramaları `traces` ekler ve yörünge (trajectory) taramaları `payload` içine bir `TrajectoryPayload` ekler. Yardımcı, kaydedilen değişken adlarını (`record_vars`) yeniden dışa aktarır, `t`/`t_all` sağlar ve `.runs` (değer başına `SweepRun` nesneleri) veya tüm yörüngeler aynı uzunluğu paylaştığında `.stack()` gibi kolaylık yardımcıları sunar. Eksik bir anahtar istendiğinde hemen hata verir ve yazım hatalarının hızlıca fark edilmesi için mevcut alanları listeler.

### `TrajectoryPayload`, `SweepRun` ve `SweepRunsView`
`TrajectoryPayload`, kaydedilen adların demetini (tuple), her çalıştırmanın `t_runs`/`data` dizilerini ve tarama ızgarası `values` değerlerini tutar. Ayrıca, isimlendirilmiş erişimi (`payload["x"]`, `payload.series([...])`) ve yığınlama (`payload.stack()`) veya varsayılan zaman eksenini alma (`payload.t`) yeteneğini güçlendiren dahili bir `_var_index` oluşturur. Farklı çalıştırmalar farklı uzunluklar kaydettiğinde (uyarlanabilir stepper'lar, `record_interval` vb.), `t_all` ve `data` hizalı kalır, böylece her çalıştırmayı manuel olarak işleyebilirsiniz.

`SweepRunsView`, çalıştırma değerleri üzerinde liste benzeri bir sarmalayıcıdır; üzerinde gezinmek `SweepRun` örnekleri üretir. Her `SweepRun`, `param_value` değerini, çalıştırma başına `t`'yi ve bir değişken arama tablosunu saklar, böylece `run["x"]` veya `run[["x","y"]]` o parametre değeri için kaydedilen izi (trace) döndürür.

## Tarama yardımcıları

### `scalar_sweep`
Parametre değeri başına tek sayılık özetler (denge noktaları, ortalama eğilimler, min/max zarfları) için `scalar_sweep` kullanın. Tek bir `var` kaydedersiniz ve bir indirgeme `mode` (modu) seçersiniz:

- `"final"` (varsayılan): kaydedilen son örnek
- `"mean"`: kaydedilen pencere üzerindeki aritmetik ortalama
- `"max"`/`"min"`: uç değerler

Yardımcı, uygun konfigürasyonlar için hızlı bir toplu çalıştırma (`fastpath_batch_for_sim`) dener. Hızlı yol (fast path) kullanılamazsa uyarır (`_warn_fastpath_fallback`) ve normal `Sim.run()` çağrılarına geri döner. Sonuç, `outputs['y']` içinde indirgenmiş diziyi (`(M,)` şeklinde) tutan ve `meta` içinde entegrasyon ayarlarını, stepper türünü ve indirgeme modunu kaydeden bir `SweepResult(kind="scalar")` olur.

```python
from dynlib.plot import series

res = sweep.scalar_sweep(
    sim,
    param="r",
    values=np.linspace(2.5, 4.0, 4000),
    var="x",
    mode="final",
    N=2000,
    transient=1000,
)
series.plot(x=res.values, y=res.y, xlabel="r", ylabel="x*")
```

### `traj_sweep`
`traj_sweep`, herhangi bir `record_vars` kombinasyonu (ör. `"x"`, `"y"`, `"z"`) için tam yörüngeleri kaydeder. Her çalıştırmanın zaman serisi bir `TrajectoryPayload` içinde yaşar, böylece parametre başına çizim yapmak için `res["x"]`, `res.series(["x","y"])`, `res.stack()` çağırabilir veya `res.runs` üzerinde gezinebilirsiniz. Tarama, hem hızlı toplu çalıştırmayı hem de `values` > 1000 olduğunda `ProcessPoolExecutor`'ı destekler. `parallel_mode` argümanı, bu toplu çalıştırmanın nasıl yürütüleceğini kontrol eder (`"auto"`, `"threads"`, `"process"`, `"none"`); `max_workers` işçi (worker) havuzu boyutunu ayarlar. İşçi sayısı bir olduğunda veya `process` modu verimli olmadığında, yardımcı şeffaf bir şekilde sıralı yürütmeye düşer.

`record_interval`, bellek tasarrufu için kaydı seyreltmenize (decimate) olanak tanır ve tarama bu aralığı `meta` içinde hatırlar. Ayrıca sabit bir `dt`, `t0`, `T` veya ayrık iterasyon sayısı `N` (haritalar/maps için kullanışlıdır) talep edebilirsiniz.

```python
from dynlib.plot import phase

res = sweep.traj_sweep(
    sim,
    param="A",
    values=[0.5, 1.0, 1.5],
    record_vars=["x", "y"],
    dt=0.01,
    T=20.0,
    record_interval=5,
)
for run in res.runs:
    phase.xy(x=run["x"], y=run["y"], label=f"A={run.param_value}")
```

### `lyapunov_mle_sweep`
Bu yardımcı, maksimum Lyapunov üssü (MLE) gözlemcisi ile bir parametre taramasını birleştirir. Hızlı toplu çalıştırma (ve Lyapunov gözlemcilerinin kendileri) için, sabit adımlı bir stepper ve açık bir `dt` ile JIT-derlenmiş bir simülasyon kullanmalısınız, ancak hızlı yol desteği yoksa yardımcı, gözlemciler eklenmiş sıralı `Sim.run()` işlemine nazikçe geri döner. Fonksiyon, `mle`, `log_growth` ve `steps` için `outputs` döndürür ve eğer `record_interval` sağladıysanız, her üssün nasıl yakınsadığını inceleyebilmeniz için `traces['mle']` (yakınsama dizileri listesi) de döndürür. `analysis_kind`, algoritma varyantları arasında seçim yapmanızı sağlar.

Tarama, isteğe bağlı `ProcessPoolExecutor` hızlandırmasıyla (değer listesinin parçaları halinde) hızlı toplu çalıştırmalar dener. Hızlı yol veya hızlı paralel işçi başlatma başarısız olursa, uyarır ve Lyapunov gözlemcisi eklenmiş sıralı `Sim.run()` çağrılarına geri döner.

```python
from dynlib.plot import series

res = sweep.lyapunov_mle_sweep(
    sim,
    param="r",
    values=np.linspace(3.0, 4.0, 400),
    N=5000,
    transient=1000,
    record_interval=10,
)
series.plot(x=res.values, y=res.mle, xlabel="r", ylabel="λ_max")
```

### `lyapunov_spectrum_sweep`
Bir parametre ızgarası boyunca ilk `k` Lyapunov üssünü hesaplayın. MLE taraması gibi, bu da JIT + sabit `dt` hızlı yol yürütmesi için ayarlanmıştır (ve teğet uzayı için isteğe bağlı bir `init_basis` kabul eder), ancak toplu hızlı yol veya işlem paralelliği mevcut olmadığında gözlemci eklenmiş sıralı `Sim.run()` işlemine de geri döner. `outputs` sözlüğü her zaman şunları içerir:

- `spectrum`: normalleştirilmiş üsleri içeren `(M, k)` şeklindeki dizi
- `log_r`: ham logaritmik büyüme değerleri (`(M, k)` şeklinde)
- `steps`: değer başına algoritma adımlarının son sayısı
- `lyap0`, `lyap1`, … `lyap{k-1}`: her üs sütunu için uygun takma adlar

`traces` yoktur, çünkü altta yatan gözlemci yalnızca en son spektrumu yayar. `parallel_mode`/`max_workers` ve `record_interval` kullanımını tıpkı MLE taramasındaki gibi yapın; yardımcı gerektiğinde sıralı yürütmeye de geri döner.

## Pratik notlar

- Hızlı yol kapalıysa (`fastpath_batch_for_sim`, `None` döndürürse), taramanın `Sim.run()`'a geri döndüğüne dair bir uyarı görürsünüz. `jit=True` sağlamak, sabit adımlı stepper'lar kullanmak ve açık `dt`/`N` değerleri kaydetmek hızlı yolu sağlıklı tutar.
- Yörünge (trajectory) taramaları için `record_interval` ve `max_steps`, bellek/işlemci takası yapmanıza olanak tanır. Yörüngeler tam olarak üretildiği gibi saklanır, böylece saçılım bulutları, uç değerler veya kırpılmış örnek setleri oluşturmak üzere `res.bifurcation("x")` aracılığıyla `dynlib.analysis.post.bifurcation.BifurcationExtractor` ile bunları tekrar kullanabilirsiniz.
- Lyapunov yardımcıları `analysis_kind` (varsayılan `1`) kabul eder, böylece sisteminize en uygun varyantı seçebilirsiniz. `max_workers`, `_resolve_process_workers` aracılığıyla makine çekirdeklerine (maksimum 8 ile sınırlı) varsayılan olarak ayarlanır.
- `traj_sweep` çalıştıran, `res.bifurcation("x")` çıkaran ve sonucu `dynlib.plot.bifurcation_diagram`'a besleyen uçtan uca bir senaryo için `examples/bifurcation_logistic_map.py` örneğine bakın.
- Tüm taramalar, verilerin nasıl oluşturulduğunu izleyebilmeniz için stepper ayarlarını, zaman damgalarını ve herhangi bir paralel çalıştırma yapılandırmasını içeren `meta` verilerini döndürür.