# Analiz Sonrası (Post-analysis)

Son-işleme (post-processing), ham simülasyon veya tarama sonuçlarını önemsediğiniz içgörülere dönüştürme yöntemidir: özet istatistikler, yükselme/düşme süreleri, eşik geçişleri ve çatallanma (bifurcation) saçılım bulutları gibi. Dynlib, kaydedilen yörüngeleri (trajectories) yerinde tutar, böylece analiz yardımcılarını (helpers) zaten çizdirdiğiniz veya dışa aktardığınız zaman ekseniyle aynı hizada kullanabilirsiniz.

## `ResultsView` üzerinden yörünge analizcilerine

Bir çalıştırmadan sonra normalde `res = sim.results()` çağrısı yaparsınız (tam API için [Simülasyon sonuçları rehberine](../simulation/results.md) bakın). `res.analyze(...)` ihtiyacınız olan analizciyi (analyzer) oluşturur:

- `res.analyze("x")`, tek bir değişken için `TrajectoryAnalyzer` döndürür.
- `res.analyze(["x", "y"])` veya herhangi bir açık demet (tuple), istenen sütunlar için `MultiVarAnalyzer` döndürür.
- Argümansız `res.analyze()`, kaydedilen durumları tercih eder (eğer durum kaydedilmediyse kaydedilen yardımcı değişkenlere, yani aux değişkenlerine başvurur) ve `MultiVarAnalyzer` döndürür.

Her iki analizci de kaydedilen NumPy görünümlerini (views) sarmalar, bu nedenle tüm istatistikler ve zamansal yardımcılar `res.t` içinde gördüğünüz aynı ızgara (grid) üzerinde çalışır.

```python
res = sim.results()
xa = res.analyze("x")
peak_time, peak_value = xa.argmax()
summary = res.analyze().summary()  # değişken başına istatistik sözlüğü
```

`TrajectoryAnalyzer` şunları sunar:

- temel istatistikler: `min()`, `max()`, `mean()` (ortalama), `std()` (standart sapma), `variance()` (varyans), `median()` (medyan), `percentile(q)`, `summary()`.
- uç değer zamanlaması: `argmin()`, `argmax()`, `range()`.
- zamansal yardımcılar: `initial()`, `final()`, `crossing_times(threshold, direction)` (eşik geçiş zamanları), `zero_crossings(direction)` (sıfır geçişleri), `time_above(threshold)` (eşik üstü süre), `time_below(threshold)` (eşik altı süre).

`MultiVarAnalyzer` aynı yöntemleri yansıtır ancak değişken adına göre anahtarlanmış sözlükler döndürür (ve yeniden oluşturmayı önlemek için değişken başına `TrajectoryAnalyzer` örneklerini "lazy" olarak önbelleğe alır). Birden fazla kaydedilen değişken için yan yana istatistikler istediğinizde bunu kullanın.

## Parametre taramaları → çatallanma verisi

`dynlib.analysis.sweep.traj_sweep(...)` aracılığıyla bir tarama çalıştırdığınızda, dönen `SweepResult`; ızgarayı, çalıştırma başına yükü (payload) ve kaydedilen istatistikleri bir araya getirir. O değişken için bir `BifurcationExtractor` almak üzere `sweep_result.bifurcation("x")` çağrısı yapın; bu yardımcı, yörüngeleri çatallanma diyagramları için gereken saçılım noktalarına "lazy" (ihtiyaç anında) bir şekilde dönüştürür.

```python
from dynlib.analysis import sweep
from dynlib.plot import bifurcation_diagram

sweep_result = sweep.traj_sweep(sim, param="r", values=r_values, record_vars=["x","y"], N=2000)
extrema = sweep_result.bifurcation("x").extrema(kind="max", tail=500, max_points=80)
bifurcation_diagram(extrema)
```

`BifurcationExtractor`, ince bir `BifurcationResult` artı bazı yardımcılar gibi davranır:

- `.all()` (eğer çıkarıcıyı doğrudan bir çizim yardımcısına verirseniz varsayılan moddur), geçici (transient) ve kararlı durum (steady-state) verilerini birlikte inceleyebilmeniz için kaydedilen her noktayı birleştirir.
- `.tail(n)`, parametre başına son `n` örneği tutar (limit döngüsü stabilize olduğunda kullanışlıdır).
- `.final()`, her tarama değeri için yalnızca son örneği tutar, bu da denge noktalarını ve yavaş sürüklenmeleri fark etmenize yardımcı olur.
- `.extrema(...)`, yoğun kümelerden kaçınmak için `max_points` ve `min_peak_distance` parametreleriyle birlikte, (isteğe bağlı) kuyruk (tail) içindeki maksimum/minimum (veya her ikisini) tespit eder.
- `.poincare(section_var=..., level=..., direction=..., tail=..., max_points=..., min_section_distance=...)`, geçiş zamanlarını ve hedef değişkenin karşılık gelen değerini enterpole ederek kesit geçişleri oluşturur.

Her yöntem şunları içeren bir `BifurcationResult` veri sınıfı (dataclass) döndürür:

- `param_name`: taranan parametre (otomatik eksen etiketleme için kullanılır).
- `values`: tam tarama ızgarası (`(M,)` şeklinde).
- `p`, `y`: saçılım grafikleri (scatter plots) için uygun düzleştirilmiş parametre/değer çiftleri dizileri.
- `mode`: veriyi hangi çıkarma stratejisinin ürettiği.
- `meta`: `SweepResult.meta`'nın bir kopyası artı sonucu oluşturan analizci ayarları.

Eksen etiketlerini ve meta verileri yeniden kullanmak için sonucu veya çıkarıcıyı doğrudan `dynlib.plot.bifurcation_diagram()` fonksiyonuna iletin veya kendi çizim araçlarınıza `.p`/`.y` dizilerini verin.

## Güvenilir son-analiz için ipuçları

- Analiz ettiğiniz değişken adının gerçekten `res.state_names`/`res.aux_names` veya `SweepResult.record_vars` içinde göründüğünü doğrulayın; aksi takdirde `res.analyze(...)` veya `sweep_result.bifurcation(...)` hemen hata verir.
- Yalnızca uzun vadeli davranışla ilgileniyorsanız, analizden önce geçici durumları (transients) `.tail(n)` ile kırpın veya `res.segment[...]` ile dilimleyin.
- Değişkenleri karşılaştırmak için `MultiVarAnalyzer` özetlerini kullanın (`mean()`, `{"x": ..., "y": ...}` döndürür) ve belirli bir değişken salınım yapıyorsa `.crossing_times(...)` ile bireysel bileşenlerin detayına inin.
- Bulgularınıza açıklama eklemek için analizci meta verilerini çizim yardımcılarıyla birleştirin (örneğin, `xa.argmax()` ve `res.t` kullanarak küresel maksimum zamanını etiketleyin).