# Lyapunov analizi

Dynlib, **maksimum Lyapunov üssünü (MLE)** ve tam **Lyapunov spektrumunu** hesaplamak için çalışma zamanına entegre “observer” (gözlemci) modülleri sağlar. Bu gözlemciler, `dynlib.runtime.observers` altında **factory** (gözlemci fabrikası) olarak sunulur. Observer’ların çalışma zamanı tarafındaki ayrıntıları genel olarak [observers rehberinde](observers.md) anlatılır; bu sayfa ise özellikle `lyapunov_mle_observer` ve `lyapunov_spectrum_observer`’ı, ister tek başına ister başka observer’larla birlikte, pratikte nasıl kullanacağınızı özetler.

## Lyapunov tanılamaları ne zaman kullanılır?

Lyapunov üsleri, yörünge boyunca **sonsuz küçük pertürbasyonların** ortalama olarak nasıl büyüyüp küçüldüğünü nicelendirir:

- **Pozitif MLE**, başlangıç koşullarına hassas bağımlılığı ve kaotik rejimleri işaret eder.
- **Sıfır veya sıfıra yakın MLE**, çoğunlukla periyodik ya da yarı-periyodik hareketle ilişkilidir.
- **Negatif MLE**, kararlı çekicileri (özellikle kararlı sabit noktalar / “sink”) gösterir.

**Lyapunov spektrumu** ise tek bir üsle yetinmeyip birden fazla yöndeki genişleme/daralma oranlarını birlikte verir. Birden çok kararsız/kararlı yönün önemli olduğu sistemlerde (ör. daha yüksek boyutlu akışlar ya da bazı haritalar) daha bütünlüklü bir kararlılık resmi sağlar.

## Observer factory’leri

İki factory de `dynlib.runtime.observers` içindedir ve aynı genel kullanım desenini paylaşır:

```python
from dynlib.runtime.observers import lyapunov_mle_observer, lyapunov_spectrum_observer

factory = lyapunov_mle_observer(model=sim.model, record_interval=record_every)
```

### Ortak argümanlar

- `model`: Derlenmiş `Model`. Doğrudan `model=...` verebilir ya da factory modunda bırakıp `Sim.run(...)` sırasında modelin enjekte edilmesini sağlayabilirsiniz.
- `mode`: `"flow"`, `"map"` veya `"auto"`.  
  - `"auto"` seçilirse `model.spec.kind` üzerinden türetilir (`"ode" -> "flow"`, `"map" -> "map"`).  
  - `"flow"`: payda (denominator) zamandır.  
  - `"map"`: payda iterasyon/adım sayısıdır.
- `record_interval` / `trace_plan`: Yakınsama izinin (trace) hangi aralıkla örnekleneceğini belirler. Uygulamada izleme, bu observer’ların standart çıktısının bir parçasıdır; bellek kullanımını azaltmak için daha büyük bir `record_interval` veya uygun bir `trace_plan` seçin.
- `prefer_variational_combined` (MLE için): Stepper destekliyorsa, durum + tanjant vektörünün **birleşik** entegrasyonunu (combined variational stepping) tercih eder; destek yoksa stepper’ın **tanjant-adımı** yoluna (tangent-only variational stepping) düşer.

### Spektruma özgü argümanlar

- `k`: Hesaplanacak **önde gelen** üs sayısı. İlk üs MLE’dir. (`1 <= k <= n_state` olmalı.)
- `init_basis`: İsterseniz başlangıç için `(n_state, k)` boyutlu bir başlangıç tabanı verebilirsiniz; verilmezse kanonik taban kullanılır.

### Çalıştırıcı/stepper gereksinimleri

Her iki observer da `ObserverRequirements` üzerinden gereksinimlerini beyan eder (özellikle `need_jvp=True`). Ayrıca:

- `"flow"` modunda, sayısal olarak tutarlı sonuçlar için stepper’ın **varyasyonel adımlama** desteği gerekir. Bu destek yoksa observer hata verir ve uyumlu stepper’lara yönlendirir.
- `"map"` modunda, tanjant evrimi `J*v` (Jacobian–vektör çarpımı, JVP) ile yürütüldüğünden stepper’ın varyasyonel adım desteği zorunlu değildir; ancak modelin `jvp` sağlaması gerekir.

## Tipik iş akışı

1. `Sim.run(...)` çağrısında observer’ları ekleyin.
2. Yakınsama eğrisini izlemek istiyorsanız `record_interval` veya `trace_plan` ile iz örneklemesini ayarlayın.
3. Çalıştırma bittiğinde `sim.results()` üzerinden observer çıktıları ve izlerini okuyun.

### Örnek (lojistik harita)

Lojistik harita demosu (`examples/analysis/lyapunov_logistic_map_demo.py`) tipik bir kurulum gösterir:

```python
sim.assign(x=0.4, r=4.0)
sim.run(
    N=5000,
    dt=1.0,
    record_interval=1,
    observers=[
        lyapunov_mle_observer(record_interval=1),
        lyapunov_spectrum_observer(k=1, record_interval=1),
    ],
)
result = sim.results()
```

## Çıktıları okuma

Her Lyapunov observer’ı iki tür veri üretir:

1. **Çıkış alanları (output)**: Çalışma sonunda biriken/nihai istatistikler.
2. **İz (trace)**: Çalışma boyunca örneklenen “anlık tahminler” (yakınsama eğrisi).

### MLE observer çıktıları

`lyapunov_mle_observer` modülü aşağıdaki output alanlarını üretir:

- `log_growth`: Biriken `log(||v||)` toplamı
- `denom`: `"flow"` için toplam zaman, `"map"` için toplam adım sayısı
- `steps`: adım sayacı
- `variational_mode`: kullanılan varyasyonel yolun meta bilgisi (0: basit/Euler, 1: combined, 2: tangent-only)

Trace tarafında ise:

- `mle`: `log_growth / denom` olarak örneklenen MLE yakınsama izi

### Spektrum observer çıktıları

`lyapunov_spectrum_observer` modülü:

- `log_r0 ... log_r{k-1}`: QR (Modified Gram–Schmidt) adımında biriken `log(diag(R))` toplamları
- `denom`, `steps`, `variational_mode`

Trace tarafında:

- `lyap0 ... lyap{k-1}`: her bir üs için `log_rj / denom` yakınsama izleri

`ResultView.observers` sözlüğü üzerinden ilgili anahtarla erişip (ör. `"lyapunov_mle"`, `"lyapunov_spectrum"`), çıktı alanlarını ve izleri inceleyebilirsiniz. İzleri, kaydedilen yörünge ile hizalamak için `trace_steps`, `trace_time` gibi yardımcılar ve doğrudan indeksleme (örn. `result["..."]`) kullanılabilir.

## İpuçları

- MLE’yi başka tanılamalarla birlikte çalıştıracaksanız, mümkünse aynı `trace_plan` / `record_interval` düzenini kullanın; böylece izler aynı örnekleme ızgarasına oturur.
- Bellek kullanımını düşürmek için `record_interval` değerini büyütün. Yakınsama eğrisini “yaklaşık” görmek çoğu zaman yeterlidir.
- Spektrum izlerinde `k`’yi gereksiz büyük seçmeyin: çalışma alanı ve hesap maliyeti `k` ile birlikte artar.
- `"flow"` modunda hata alıyorsanız, seçtiğiniz stepper’ın varyasyonel adımlamayı desteklemediği anlamına gelir. Bu durumda ya uyumlu bir stepper seçin ya da model türüne uygunsa `"map"` modunu tercih edin.
- Yalnızca nihai değere odaklanıyorsanız, izleri çizmek yerine output alanlarından (`log_growth/denom` veya `log_rj/denom`) nihai sonucu türetin; trace’i sadece gerektiğinde inceleyin.
