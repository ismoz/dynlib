# Manifold Çizimleri

`plot.manifold()`, 1B (bir boyutlu) manifold izlerini (kararlı/kararsız kollar, heteroklinik bağlantılar, homoklinik döngüler) tutarlı bir stil ve lejant (gösterge) yönetimi ile 2B izdüşümler olarak görselleştirir. Bu, manifold analizi araçlarının çizim arkadaşıdır; böylece siz manifoldları çıkarmaya odaklanabilir ve analizlerin ürettiği segmentleri görselleştirmek için bu yardımcıya güvenebilirsiniz.

## Yardımcının çizdiği şeyler

`plot.manifold()`, bir manifold kolunu takip eden `(state_x, state_y)` (durum_x, durum_y) örneklerinden oluşan diziler bekler. Ham segmentleri veya daha üst seviye sonuçları kabul eder ve seçilen durum bileşenlerini `x`/`y` eksenlerine izdüşürür. Sağlanan her bir kol, ya bir `LineCollection` (düz çizgiler için) ya da bireysel `plot` çağrıları (işaretçiler/marker istendiğinde) olarak görünür ve `None` olmayan bir etikete sahip herhangi bir kol otomatik olarak lejanta dahil edilir.

## Veri sağlama (Feeding the helper)

- **Segments (Segmentler):** Her `arr`'in bir `(adımlar, durumlar)` dizisi olduğu `segments=[arr1, arr2, …]` yapısını iletin. Yardımcı fonksiyon, her dizinin en az iki satıra ve istenen `components` (bileşenleri) kapsayacak kadar sütuna sahip olmasını zorunlu kılar.
- **Branches (Kollar):** Pozitif/negatif kolları kendiniz yönetiyorsanız `(branch_pos, branch_neg)` gibi bir `branches` demeti (tuple) sağlayın. Her kol listesi birden fazla segment içerebilir.
- **Result objects (Sonuç nesneleri):** Çoğu manifold analizi sonucu, `plot.manifold()` ile uyumlu bir `branches` özelliğine (attribute) sahiptir, bu nedenle onları doğrudan iletebilirsiniz:
  - `dynlib.analysis.trace_manifold_1d_map(...)` / `trace_manifold_1d_ode(...)`, iki kol listesi (pozitif/negatif) içeren `ManifoldTraceResult` döndürür.
  - `dynlib.analysis.heteroclinic_tracer(...)` ve `dynlib.analysis.homoclinic_tracer(...)` sırasıyla `HeteroclinicTraceResult` ve `HomoclinicTraceResult` döndürür; her ikisi de `.branches` özelliğini (ikincisinin kolları tek izlenen yörüngeyi içerir) ve ayrıca `.kind`/`.meta` özelliklerini açığa çıkarır, böylece `plot.manifold()` bunları otomatik olarak etiketler.
  - Kendi segmentlerinizi özel bir yapıda sarmalıyorsanız, `result=…` olarak geçmeden önce bu yapının, iki demetlik (tuple) dizilere çözümlenen bir `branches` özelliği sağladığından emin olun.

```python
from dynlib import setup
from dynlib.analysis import trace_manifold_1d_map, heteroclinic_tracer
from dynlib.plot import fig, manifold

sim = setup("models/henon.map", stepper="map")
unstable = trace_manifold_1d_map(sim, kind="unstable", branch_len=500)
hex_trace = heteroclinic_tracer(sim, source_eq="E0", target_eq="E1", preset="default")

ax = fig.single()
manifold(result=unstable, components=(0, 1), label="Kararsız manifold", ax=ax)
manifold(result=hex_trace, components=(0, 1), style="discrete", label="Heteroklinik yörünge", ax=ax)
```

## Kolları stillendirme

`style`, `color`, `lw`, `ls`, `marker`, `ms` ve `alpha` diğer `plot` yardımcıları gibi davranır, ancak `style` ayrıca yerleşik hazır ayarları da kabul eder:

- `"continuous"`, `"flow"` / `"cont"`: İşaretçisiz düz çizgi (ODE kaynaklı kollar için idealdir).
- `"discrete"`, `"map"`: Sadece işaretçi (ayrık zamanlı manifoldlar için iyidir).
- `"mixed"` / `"connected"`: Çizgilerle birbirine bağlanan işaretçiler.
- `"line"` / `"scatter"`: Sadece çizgiler veya sadece işaretçiler için açık kısaltmalar.

Kol bazında geçersiz kılmalar `groups` ile yönetilir. Her grup bir eşleme (`{"segments": …, "label": …, "style": …}`) veya bir demettir `(segments, label?, style?)` (? isteğe bağlı demektir; `None` olabilir). Yardımcı fonksiyon küresel `style`'ı devralır ancak her grup için hazır ayar geçersiz kılmalarını veya açık eşlemeleri karıştırır; bu da kararlı kolu kararsız koldan veya heteroklinik izden farklı renklendirmenize olanak tanır.

## İzdüşüm ve eksen seçimi

Hangi durum indislerinin (`int` türünde) çizileceğini seçmek için `components=(i, j)` kullanın (örneğin, ilk iki durum için `(0, 1)`). Bileşenler farklı olmalı ve sağlanan segmentlerin boyutsallığı içinde kalmalıdır.

Eksen etiketleri, limitler ve en boy oranları şunlarla kontrol edilir:

- `xlabel`, `ylabel`, `title`, `xlabel_fs`, `ylabel_fs`, `title_fs`
- `xlim`, `ylim`, `aspect`, `xlabel_rot`, `ylabel_rot`
- `xpad`, `ypad`, `titlepad` ekstra boşluklar için

Sonucunuz meta veriler sağlıyorsa (örneğin, bir `ManifoldTraceResult`'tan gelen `result.meta`), `manifold()` çağırmadan önce çizim başlığını/etiketlerini not etmek için bunu yeniden kullanabilirsiniz.

`plot.manifold()` bir `Axes` (eksen) nesnesi döndürür, böylece [Çizim Süslemeleri](decorations.md) bölümünde açıklandığı gibi süslemeler (dikey/yatay çizgiler, bantlar) katmanlayabilir veya manifoldu `fig.grid()`/`plot.fig()` tarafından oluşturulan çok panelli figürlere entegre edebilirsiniz.

## Lejant ve gruplama ipuçları

- Her kolu etiketlemek için yardımcı üzerinde veya `groups` aracılığıyla `label=` ayarlayın. Lejant yalnızca en az bir etiket ayarlandıysa ve `legend=True` (varsayılan) ise görünür.
- Farklı stillere veya renklere sahip kol parçalarını üst üste bindirmek için `groups` kullanın (örneğin, heteroklinik bir izin ilk segmentini daha kalın bir çizgiyle vurgularken geri kalanını ince tutmak gibi).
- Birden fazla manifoldu birlikte çizerken (örneğin, kararlı vs. kararsız), yinelenen tutamaçlardan kaçınmak için `legend=True` değerini yalnızca son çağrıda iletin.

## İpuçları

- Yalnızca pencereli bir görünüm istiyorsanız, çizimden önce büyük sonuç dizilerini dilimleyin; yardımcı fonksiyon sağlanan segmentlere tam olarak uyar.
- Aynı `ax=` değerini geçirerek ve `legend`'ı kontrol ederek `plot.manifold()`'u diğer çizim yardımcılarıyla (faz portreleri, zaman serileri) birleştirin.
- İşaretçiler istenmediğinde `linecollection` kullanıldığından, `alpha`/`linewidth` tüm segmentlere eşit olarak uygulanır.
- Aynı stili çağrılar arasında yeniden kullanmak istiyorsanız, hazır ayar dizesini tekrarlamak yerine bir sözlük tutun ve her grup için güncelleyin (`style={"color": "C0"}`).

`plot.manifold()`, analiz araçlarının görünümünü ve iş akışını yansıtır, bu nedenle bir `ManifoldTraceResult`, `HeteroclinicTraceResult` veya `HomoclinicTraceResult` elde ettiğinizde, ek veri düzenlemesi yapmadan manifoldu belgeleyebilir ve stillendirebilirsiniz.