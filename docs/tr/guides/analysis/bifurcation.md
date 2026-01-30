# Çatallanma (Bifurcation) diyagramları

Çatallanma diyagramı, bir **parametre taramasının çıktısını sonradan işleyerek** üretilir: `dynlib.analysis.sweep` içindeki tarama yardımcıları (ör. `traj_sweep`, `lyapunov_mle_sweep`, `lyapunov_spectrum_sweep`) her parametre değeri için bir yörünge (veya teşhis dizisi) kaydeder. Ardından `SweepResult.bifurcation(var)` bu kayıtları alır ve diyagramda görünen **saçılım noktalarına** (x-ekseni: parametre, y-ekseni: seçilen değişken) dönüştürür.

## İş akışı

1. **Taramayı çalıştırın:** İncelediğiniz parametre için `dynlib.analysis.sweep.traj_sweep` (veya başka bir tarama yardımcısı) ile tarama yapın. Diyagramda kullanacağınız değişkenleri `record_vars` ile kaydedin. Başlangıçtaki **geçici rejimi** (transient) atmak istiyorsanız tarama tarafında `transient`/kayıt aralığı ayarlarını kullanın; ayrıca çıkarım aşamasında `tail` gibi seçeneklerle yalnızca son kısımları alabilirsiniz.

2. **Çatallanma noktalarını çıkarın:** `result.bifurcation("x")` bir `BifurcationExtractor` döndürür. Bu nesne, farklı çıkarım stratejileri için aşağıdaki yöntemleri sağlar:
   - `.all()` : Kaydedilmiş **tüm** noktaları kullanır (filtre yok).
   - `.tail(n)` : Her parametre değeri için kaydın **son n örneğini** alır.
   - `.final()` : Her parametre değeri için **son tek noktayı** alır.
   - `.extrema(...)` : Yerel **maksimum/minimum** noktalarını (veya ikisini birden) çıkarır. İsteğe bağlı `tail=...` ile yalnızca son bölümde arama yapar; `min_peak_distance` ile art arda tepe/çukur seçiminde mesafe kısıtı uygular.
   - `.poincare(...)` : Seçilen `section_var` serisinin `level` düzeyini belirli yönde (`direction`) kestiği anlarda, hedef değişkenin **kesit değerlerini** (doğrusal enterpolasyonla) üretir.

3. **Sonucu çizin:** `dynlib.plot.bifurcation_diagram(extractor)` ile doğrudan çizebilir veya çıkarıcının `p`/`y` dizilerini kendi çizim rutinlerinize verebilirsiniz. Üretilen `BifurcationResult`, eksen etiketleri ve tarama ızgarası için gerekli bilgileri (`param_name`, `values`, `meta`) birlikte taşır.

Bifürkasyon diyagramı hesaplama ve çizdirme için aşağıdaki örneklere bakabilirsiniz:
- [Bifürkasyon Örnekleri](../../examples/bifurcation.md)

## Notlar

- Tarama sırasında bazı parametre noktalarında hiç örnek kaydedilmediyse, `.tail()`/`.final()` gibi modlar hata verir (ör. `No samples recorded; ...`). Böyle bir durumda `T/N` değerlerini artırın veya `record_interval` ayarını gözden geçirin.
- Çıkarıcı, parametre noktaları arasında **farklı uzunlukta** yörüngeler olsa bile çalışacak şekilde tasarlanmıştır; her parametre değeri için ilgili seriyi ayrı işler ve sonuçları birleştirir.
- Birçok tarama yardımcısı `meta` içine stepper ayarlarını (`dt`, `record_interval` vb.) ekler. Bu sayede diyagramdaki noktaların aynı koşullarda üretilip üretilmediğini sonradan denetleyebilirsiniz.
- Yayın kalitesinde figürler için, çatallanma çıktısını kendi düzeninizde kullanmak üzere `p`/`y` dizilerini alıp `series.plot`, `phase.xy` veya `fig.grid` gibi çizim yardımcılarıyla birleştirmek genellikle daha esnektir.