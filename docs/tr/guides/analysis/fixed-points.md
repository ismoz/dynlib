# Sabit noktalar ve dengeler

Sabit noktalar (haritalar için) ya da denge noktaları (ODE modelleri için), **kök bulma** diliyle `g(x) = 0` koşulunu sağlayan durum vektörleridir.

- ODE’lerde bu genellikle `f(x, p) = 0` (vektör alanı sıfır).
- Haritalarda ise sabit nokta, `x_{n+1} = F(x_n, p)` için `F(x, p) - x = 0` demektir.

Dynlib bunu bulmak için iki giriş noktası sunar:

1. `dynlib.analysis.fixed_points` içindeki bağımsız yardımcı: **`find_fixed_points(...)`**  
   Kullanıcı tarafından sağlanan `f(x, params)` (ve isteğe bağlı `jac(x, params)`) üzerinden çalışır.

2. Derlenmiş model üzerinde kolay yöntem: **`FullModel.fixed_points(...)`**  
   Derlenmiş modelin parametre varsayılanlarını, çalışma zamanı çalışma alanını ve (varsa) analitik Jacobian’ını otomatik bağlayarak aynı çözücüyü kullanır.

Her iki yol da özünde aynı Newton tabanlı çözücüyü çalıştırır (yakınsama ölçütleri, çözüm birleştirme ve kararlılık teşhisleri dahil). Pratikte çoğu kullanıcı için tercih edilen yol `FullModel.fixed_points(...)` olur; çünkü parametreleri/tohumları/Jacobian’ı derlenmiş modelden doğru biçimde bağlar.

> Not: API sade kalsın diye dokümantasyonda “fixed point” terimi kullanılır. ODE bağlamında bu, “denge noktası (equilibrium)” ile aynı şeydir.

---

## Hızlı başlangıç

Sabit nokta bulmak için tipik olarak şunlar gerekir:

- Bir model (`FullModel.fixed_points(...)` için) **veya** bir sağ taraf fonksiyonu (`find_fixed_points(...)` için),
- Beklenen çözümlere yakın **başlangıç tahminleri** (tohumlar / *seeds*),
- İsteğe bağlı parametreler ve çözücü ayarları.

Derlenmiş bir model ile:

```python
from dynlib.analysis import FixedPointConfig

# Modelinizi oluşturun veya yükleyin
model = ...  # örn: from dynlib import build; model = build("model.toml")

# Çözücüyü yapılandırın
cfg = FixedPointConfig(tol=1e-10, classify=True)

# Sabit noktaları bulun
result = model.fixed_points(
    params={"param_name": value},  # isteğe bağlı: parametre geçersiz kılmaları
    seeds=[[x1, y1], [x2, y2]],    # başlangıç tahminleri (n_seeds, n_state)
    cfg=cfg,
)

print("Fixed points:", result.points)
print("Stability:", result.stability)
```

Bu çağrı, çözümleri ve tanılama bilgilerini içeren bir `FixedPointResult` döndürür.

---

## `find_fixed_points(...)`

Bu yardımcı, verilen bir veya daha fazla tohumdan başlayarak **`f(x, params) = 0`** denklemini Newton yöntemiyle çözer.

### Girdi sözleşmeleri

- **Fonksiyon imzası:** `f(x, params)` NumPy dizileri alır ve **`x` ile aynı şekle sahip** bir çıktı döndürmelidir.  
- **Jacobian (isteğe bağlı):** `jac(x, params)` sağlanırsa, çıktısı çözüm sırasında `(n_state, n_state)` şekline zorlanır/doğrulanır.

### Tohum ve parametre biçimleri

- `seeds`, tek bir vektör (`(n_state,)`) ya da bir parti (`(n_seeds, n_state)`) olabilir. Tek vektör verilirse otomatik olarak tek tohumluk partiye çevrilir.
- Şekil uyuşmazsa yardımcı **hata verir** (sessizce düzeltmeye çalışmaz).
- `params`, **1-boyutlu** bir vektör olmalıdır. Parametre yoksa `None` geçebilirsiniz; bu durumda sıfır uzunluklu vektör kullanılır.

### Newton ayarları

`cfg` (aşağıdaki `FixedPointConfig`) şu davranışları belirler:

- yakınsama toleransı (`tol`)
- tohum başına azami iterasyon (`max_iter`)
- sonlu fark Jacobian adımı (`fd_eps`)
- özdeğer tabanlı kararlılık etiketlemesi (`classify`)
- problemin türü (`kind`: `"ode"` veya `"map"`)

Çözücü, artık normu `tol` altına düşer düşmez ilgili tohum için yakınsamış sayar.

### Yinelenen çözümleri birleştirme (de-duplication)

Çözümler bulunduğunda, yakın olan kökler birleştirilir:

- `unique_tol > 0` ise, birbirine `unique_tol` mesafesinden yakın çözümler aynı kabul edilir.
- Birleştirme sırasında, aynı kümeye düşen adaylardan **artığı daha küçük olan** korunur.
- `unique_tol` değerini `None` veya `≤ 0` yapmak birleştirmeyi kapatır (yakınsayan her tohum ayrı ayrı tutulur).

### Kararlılık sınıflandırması

`cfg.classify=True` ise her benzersiz çözüm için özdeğerler hesaplanır ve `cfg.stability_tol` marjına göre etiket verilir:

- Etiketler: `stable`, `unstable`, `neutral`, `saddle`
- ODE’lerde ölçüt: özdeğerlerin **gerçel kısımları** (tamamı < 0 ise kararlı vb.)
- Haritalarda ölçüt: özdeğerlerin **mutlak değeri** (tamamı < 1 ise kararlı vb.)

> Haritalar için not: Çözücü kök fonksiyonunu `g(x)=F(x)-x` biçiminde ele aldığı için, kararlılık analizi sırasında özdeğerler “haritanın türevi” üzerinden değerlendirilir (bu amaçla gerekli dönüşüm içerde yapılır).

### Tanılama (meta)

Her çalıştırma, tohum bazında ve benzersiz çözüm bazında teşhis bilgilerini kaydeder. `FixedPointResult.meta` içinde şunlar bulunur:

- `seed_points`: Her tohumdan çıkan son iterat (yakınsamasa bile).
- `seed_residuals`: Her tohum için son artık normu.
- `seed_converged`: Her tohumun yakınsayıp yakınsamadığı.
- `seed_iterations`: Tohum başına iterasyon sayısı.
- `seed_to_unique`: Tohum indeksinden benzersiz çözüm indeksine eşleme (`-1`: yakınsamadı).
- `unique_seed_indices`: Benzersiz çözümleri temsil eden kaynak tohum indeksleri.
- `params`: Kullanılan parametre vektörü.
- `config`: Kullanılan `FixedPointConfig` nesnesi.

---

## `FixedPointConfig`

| Alan | Açıklama |
| --- | --- |
| `method` | Çözücü yöntemi adı (şimdilik yalnızca `"newton"`). |
| `tol` | Yakınsama için artık (residual) toleransı (varsayılan `1e-10`). |
| `max_iter` | Tohum başına en fazla Newton iterasyonu (varsayılan `50`). |
| `unique_tol` | Çözümleri birleştirmek için mesafe eşiği (varsayılan `1e-6`). Birleştirmeyi kapatmak için `None` veya `≤ 0`. |
| `jac` | Jacobian modu: `"auto"` (varsa sağlanan/analitik, yoksa sonlu fark), `"fd"` (sonlu farkı zorla), `"provided"` (mutlaka `jac` verilmiş olmalı). |
| `fd_eps` | Sonlu fark Jacobian’ı için adım boyu (varsayılan `1e-6`). |
| `classify` | Özdeğerleri hesaplayıp kararlılık etiketi üret (varsayılan `True`). |
| `kind` | `"ode"` veya `"map"`. `find_fixed_points` bunu doğrular ve sınıflandırmada kullanır. |
| `stability_tol` | Nötrlük marjı (ODE: sanal eksen civarı; harita: birim çember civarı), varsayılan `1e-6`. |

---

## `FixedPointResult` nasıl yorumlanır?

`FixedPointResult` bulunan sabit noktaları ve ilgili çıktılarını taşır:

- `points`: **(n_points, n_state)** biçiminde NumPy dizisi. (Birleştirme açıksa `unique_tol` ile tekilleştirilmiştir.)
- `residuals`: Her benzersiz noktanın artık normu (yakınsayan çözümlerde çok küçük olmalıdır; ör. `~1e-10` ve altı).
- `jacobians`: Her benzersiz noktada kullanılan Jacobian matrisleri listesi veya `None`.  
  - Haritalarda burada tutulan Jacobian, kök fonksiyonuna (`F(x)-x`) aittir.
- `eigvals`: Her nokta için özdeğer dizileri listesi veya `None` (yalnızca `classify=True` iken).
- `stability`: Her nokta için kararlılık etiketi listesi veya `None`.
- `meta`: Ayrıntılı tanılama sözlüğü (bkz. üstteki “Tanılama”).

Bir tohum yakınsamazsa, önce `result.meta["seed_converged"]` ve `result.meta["seed_residuals"]` alanlarına bakın; çoğu durumda daha iyi tohumlar, daha uygun `max_iter` veya `tol` seçimi gerekir.

---

## `FullModel.fixed_points(...)`

Derlenmiş bir `FullModel` üzerinde bu yöntemi çağırmak, kök bulmayı modelin çalışma zamanı üzerinden yürütür; böylece tutarlı varsayılanlar ve (varsa) analitik Jacobian kullanımı otomatikleşir:

- `params` ve `seeds`:
  - doğrudan dizi/vektör verebilirsiniz,
  - ya da ad → değer eşlemesi vererek modelin varsayılan parametre/durum vektörünü “seçerek” güncelleyebilirsiniz.
- `cfg` vermezseniz varsayılan `FixedPointConfig()` kullanılır; tek tek `tol`, `max_iter`, `unique_tol`, `classify` gibi alanları ayrıca geçerseniz `cfg` üzerinde ilgili alanlar geçersiz kılınır.
- `jac` seçeneği:
  - `"auto"`: model Jacobian’ı varsa kullan, yoksa sonlu farka düş
  - `"fd"`: sonlu farkı zorla
  - `"analytic"`: model Jacobian’ını zorla (model Jacobian sunmuyorsa hata)
- `t`: otonom olmayan sistemlerde değerlendirme zamanı. Varsayılan `spec.sim.t0`’dır.

> Haritalar için: yöntemin içerde kurduğu kök fonksiyonu `F(x)-x` biçimindedir; bu nedenle Newton çözümü ve Jacobian işlemleri bu kök fonksiyonuna göre yürütülür. Kararlılık analizi ise uygun dönüşümle haritanın türevi üzerinden yapılır.

---

## Örnek

Lojistik harita için bir örnek: `x_{n+1} = r * x_n * (1 - x_n)`. `r=3.8` civarında sabit noktalar yaklaşık `0.0` ve `0.7368` olur.

```python
from dynlib.analysis import FixedPointConfig

top_cfg = FixedPointConfig(unique_tol=1e-8, classify=True)
result = model.fixed_points(
    params={"r": 3.8},
    seeds=[[-0.5], [0.4]],  # 1D durum için tohumlar (n_seeds, 1)
    cfg=top_cfg,
)

print("Number of fixed points found:", len(result.points))
print("Fixed points:")
for i, point in enumerate(result.points):
    print(f"  Point {i}: {point}")
print("Stability labels:", result.stability)
print("Residuals (should be near 0):", result.residuals)
```

Bu örnek, birden fazla tohumdan başlayıp benzersiz sabit noktalara nasıl yakınsandığını ve kararlılığın özdeğerlerle nasıl etiketlendiğini gösterir.

---

## İpuçları ve yaygın sorunlar

- **Tohum seçimi:** Rastgele atmak yerine, kısa simülasyonlar çalıştırıp yörüngenin “yakınından geçtiği” bölgeleri tohum olarak kullanın. Gerekirse vektör alanını/harita iterasyonlarını kaba bir ızgarada tarayın.
- **Yakınsamıyor:**  
  - `max_iter` artırın,  
  - `tol` değerini çok agresif seçmeyin (aşırı küçük tol bazen gereksiz iterasyon doğurur),  
  - daha iyi tohumlar kullanın,  
  - mümkünse analitik Jacobian sağlayın.
- **Aynı çözüm tekrar tekrar geliyor:** `unique_tol` değerini büyütmeyi deneyin (çok büyük seçmek farklı çözümleri de birleştirebilir).
- **Kararlılık etiketleri beklediğiniz gibi değil:** Harita/ODE ayrımının doğru olduğundan emin olun; `FullModel.fixed_points(...)` bunu modelden otomatik ayarlar. `find_fixed_points(...)` kullanıyorsanız `cfg.kind` değerini doğru verin.
- **Performans:** Analitik Jacobian, hem hız hem de doğruluk açısından genellikle sonlu farka göre daha iyidir.
- **Şekil hataları:** `seeds` boyutunun `(n_seeds, n_state)` ile, `params` boyutunun `(n_params,)` ile uyumlu olduğundan emin olun (özellikle `find_fixed_points(...)` kullanırken).

