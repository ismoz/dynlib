# Manifold analizi

Dynlib şu anda hem ayrık haritalar (discrete maps) hem de ODE'ler için **1D manifold izlemeyi**, ayrıca ODE modellerinin **heteroklinik ve homoklinik yörüngeleri için arama/izleme yardımcılarını** desteklemektedir. Manifoldları (kararlı/kararsız dallar veya bağlantı yörüngeleri) çıkardıktan sonra, sonuçları doğrudan `dynlib.plot.manifold`'a veya [manifold grafikleri](../plotting/manifold-plot.md) içindeki çizim rehberine besleyebilirsiniz.

## 1D manifold izleme (tracing)

`dynlib.analysis.manifold` içinde iki yardımcı bulunur:

- `trace_manifold_1d_map(...)`: Kararlı veya kararsız alt uzayı 1D olan otonom haritalar için. Kararlı dallar, modelin analitik bir ters harita (`model.inv_rhs`) sunmasını gerektirirken, kararsız dallar yalnızca ileri haritaya (`model.rhs`) ihtiyaç duyar.
- `trace_manifold_1d_ode(...)`: ODE sistemleri için; her zaman dahili bir RK4 entegratörü (Sim stepper'ından bağımsız) kullanır ve bir denge noktasından zamanda ileri (kararsız) veya geri (kararlı) izleme yapar.

### Harita manifoldları (`trace_manifold_1d_map`)

Temel argümanlar ve iş akışı:

- `sim`, `fp`: Derlenmiş modeli bir harita olan bir `Sim` ve genişletmek istediğiniz denge noktasını (sözlük veya dizi kabul edilir) sağlayın.
- `kind="stable"` veya `"unstable"` dalı seçer. Kararlı mod ek olarak `model.inv_rhs` gerektirir.
- `params` ekstra parametreleri geçersiz kılar ve `bounds`, izlemenin bu kutudan çıktığında durması için bir `(n_state, 2)` gözlem kutusu tanımlar.
- `clip_margin` entegrasyon sırasında kesirli bir tampon ekler; `seed_delta` sabit noktayı seçilen özvektör boyunca bozar (perturb).
- `steps`, `hmax`, `max_points_per_segment` ve `max_segments`, örnekleyicinin ne kadar yürüyeceğini ve kaç segment saklayacağını kontrol eder.
- `eig_rank`, `strict_1d`, `eig_unit_tol` ve `eig_imag_tol`, özdeğer seçimini ayarlar; böylece belirli bir kökü zorlayabilir veya katı-1D varsayımını gevşetebilirsiniz.
- `jac="auto" | "fd" | "analytic"`, Jacobian stratejisini seçer; `fd_eps` sonlu fark adımını ayarlar.
- `fp_check_tol`, sağlanan parametrelerde `fp`'nin hala bir sabit nokta olup olmadığını isteğe bağlı olarak doğrular.

Numba/JIT mevcutsa ve model `jit=True` ile derlenmişse, yardımcı hızlı toplu değerlendirme veya önceden ayrılmış hızlı yolları (fastpaths) kullanır; aksi takdirde bir uyarı ile güvenli Python döngülerine geri döner.

### ODE manifoldları (`trace_manifold_1d_ode`)

Temel ayarlar:

- `sim`, `fp`, `params` ve `bounds` yukarıdaki gibi çalışır. `bounds` kutusu entegrasyon sırasında dikkate alınır ve `clip_margin` ile tamponlanabilir; `strict_1d` ise seçilen özvektörün gerçekten 1D bir manifoldu kapsadığından emin olur.
- `dt`, `max_time` ve `max_points` dahili RK4 entegrasyonunu sınırlar. `resample_h` (`None` değilse), daha temiz çizim için her dalı kabaca eşit yay uzunluğu aralığında yeniden örnekler.
- `seed_delta`, dalı normalleştirilmiş özvektör boyunca tohumlar (dal erken ayrılmadığı sürece hem pozitif hem de negatif yönler izlenir).
- Jacobian yönetimi harita yardımcısını yansıtır (`jac`, `fd_eps`, `eig_real_tol`, `eig_imag_tol`). Birden fazla kararlı/kararsız özdeğer mevcut olduğunda `eig_rank` kullanın.
- `fp_check_tol`, `fp` artık kararlı bir durum değilse (örneğin parametre geçersiz kılmaları nedeniyle) izlemeyi reddetmenizi sağlar.

Harita yardımcısı gibi, ODE izleyici de JIT derlenmiş bir modeli tercih eder ancak Numba olmadan da çalışır (geri dönüş/fallback uyarısı ile).

### `ManifoldTraceResult`

Her iki izleme aracı da şu niteliklere sahip bir `ManifoldTraceResult` döndürür:

- `kind`: `"stable"` veya `"unstable"`.
- `fixed_point`: Dalları tohumlayan denge noktası.
- `branches`: `(positive_side, negative_side)` demeti (tuple); her bir taraf bir nokta dizileri listesidir (`(n_points, n_state)` şeklinde `np.ndarray`).
- `branch_pos` / `branch_neg`: Yukarıdaki demetin kullanışlı görünümleri.
- `eigenvalue`, `eigenvector`, `eig_index`, `step_mul`: İzleme sırasında kullanılan spektral bilgiler.
- `meta`: Sonucu üreten yapılandırmayı kaydeden sözlük (bounds, params, dt, clip margins vb.).

Bu sonuçlar doğrudan `dynlib.plot.manifold` tarafından tüketilebilir ve `branches`'ı dışa açar, böylece bunları heteroklinik izler, zaman serileri veya diğer süslemelerle üst üste bindirebilirsiniz (bkz. [çizim rehberi](../plotting/manifold-plot.md)).

Somut örnekler:

- `examples/analysis/manifold_henon.py`, Henon haritası kararlı/kararsız manifoldlarını izler ve `plot.manifold` ile oluşturur.
- `examples/analysis/manifold_ode_saddle.py`, analitik bir eyeri (saddle) ele alarak her iki tarafın nasıl tohumlanacağını, izlenen eğrinin kapalı form ifadelere karşı nasıl kontrol edileceğini ve sonucun nasıl çizileceğini gösterir.

## Heteroklinik ve homoklinik bulucu/izleyici (finder/tracer)

ODE modelleri için, atış (shooting) segmentlerini manuel olarak ayarlamadan bağlantı yörüngelerini arayabilir veya izleyebilirsiniz. İş akışı tipik olarak şöyledir:

1.  Bir bağlantı sağlayan parametre değerini (ve denge çiftini) bulmak için bir **bulucu (finder)** çağırın.
2.  Yörüngeyi kaydetmek için doğrulanan parametrede bir **izleyici (tracer)** kullanın.
3.  Ortaya çıkan izi, `dynlib.plot.manifold` kullanarak kaynak/hedef manifoldlarıyla birlikte çizin.

### Heteroklinik araçlar

- `heteroclinic_finder(...)`: `source_eq_guess`'ten gelen kararsız manifoldun `target_eq_guess`'in kararlı manifolduna yakın bir yere düştüğü `[param_min, param_max]` aralığında bir `param` parametresi arar. Basitleştirilmiş API; `preset` (`"fast"`, `"default"`, `"precise"`), aramayı kısıtlamak için bir `window` ve `gap_tol` (ıskalama mesafesi) ve `x_tol` (parametre hassasiyeti) gibi yakınsama toleranslarını kabul eder. Dönüş değeri `HeteroclinicFinderResult` şunları içerir:
  - `success`: Geçerli bir yörüngenin bulunup bulunmadığı.
  - `param_found`: Iskalamayı en aza indiren parametre değeri.
  - `miss`: Kesişme noktaları, boşluk metrikleri ve çözücü durumunu içeren tanısal yapı (`HeteroclinicMissResult2D` veya `HeteroclinicMissResultND`).
  - `info`: Yardımcı tanılar (ön ayar adı, tarama sayısı vb.).
- Bulucu başarılı olduktan sonra, `heteroclinic_tracer(...)` gerçek bağlantıyı `param_value` değerinde kaydeder. Hem denge noktalarını (`source_eq`, `target_eq`) hem de kararsız yön işaretini (`sign_u`) belirtmelisiniz. İzleyici şunları sunar:
  - `t`, `X`, `meta`, `branches` alanlarına ve bir `success` boolean özelliğine sahip `HeteroclinicTraceResult`.
  - `hit_radius`: Kararsız segmentin durmadan önce hedefe ne kadar yaklaşması gerektiğini kontrol eder (varsayılan `1e-2`).
  - Aynı `preset`/`window`/`t_max`/`r_blow` kısayolları ve daha ince kontrol gerekiyorsa tam `HeteroclinicBranchConfig`.

#### Yapılandırma veri sınıfları (Configuration dataclasses)

Bulucu/izleyici çiftleri, deterministik kontrole ihtiyaç duyduğunuzda `dynlib.analysis.manifold`'dan yapılandırılmış veri sınıflarını (dataclasses) da kabul eder. `heteroclinic_finder` bir `cfg` (`HeteroclinicFinderConfig2D` veya `HeteroclinicFinderConfigND`) alabilirken, `heteroclinic_tracer` `cfg_u` kabul eder; her ikisi de yukarıda açıklanan basitleştirilmiş `preset`, `trace_cfg` ve anahtar kelime argümanlarıyla geçersiz kılınabilir.

- `HeteroclinicRK45Config`: Dahili RK45 entegratörünü (`dt0`, `min_step`, `dt_max`, `atol`, `rtol`, `safety`, `max_steps`) ayarlar; bu, her dal yapılandırmasında saklanır.
- `HeteroclinicBranchConfig`: Tek bir manifold izlemesi için ayarları paketler: denge iyileştirme (`eq_tol`, `eq_max_iter`, isteğe bağlı `eq_track_max_dist`), ayrılma yarıçapı (`eps_mode`, `eps`, `eps_min`, `eps_max`, `r_leave`, `t_leave_target`), entegrasyon sınırları (`t_max`, `s_max`, `r_blow`), isteğe bağlı bölümler/pencereli çıkış koşulları (`window_min`, `window_max`, `t_min_event`, `require_leave_before_event`), spektral toleranslar (`eig_real_tol`, `eig_imag_tol`, `strict_1d`), Jacobian yönetimi (`jac`, `fd_eps`) ve bir `HeteroclinicRK45Config`'e işaret eden `rk` alanı.
- `HeteroclinicFinderConfig2D`/`HeteroclinicFinderConfigND`: İki dal yapılandırmasını (`trace_u`, `trace_s`) arama davranışıyla (`scan_n`, `max_bisect`, `x_tol`, `gap_tol`, `gap_fac`, `branch_mode`, `sign_u`, `sign_s`, `r_sec`, `r_sec_mult`, `r_sec_min_mult`, artı ND için `tau_min`) ve isteğe bağlı `eq_tol`/`eq_max_iter` geçersiz kılmalarıyla eşleştirir. Bu tam yapılandırma `cfg` üzerinden geçirilir ve mevcut olduğunda basitleştirilmiş kwargs'ı atlar.
- `HeteroclinicPreset`: Bir dal yapılandırmasını, RK ayarlarını ve tarama parametrelerini paketler, böylece her alanı manuel olarak ayarlamak yerine `"fast"`, `"default"` veya `"precise"` (veya veri sınıfını kendiniz başlatarak özel bir ön ayar) isteyebilirsiniz.

Tam bir heteroklinik avı + çizim rutini için `examples/analysis/heteroclinic_finder_tracer.py`'ye bakın.

### Homoklinik araçlar

- `homoclinic_finder(...)`: Aynı eyer (saddle) denge noktasının kararsız ve kararlı manifoldlarının yeniden bağlandığı bir parametre arar. Aynı `preset` adlarını ve basitleştirilmiş geçersiz kılmaları (`window`, `scan_n`, `max_bisect`, `gap_tol`, `x_tol`, `t_max`, `r_blow`, `r_sec`, `t_min_event`) veya tam bir `HomoclinicFinderConfig` kabul eder. Döndürülen `HomoclinicFinderResult`, heteroklinik bulucuyu yansıtır (en yakın vuruşu tanımlayan bir `HomoclinicMissResult` ile).
- `homoclinic_tracer(...)`: İşaretle tanımlanmış tek bir kararsız dalı eyer noktasına geri dönene kadar takip eder; `branches` niteliği `plot.manifold`'a gönderilebilen bir `HomoclinicTraceResult` döndürür. İzleyici, RK45 toleranslarını, ayrılma/dönüş yarıçaplarını ve olay algılama korumalarını ayarlamanıza olanak tanıyan `HomoclinicBranchConfig` kullanır.

#### Yapılandırma veri sınıfları

Bulucu/izleyici çifti, gelişmiş ayarlama için yapılandırılmış veri sınıfları sunar. `homoclinic_finder` bir `cfg: HomoclinicFinderConfig` kabul ederken, `homoclinic_tracer` bir `cfg_u: HomoclinicBranchConfig` alabilir; bu nesneleri sağlamak basitleştirilmiş `preset`, `trace_cfg` ve anahtar kelime argümanlarını atlar.

- `HomoclinicRK45Config`: RK45 parametrelerine (`dt0`, `min_step`, `dt_max`, `atol`, `rtol`, `safety`, `max_steps`) sahiptir.
- `HomoclinicBranchConfig`: Denge iyileştirme (`eq_tol`, `eq_max_iter`, `eq_track_max_dist`), ayrılma-olayı kontrolü (`eps_mode`, `eps`, `eps_min`, `eps_max`, `r_leave`, `t_leave_target`, `r_sec`, `t_min_event`, `require_leave_before_event`), entegrasyon sınırları (`t_max`, `s_max`, `r_blow`), isteğe bağlı pencere kısıtlamaları, spektral eşikler (`eig_real_tol`, `eig_imag_tol`, `strict_1d`), Jacobian yönetimi (`jac`, `fd_eps`) ve bir `HomoclinicRK45Config`'e referans veren `rk` alanını katmanlar.
- `HomoclinicFinderConfig`: Bir `trace` dal yapılandırmasını arama-özel düğmelerle (`scan_n`, `max_bisect`, `gap_tol`, `x_tol`, `branch_mode`, `sign_u`) sarar, böylece hem izleme hem de parametre ikiye bölme (bisection) davranışını kontrol edebilirsiniz.
- `HomoclinicPreset`: Bir dal yapılandırmasını, RK ayarlarını ve tarama toleranslarını paketler, böylece `"fast"`, `"default"` veya `"precise"` doğrudan `preset` argümanı olarak geçirilebilir; bu varsayılanlar yeterince agresif değilse kendi ön ayarınızı da oluşturabilirsiniz.

Ön Ayar Özeti:

| İsim | Açıklama |
| --- | --- |
| `fast` | Keşif için daha gevşek toleranslarla hızlı tarama. |
| `default` | Standart kullanım durumları için hız ve sağlamlık dengesi. |
| `precise` | Zorlu yörüngeler için sıkı toleranslar, daha küçük adımlar ve daha uzun entegrasyonlar. |

Her iki bulucu/izleyici de sonuç `meta` nesnelerine `diag` meta verilerini kaydeder, böylece bir çalışma başarısız olursa veya yalnızca zar zor başarılı olursa ODE adım sayılarını, RK ayarlamalarını ve olay tetikleyicilerini inceleyebilirsiniz.

## Sonraki adımlar

- Hedef denge noktasını bildiğinizde ve kararlı/kararsız dallarını görselleştirmek istediğinizde `trace_manifold_1d_map` veya `trace_manifold_1d_ode` kullanın. `ManifoldTraceResult` sonuçlarını [manifold grafikleri](../plotting/manifold-plot.md) içindeki referans çizimlerle birleştirin.
- Parametre değerlerini avlamak için `examples/analysis/{heteroclinic_,homoclinic_}finder_tracer.py` içindeki heteroklinik/homoklinik bulucu betiklerini çalıştırın, ardından yayına hazır görseller oluşturmak için keşfedilen parametrede yörüngeyi izleyin (trace).