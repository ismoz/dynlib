# Stepper'lar

`dynlib`, simülasyonu bir adım ilerletmek için stepper'ları kullanır. Bunlar çoğunlukla integratörlerdir; ancak ayrık haritaları (discrete maps) ilerletmek için de kullanıldıklarından, sadece "integratör" veya "çözücü" (solver) olarak adlandırılmazlar. Ayrıca, ODE integratörleri gelecekteki farklı dinamik sistem türleri için uygun olmayabilir. Probleminizin sınıfına (ODE vs map), zaman kontrol stratejisine (sabit vs uyarlamalı) ve sayısal şemasına (açık, kapalı, ayrıştırma) uygun olan stepper'ı seçin. Model varsayılanını `build(..., stepper="rk4")` üzerinden veya uygulama kodundaki hızlı yol için `setup(..., stepper="rk4")` ile geçersiz kılabilirsiniz. Derlenmiş her `Model` veya `Sim` yüzeyi `model.stepper_name` değerini dışa açar; böylece derlemeden sonra hangi integratörün seçildiğini teyit edebilirsiniz.

## Bir Stepper Seçmek

Bir simülasyonu çalıştırmadan önce her zaman kontrol etmeniz gereken üç ana eksen vardır:

- **Tür** (`Kind = "ode" | "map"`): Stepper'ın matematiksel doğasını tanımlar. ODE stepper'ları bir RHS (sağ taraf) `f(t, y)` bekler ve bunu `dt` ile çarpar; `map` stepper'ı ise derlenmiş fonksiyonu, doğrudan bir sonraki durumu döndüren ayrık bir güncelleme olarak ele alır (`dt` burada sadece bir etikettir).
- **Zaman Kontrolü** (`TimeCtrl = "fixed" | "adaptive"`): Integratörün sabit bir `dt` ile mi ilerleyeceğini, yoksa adımı dahili olarak yeniden deneyip boyutlandıracağını mı belirler. Uyarlamalı (adaptive) stepper'lar (RK45, BDF2a, TR-BDF2a) `atol/rtol` tolerans kontrollerini sunarken, sabit adımlı (fixed) stepper'lar sürücü `Sim.run()` argümanlarına güvenir.
- **Şema** (`Scheme = "explicit" | "implicit" | "splitting"`): Yöntemin cebirsel yapısıdır. Açık (explicit) stepper'larda doğrusal olmayan çözümler (nonlinear solves) yoktur; oysa kapalı (implicit) stepper'lar (SDIRK2, BDF2, BDF2a, TR-BDF2a) Newton iterasyonlarını çağırır ve genellikle isteğe bağlı analitik Jakobiyenleri destekler. Ayrıştırma (splitting) şemaları gelecekte eklenecektir.

Bu eksenlerin kombinasyonu ve her `StepperMeta` içindeki `family`/`order` (aile/mertebe) üst verileri, size fiziksel olarak ne olup bittiğine dair özlü bir görünüm sunar. Jakobiyenlere, yoğun çıktıya (dense output) veya Lyapunov analizi için varyasyonel adımlamaya ihtiyacınız varsa, `StepperCaps` bloğuna bakın (dokümanlarda `dense_output`, `jacobian`, `jit_capable`, `requires_scipy` ve `variational_stepping` bayraklarını sunuyoruz).

## Mevcut Stepper'lar

| İsim      | Tür  | Zaman Kontrolü | Şema | Mertebe | Önemli Notlar |
| :---      | :--- | :---           | :---   | :---  | :---          |
| `map`     | map  | sabit (fixed)  | açık   | 1     | Ayrık yinelemeler (`F(t, y)` sonraki durumu döner). `dt` sadece zamanı etiketler. |
| `euler`   | ode  | sabit (fixed)  | açık   | 1     | İleri Euler, minimum çalışma alanı, varyasyonel stepping uyumlu. |
| `rk2`     | ode  | sabit (fixed)  | açık   | 2     | Basit 2 aşamalı güncelleme ve varyasyonel destekli açık orta nokta (RK2). |
| `rk4`     | ode  | sabit (fixed)  | açık   | 4     | Klasik Runge–Kutta 4. mertebe; takma adları: `rk4_classic`, `classical_rk4`. |
| `rk45`    | ode  | uyarlamalı     | açık   | 5¹    | 4. mertebe hata tahminli Dormand–Prince RK45. |
| `ab2`     | ode  | sabit (fixed)  | açık   | 2     | Heun başlangıçlı Adams–Bashforth 2 çok adımlı yöntemi; türev geçmişini tutar. |
| `ab3`     | ode  | sabit (fixed)  | açık   | 3     | İki adımlı başlangıca sahip Adams–Bashforth 3. |
| `sdirk2`  | ode  | sabit (fixed)  | kapalı | 2     | Alexander SDIRK2 (γ = (2−√2)/2), sert (stiff) problemlerde hassas ama Jakobiyen gerektirir. |
| `bdf2`    | ode  | sabit (fixed)  | kapalı | 2     | Newton çözücülü kapalı BDF2; isteğe bağlı harici Jakobiyenleri kabul eder. |
| `bdf2a`   | ode  | uyarlamalı     | kapalı | 2     | Hata tahminli değişken adımlı BDF2. |
| `tr-bdf2a`| ode  | uyarlamalı     | kapalı | 2     | `bdf2a` ile aynı ayarlara sahip TR-BDF2 uyarlamalı integratör (L-kararlı). |

¹ Gömülü mertebe: 4 (hata tahmini). Uyarlamalı stepper'lar `dt` değerini dahili olarak geçersiz kılar ancak çalıştırıcı için yine de `dt_next` rapor eder.

Her temel stepper ismi bir kez kaydedilir; `forward_euler`, `rk4_classic`, `trbdf2a` ve `sdirk2_jit` gibi takma adlar (aliases) otomatik olarak aynı spesifikasyona eşlenir. Yapılandırmaları paylaşırken sürprizlerle karşılaşmamak için temel (canonical) ismi kullanın.

Haritalar (maps) için `stepper=map` tanımlamanız gerekmez. ODE modelleri için varsayılan stepper `rk4`'tür.

## Stepper Kaydı ve Keşfi

Stepper kayıt defteri hem kullanıcı hem de geliştirici odaklıdır. Stepper modülleri içe aktarıldığında otomatik olarak doldurulur, ancak özel bir yönteme ihtiyacınız varsa `dynlib.register()` ile kendi tanımlarınızı da kaydedebilirsiniz.

```python
from dynlib import list_steppers, select_steppers, get_stepper

print(list_steppers(kind="ode"))
infos = select_steppers(scheme="implicit", stiff=True, jit_capable=True)
print([info.name for info in infos])
spec = get_stepper("rk45")
print(spec.meta.order, spec.meta.aliases)
```

`list_steppers()` sıralanmış temel isimleri döndürür ve `select_steppers()` ile aynı anahtar kelime filtrelerini (kind, scheme, stiff, jit_capable vb.) kabul eder. `select_steppers()` fonksiyonu `StepperInfo` örneklerini döndürür. Ayrıca ince ayarlı keşif için `name_pattern` veya özel bir `predicate` (koşul) geçebilirsiniz (örneğin varyasyonel stepping veya yoğun çıktı destekleyen seçenekleri aramak için).

CLI, Python API'sini yansıtır: `dynlib steppers list` komutu aynı temel isimleri yazdırır ve mevcut bayraklar yukarıdaki filtreleri yansıtarak çıktıyı daraltmanıza olanak tanır.

### Stepper Üst Veri (Metadata) Alanları

- `name`: Temel stepper ismi.
- `kind`: `ode` veya `map`.
- `time_control`: `fixed` (sabit) veya `adaptive` (uyarlamalı).
- `scheme`: `explicit` (açık), `implicit` (kapalı) veya `splitting` (ayrıştırma).
- `geometry`: Geometri duyarlı yöntemler için ayrılmış set (şu an yerleşik stepper'larda boştur).
- `family`: `runge-kutta`, `adams-bashforth`, `bdf`, `dirk`, `tr-bdf2` veya `iter` gibi sınıflandırmalar.
- `order`, `embedded_order`: Birincil ve gömülü doğruluğu tanımlar.
- `stiff`: Yöntemin sert (stiff) problemler için uygun olup olmadığını belirtir.
- `aliases`: Aynı spesifikasyona eşlenen diğer isimler.
- `caps`: Aşağıdaki yetenek bayrakları.

### Stepper Yetenek Bayrakları (`StepperCaps`)

- `dense_output`: Sürekli interpolasyon / yoğun çıktı desteği.
- `jacobian`: Stepper'ın harici Jakobiyenleri nasıl tükettiğini açıklar (`none`, `internal`, `optional`, `required`).
- `jit_capable`: Tüm yerleşik stepper'lar için doğrudur; yöntem dış bağımlılıklara dayanıyorsa yanlıştır.
- `requires_scipy`: SciPy gerekliyse doğrudur.
- `variational_stepping`: `emit_step_with_variational()` desteğini belirtir (Lyapunov analizinde kullanılır).

## Stepper Workspace

Stepper workspace'i, çalışma zamanı üst verilerinin yanında yaşayan, stepper'a özel bir karalama alanıdır. Her stepper, tek bir adım sırasında ihtiyaç duyduğu dizileri tanımlayan bir `NamedTuple` belirler. Workspace, her simülasyon için bir kez stepper'ın `make_workspace(n_state, dtype)` kancası üzerinden tahsis edilir ve her adımda yeniden kullanılır.

### Temel Özellikler

- **Sahiplik**: Her stepper spesifikasyonuna özeldir ve ABI'ye `stepper_ws` olarak aktarılır.
- **Ömür**: `Sim` yok edilene veya workspace belleği açıkça serbest bırakılana kadar kalıcıdır.
- **İçerik**: NumPy dizileri (aşama tamponları, türev geçmişleri, Jakobiyen karalama alanı vb.).
- **Tahsis**: Stepper'ın `workspace_type()` ve `make_workspace()` metodları tarafından yönetilir.
- **Tür**: Alanlara dizin yerine isimle erişilmesi için bir `NamedTuple`'dır.

### Örnek Workspace Düzenleri

#### Euler Workspace
```python
class Workspace(NamedTuple):
    dy: np.ndarray
    kv: np.ndarray
```

#### RK4 Workspace
```python
class Workspace(NamedTuple):
    y_stage: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    v_stage: np.ndarray
    kv1: np.ndarray
    kv2: np.ndarray
    kv3: np.ndarray
    kv4: np.ndarray
```

### Neden Her Stepper RHS Dizilerine İhtiyaç Duyar?

Şema ne olursa olsun, stepper bir teklif oluşturabilmek için `f(t, y)`'nin bir veya daha fazla değerlendirmesini saklamalıdır. Workspace bu RHS tamponlarını tutar. Her yöntem türev vektörlerini kendi yuvalarına yazar, ardından `y_prop`, `t_prop`, `dt_next` ve `err_est` değerlerini birleştirir.

## Runtime Workspace

Runtime workspace'i, stepper'ın karalama alanından ayrı olarak, gecikme tamponlarını (lag buffers) ve diğer DSL makinelerinin durumunu yönetir.

## Stepper ABI

Derlenmiş stepper çağrılabilir yapısı (callable), runner ve sonuç altyapısının genel kalabilmesi için sabit bir ABI izler. İmzası şöyledir:

```python
status = stepper(
    t: float,
    dt: float,
    y_curr: float[:],
    rhs,
    params: float[:] | int[:],
    runtime_ws,
    stepper_ws,
    stepper_config: float64[:],
    y_prop: float[:],
    t_prop: float[:],
    dt_next: float[:],
    err_est: float[:],
) -> int32
```

- `rhs`: Stepper'ın tekrar tekrar çağırdığı derlenmiş RHS fonksiyonu.
- `runtime_ws`: Gecikme tamponları ve üst veriler için paylaşılan çalışma zamanı alanı.
- `stepper_ws`: Yukarıda açıklanan aktif stepper workspace'i.
- `stepper_config`: Stepper'ın dataclass'ından paketlenmiş ayar dizisi.
- `y_prop`, `t_prop`, `dt_next`, `err_est`: Runner'ın her çağrıdan sonra tükettiği çıktı tamponları.

## Varyasyonel Stepping Genel Bakış

Bazı stepper'lar, Lyapunov, duyarlılık veya varyasyonel analizler için yararlı olan birleşik bir "durum + teğet" (state + tangent) entegrasyon yolunu destekler. `StepperCaps(variational_stepping=True)` ayarına sahip `StepperSpec` uygulamalarına bakın. Bu tesisler, teğet tamponlarını durum workspace'inin yanında tutar ve teğet durumunun `y_prop` ile senkronize kalmasını sağlar.

## Stepper'ları Genişletme (Geliştirici Rehberi)

Özel bir integratör veya haritaya (map) ihtiyaç duyan katkıcılar için `dynlib` stepper yapısı modüler tasarlanmıştır. Yeni bir stepper spesifikasyonu sadece `StepperSpec` protokolünü uygular, kendini kaydeder ve isteğe bağlı olarak `ConfigMixin` aracılığıyla çalışma zamanı ayarlarını dışa açar.

1. **Üst verileri ve yetenekleri tanımlayın**: Yeni yöntemi açıklayan bir `StepperMeta` ve `StepperCaps` oluşturun.
2. **Spesifikasyonu uygulayın**: `emit` içinde, `stepper_config` (paketlenmiş float dizisi) üzerinden çalışma zamanı geçersiz kılmalarını uygulayın.
3. **Workspace yardımcılarını hazırlayın**: Workspace düzenini NumPy dizileri içeren bir `NamedTuple` olarak tanımlayın.
4. **Spec'i kaydedin**: Stepper'ı genel kayıt defterine `dynlib.register()` ile ekleyin.