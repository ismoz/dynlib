# Çekim havzası analizi: bilinen ve otomatik yöntemler

Bu sayfa, dynlib’de çekim havzası (basin of attraction) çıkarmak için iki aracı açıklar:

- **`basin_known`**: Önceden tanımladığınız (bilinen) çekicilere göre sınıflandırır (**FixedPoint** ve/veya **ReferenceRun**).
- **`basin_auto`**: Çekicileri otomatik keşfeder ve çekim havzasını **PCR-BM** (Persistent Cell-Recurrence Basin Mapping) ile haritalar.

> Not: Sonuç etiketleri (labels) şu anlamlara gelir:
>
> - `>= 0`: çekici kimliği (attractor ID)
> - `BLOWUP`: NaN/Inf veya `b_max` eşiği ile taşma (divergence)
> - `OUTSIDE`: gözlem bölgesinin dışına kaçış
> - `UNRESOLVED`: sınıflandırılamadı


---

## `basin_known`: Bilinen çekicilere göre sınıflandırma

`basin_known`, başlangıç koşullarını (**IC**) sizin verdiğiniz çekicilere göre etiketler. İki çekici türünü destekler:

- **`FixedPoint`**: sabit nokta. Hızlı yol (fast-path) ile sınıflandırılır: yörünge, sabit noktanın yarıçapı içinde **art arda** `fixed_point_settle_steps` adım kalırsa anında atanır.
- **`ReferenceRun`**: referans koşu. Bu çekiciler için çevrimiçi **PSC** tabanlı bir imza/kanıtlama yaklaşımı kullanılır (gözlem uzayı hücrelerine kuantalama + log-olasılık skorlama). Kaotik ve çevrimsel çekicilerde daha dayanıklıdır.

### Ne zaman tercih edilmeli?

- Çekicileri zaten biliyorsanız (sabit noktalar, belirli periyot çevrimleri, bilinen kaotik çekiciler).
- “Şu çekiciler var, bunlara göre havzayı boya” iş akışı istiyorsanız.

### Parametreler (özet)

Aşağıdaki isimler **API’deki anahtarlar**dır; aynen kullanın:

- `sim: Sim`  
  Simülasyon nesnesi.

- `attractors: Sequence[FixedPoint | ReferenceRun]`  
  Bilinen çekiciler listesi.

- `ic` **veya** `ic_grid`  
  - `ic: np.ndarray`: Doğrudan IC matrisi (`(n_points, n_states)`).
  - `ic_grid: Sequence[int]` + `ic_bounds: Sequence[(min,max)]`: IC ızgarası üretir.

- `observe_vars: Sequence[str|int] | None`  
  Eşleştirme/kuantalama için gözlenen durum değişkenleri. `None` ise varsayılan seçim kullanılır.

- `escape_bounds: Sequence[(min,max)] | None`  
  Kaçış denetimi için sınırlar. Yörünge bu bölgenin dışında art arda belirli sayıda örnek görürse `OUTSIDE` sayılır.

- `max_samples`, `transient_samples`  
  - `max_samples`: Sınıflandırma için en fazla örnek/adım.
  - `transient_samples`: İlk geçiş (transient) örneklerini yok sayma.

- `dt_obs`  
  - ODE modunda zorunlu örnekleme aralığı.
  - map modunda varsayılan olarak modelin `dt`’si kullanılır.

- `tolerance` / `tolerance_absolute`  
  Kod içinde bir **mesafe eşiği** (`dist_threshold`) hesaplamak için kullanılır.  
  `tolerance_absolute` verilirse doğrudan o kullanılır; verilmezse `tolerance * mean(obs_range)` ile türetilir.

- `b_max`, `blowup_vars`  
  `b_max` verilirse seçili değişkenlerde `abs(value) > b_max` taşma sayılır ve `BLOWUP` döner.  
  `b_max=None` ise yalnızca NaN/Inf denetimi ile taşma yakalanır.

- `parallel_mode: {"auto","threads","process","none"}` ve `max_workers`  
  Büyük grid’lerde process tabanlı paralellik devreye girebilir.

- **Coarse-to-fine iyileştirme (refinement)**:
  - `refine: bool`: önce kaba grid, sonra yalnızca sınır bölgelerinde tam çözünürlük.
  - `coarse_factor: int`: kaba grid ölçek oranı.
  - `boundary_dilation: int`: sınır maskesini güvenlik payı için genişletme.

- **Sabit nokta hızlı yolu**:
  - `fixed_point_settle_steps: int`: sabit nokta yarıçapı içinde art arda kaç adım kalınca “yakınsadı” kabul edileceği.

### Çıktı

`BasinResult` döner:

- `labels`: her IC için çekici kimliği veya özel kodlar (`BLOWUP/OUTSIDE/UNRESOLVED`)
- `registry`: `Attractor` listesi (bu fonksiyonda çekici sayısı bilindiği için id’ler 0..n_attr-1)
- `meta`: kullanılan ayarlar ve (varsa) grid/refine metadatası

### Kısa örnek

```python
from dynlib.runtime.sim import Sim
from dynlib.analysis.basin import FixedPoint, ReferenceRun
from dynlib.analysis.basin_known import basin_known
import numpy as np

# IC grid
ic_grid = [400, 400]
ic_bounds = [(-2.0, 2.0), (-2.0, 2.0)]

# Bilinen çekiciler (örnek)
attractors = [
    FixedPoint(label="fp0", ic=[0.0, 0.0], radius=1e-3),
    ReferenceRun(label="attr1", ic=[0.1, 0.1]),
]

result = basin_known(
    sim,
    attractors,
    ic_grid=ic_grid,
    ic_bounds=ic_bounds,
    mode="map",
    max_samples=2000,
    transient_samples=200,
    dt_obs=None,
    refine=True,
    coarse_factor=8,
    boundary_dilation=1,
)
labels_2d = result.labels.reshape(*ic_grid)
```


---

## `basin_auto`: PCR-BM ile otomatik çekim havzası

`basin_auto`, çekicileri **önceden tanımlamanızı gerektirmeden** keşfeder. Yaklaşım:

1. Yörüngeleri gözlem uzayında bir **hücre ızgarasına** kuantalar (`grid_res`, `obs_min/obs_max`).
2. Zaman penceresi içinde ziyaret edilen **benzersiz hücre oranı** küçükse tekrarlılık (recurrence) tespit eder (`window`, `u_th`, `recur_windows`).
3. Tespit edilen parçalardan bir **parmak izi** (fingerprint) çıkarır ve benzerlerini birleştirir (`s_merge`, `merge_downsample`).
4. Yörüngeleri çekicilere **süreklilik/persistans** ile atar: bir çekiciye ait hücre setinde art arda `p_in` isabet (hit) görürse etiketler.

### Online / Offline

- `online=True` (önerilen): Entegrasyon sırasında analiz yapar; tam yörünge saklamaz. Büyük batch’lerde bellek açısından çok daha iyidir.
- `online=False`: Tam yörüngeleri saklayıp sonra işler; hata ayıklama/deney için yararlı ama bellek tüketir. Bu modda `max_memory_bytes` ile koruma devreye girebilir.

### Parametreler (özet)

- `ic` **veya** `ic_grid` (+ `ic_bounds`)  
  IC’leri doğrudan verin ya da uniform grid üretin.

- `observe_vars`  
  Gözlem uzayı değişkenleri (isim veya indeks). Boyut `d`, `grid_res/obs_min/obs_max` ile uyumlu olmalıdır.

- `obs_min`, `obs_max`  
  Gözlem uzayının alt/üst sınırları. `None` ise `ic_bounds`’tan türetilebilir.

- `grid_res`  
  Her gözlem boyutu için hücre sayısı. Toplam hücre sayısı `product(grid_res)`.

- `max_samples`, `transient_samples`  
  - `max_samples`: her IC için en fazla örnek/adım
  - `transient_samples`: tekrarlılık tespiti başlamadan önce yok sayılacak örnek sayısı

- `window`, `u_th`, `recur_windows`  
  Tekrarlılık (recurrence) tespiti ayarları:
  - `window`: kayan pencere boyu
  - `u_th`: `unique_cells/window <= u_th` koşulu sağlanırsa “tekrarlıyor” say
  - `recur_windows`: bu koşulun art arda kaç pencerede sağlanması gerektiği

- `p_in`  
  Persistans eşiği: bir çekici hücre setinde art arda kaç isabetle atanacağı. `0` verilirse persistansla atama kapatılır.

- `s_merge`  
  Parmak izi benzerliği (Jaccard) bu eşiğin üstündeyse çekiciler birleştirilir.

- `merge_downsample`  
  Parmak izi çıkarırken daha kaba bir ızgaraya indirgeme (robust birleştirme için).

- `b_max`, `blowup_vars`, `outside_limit`  
  - `b_max`: taşma eşiği
  - `outside_limit`: art arda kaç örnek `obs_min/obs_max` dışında kalınca `OUTSIDE`

- `parallel_mode`, `max_workers`  
  Büyük batch’lerde process paralelliği kullanılabilir (`parallel_mode="process"` veya `"auto"`).

- `post_detect_samples`  
  Tekrarlılık tespitinden sonra kanıt segmentine eklenecek ek örnek sayısı.

- `refine_unresolved` (online mod)  
  İlk geçişte `UNRESOLVED` kalan noktaları, çekici kaydı (registry) tamamlandıktan sonra yeniden persistans taramasıyla atamayı dener.

- `online_max_attr`, `online_max_cells`  
  Online modda persistans taraması için belleği sınırlamak üzere:
  - izlenecek maksimum çekici sayısı
  - çekici başına saklanacak maksimum hücre sayısı

### Çıktı

`BasinResult` döner:

- `labels`: her IC için çekici kimliği veya `BLOWUP/OUTSIDE/UNRESOLVED`
- `registry`: keşfedilen çekiciler (parmak izi + hücre setleri)
- `meta`: algoritma parametreleri ve yürütme metadatası

### Kısa örnek

```python
from dynlib.analysis.basin_auto import basin_auto
import numpy as np

result = basin_auto(
    sim,
    ic_grid=[300, 300],
    ic_bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    mode="map",
    observe_vars=("x", "y"),
    obs_min=[-2.0, -2.0],
    obs_max=[2.0, 2.0],
    grid_res=64,
    max_samples=1500,
    transient_samples=100,
    window=64,
    u_th=0.6,
    recur_windows=3,
    p_in=8,
    online=True,
)

labels_2d = result.labels.reshape(300, 300)
```

### Ayar önerileri (pratik)

- **Basit periyodik çekiciler**: `u_th` daha küçük (0.3–0.5), `window` daha küçük (32–64).
- **Kaotik çekiciler**: `u_th` daha büyük (0.6–0.8), `window` daha büyük (64–256).
- **Çok `UNRESOLVED`**: `max_samples` artırın; `u_th` düşürün; `window` artırın.
- **Çekiciler gereksiz birleşiyor**: `s_merge` artırın veya `merge_downsample` düşürün.
- **Birleşmesi gerekenler birleşmiyor**: `s_merge` düşürün veya `merge_downsample` artırın.
