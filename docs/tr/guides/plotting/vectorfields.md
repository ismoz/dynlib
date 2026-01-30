# Vektör Alanları ve Vektör Alanı Animasyonları

Vektör alanları, dinamik sistemlerdeki değişimin yönünü ve büyüklüğünü görselleştirir. Dynlib'de vektör alanları, sistemin sağ taraf (RHS) denklemlerini 2B bir nokta ızgarası üzerinde değerlendirerek ve her bir noktadan yörüngelerin nasıl evrileceğini göstererek hesaplanır.

Bu rehberdaki çoğu kod parçası `from dynlib import build, plot` yapıldığını varsayar. Taramalar (sweeps) veya animasyonlar için sayısal dizilere ihtiyacımız olduğunda, `numpy`'ın da `import numpy as np` şeklinde içe aktarıldığını göreceksiniz.

## Temel Vektör Alanı Çizimi

Vektör alanlarını çizmek için temel fonksiyon `plot.vectorfield()` fonksiyonudur. Modelinizin denklemlerini bir ızgara üzerinde değerlendirir ve ortaya çıkan vektörleri görüntüler.

### Basit Örnek

```python
from dynlib import build, plot

# Basit bir 2B sistem tanımla
model_uri = """
inline:
[model]
type = "ode"

[states]
x = 0.0
y = 0.0

[params]
a = 1.0
b = -1.0

[equations.rhs]
x = "a * x + y"
y = "b * x + y"
"""

# setup() ile oluşturulan bir Sim nesnesi de geçirebilirsiniz
model = build(model_uri)
plot.theme.use("notebook")

# Vektör alanını çiz
plot.vectorfield(
    model,
    xlim=(-2, 2),
    ylim=(-2, 2),
    grid=(25, 25)
)

plot.export.show()
```

`build()` (veya `setup()`) fonksiyonuna satır içi (inline) DSL geçirirken, dynlib'in içeriği bir yol yerine gömülü model tanımı olarak ele alması için dizeyi yukarıda gösterildiği gibi `inline:` ile başlatın.

## Vektör Alanı Seçenekleri

### Izgara ve Limitler

- `xlim`, `ylim`: Çizim sınırlarını belirten demetler (varsayılan: `(-1, 1)`)
- `grid`: Izgara çözünürlüğünü belirten `(nx, ny)` demeti (varsayılan: `(20, 20)`)

Yüksek ızgara değerleri daha pürüzsüz, daha detaylı çizimler sağlar ancak hesaplanması daha uzun sürer.

### Değişken Seçimi

2'den fazla değişkene sahip sistemler için, hangilerinin çizileceğini belirtin:

```python
# 3B Lorenz sistemi için
plot.vectorfield(
    model,
    vars=("x", "y"),  # x'e karşı y çiz
    fixed={"z": 10.0},  # z'yi 10'da sabitle
    xlim=(-20, 20),
    ylim=(-30, 30)
)
```

### Vektör Normalizasyonu

- `normalize=True`: Tüm vektörleri birim uzunluğa ölçekler, yalnızca yönü gösterir.
- `normalize=False` (varsayılan): Gerçek büyüklükleri gösterir.

```python
# Normalleştirilmiş ile büyüklüğü koruyan karşılaştırması
plot.vectorfield(model, normalize=True)   # Akış yönlerini gösterir
plot.vectorfield(model, normalize=False)  # Akış hızlarını gösterir
```

### Renklendirme Seçenekleri

#### Tek Renk
```python
plot.vectorfield(model, color="blue")
```

`color` argümanı doğrudan Matplotlib'e akar, bu nedenle tutarlı bir palet için herhangi bir isimlendirilmiş renk, hex dizesi veya RGBA demeti kullanabilirsiniz.

#### Hıza Dayalı Renklendirme
Vektörleri büyüklüklerine göre renklendirin:

```python
plot.vectorfield(
    model,
    speed_color=True,
    speed_cmap="plasma",
    normalize=False  # Hız renklendirmesi en iyi gerçek büyüklüklerle çalışır
)
```

Farklı çizimleri aynı ölçekte karşılaştırabilmek için tam aralığı manuel olarak sabitlemek üzere `speed_norm` geçirin (veya `share_speed_norm=True` ile bir taramanın paylaşılan normu hesaplamasını sağlayın).

### Çizim Modları

- `mode="quiver"` (varsayılan): Ok/sadak (quiver) çizimi
- `mode="stream"`: Matplotlib'in streamplot'unu kullanan akış çizgisi (streamline) çizimi

```python
# Akış çizgileri, yoğun akışlar için daha pürüzsüz olabilir
plot.vectorfield(model, mode="stream", speed_color=True)
```

### Nullcline'lar (Sıfır Eğrileri)

Nullcline'lar, sistemin x veya y yönlerinde sıfır hıza sahip olduğu yerleri gösterir:

```python
plot.vectorfield(
    model,
    nullclines=True,
    nullcline_style={"colors": ["red", "blue"], "linewidths": 1.5}
)
```

Doğruluk için nullcline'lar varsayılan olarak daha yoğun bir ızgara üzerinde hesaplanır.

Daha ince konturlara veya ana ızgaraya göre yeniden boyutlandırmaya ihtiyacınız olduğunda `nullcline_grid` kullanın.

## Etkileşimli Özellikler

`plot.vectorfield`, bir `VectorFieldHandle` döndürür; böylece okları çizen aynı çağrı, size güncelleyebileceğiniz, simüle edebileceğiniz veya temizleyebileceğiniz programatik bir tutamaç (handle) da verir. Aşağıda açıklanan tıklama/dokunma geri çağrımlarına (callbacks) bağlanmak için `interactive=True` geçirin veya çizimi yeniden oluşturmadan yeni parametrelerle/sabit durumlarla yeniden çizmek için `handle.update()` çağırın.

Yörüngeleri keşfetmek için etkileşimli çizimi etkinleştirin:

```python
handle = plot.vectorfield(
    model,
    interactive=True,
    T=10.0,  # Yörünge süresi
    trajectory_style={"color": "red", "linewidth": 2}
)
```

**Etkileşimli Kontroller:**
- O noktadan bir yörünge başlatmak için çizim üzerinde herhangi bir yere **Tıklayın**
- Nullcline'ları açmak/kapatmak için **N** tuşuna basın
- Çizilen yörüngeleri temizlemek için **C** tuşuna basın
`handle.clear_trajectories()` de programatik olarak sıfırlamak isterseniz toplanan yolları kaldırır.

## Parametre Taramaları (Sweeps)

Vektör alanlarını parametre değerleri arasında karşılaştırmak için `plot.vectorfield_sweep()` kullanın. Bu fonksiyon `.handles`, `.axes` ve `.colorbar` içeren bir `VectorFieldSweep` döndürür; böylece çizimden sonra bireysel facet'leri ayarlayabilir veya renklendirme için kullanılan paylaşılan `speed_norm` değerini alabilirsiniz. Basit bir 1D tarama için `param`/`values` geçirin veya parametreler ve sabit durumlar için özel geçersiz kılmalara ihtiyacınız olduğunda `sweep` eşlemesini sağlayın; `target` argümanı her tarama `value`sunun params (varsayılan) mı yoksa sabit durumları (fixed states) mı düzenleyeceğini seçer.

```python
plot.vectorfield_sweep(
    model,
    param="a",  # Değiştirilecek parametre
    values=[-1.0, 0.0, 1.0, 2.0],  # Test edilecek değerler
    xlim=(-3, 3),
    ylim=(-3, 3),
    cols=2,  # Izgarada 2 sütun
    normalize=True,
    facet_titles="a = {value:.1f}"  # Özel başlıklar
)

```

Paylaşılan normalizasyon (`share_speed_norm=True`), renklendirmenin tüm facet'lerde tutarlı kalmasını sağlar ve `add_colorbar=True`, mevcut olduğunda hız renklendirmesi için tek bir lejant çizer.

## Vektör Alanı Animasyonları

`plot.vectorfield_animate()` kullanarak vektör alanlarının parametrelerle nasıl değiştiğini canlandırın:

```python
import numpy as np

# Parametre değişikliklerini canlandır
anim = plot.vectorfield_animate(
    model,
    param="a",
    values=np.linspace(-2, 2, 100),  # 100 kare
    fps=15,
    title_func=lambda v, idx: f"Parametre a = {v:.2f}",
    normalize=True,
    speed_color=True
)

# Animasyonu kaydet
anim.animation.save("vectorfield_animation.gif", writer="pillow")
```

`plot.vectorfield_animate()`, hem canlı `VectorFieldHandle`'ı (erişilebilir `.handle` aracılığıyla) hem de alttaki `matplotlib.animation.FuncAnimation` nesnesini tutan bir `VectorFieldAnimation` döndürür, böylece tutamacı manuel olarak güncelleyebilir veya animasyonu daha sonra kaydedebilirsiniz. `frames`, `values` (bir `param` ile) veya `duration`/`fps` sağlayın ve her kare için özel geçersiz kılmalara ihtiyacınız olduğunda `params_func`, `fixed_func` veya `title_func` kullanın.

### Animasyon Seçenekleri

- `fps`: Saniyedeki kare sayısı (varsayılan: 15)
- `interval`: Kareler arasındaki milisaniye (fps alternatifi)
- `title_func`: Her kare için başlık üretme fonksiyonu
- `repeat`: Animasyonun döngü yapıp yapmayacağı (varsayılan: True)
- `blit`: Daha akıcı animasyon için blitting kullan (tüm arka uçlarda çalışmayabilir)

## İleri Seviye Kullanım

### Çizimleri Dinamik Olarak Güncelleme

`vectorfield()` fonksiyonu, dinamik güncellemelere izin veren bir `VectorFieldHandle` döndürür:

```python
handle = plot.vectorfield(model, params={"a": 1.0})

# Yeniden çizim yapmadan parametreleri güncelle
handle.update(params={"a": 2.0})

# Sabit değerleri güncelle
handle.update(fixed={"z": 15.0})
```

### Özel Değerlendirme

Düşük seviyeli kontrol için, ham vektör verilerini almak üzere `plot.eval_vectorfield()` kullanın:

```python
X, Y, U, V = plot.eval_vectorfield(
    model,
    xlim=(-2, 2),
    ylim=(-2, 2),
    grid=(50, 50),
    normalize=True
)

# Doğrudan matplotlib ile kullan
import matplotlib.pyplot as plt
plt.quiver(X, Y, U, V)
```

Büyüklük ızgarasına (örneğin, bir renk haritası ile renklendirmek veya normalleştirilmiş ile normalleştirilmemiş hızları karşılaştırmak için) ihtiyacınız olduğunda `return_speed=True` geçirin.

### Yüksek Boyutlu Sistemler

2 boyuttan fazla sisteme sahip sistemler için, 2B dilimlere izdüşüm yapın:

```python
# Lorenz sistemi: z sabitken x-y düzlemi
plot.vectorfield(
    lorenz_model,
    vars=("x", "y"),
    fixed={"z": 25.0}
)

# Aynı sistem: x sabitken y-z düzlemi
plot.vectorfield(
    lorenz_model,
    vars=("y", "z"),
    fixed={"x": 0.0}
)
```

Yüksek boyutlu sistemleri dilimlerken, değerlendirmenin istenen düzlem içinde kalması için diğer her durumun `fixed` ile sabitlendiğinden emin olun.

## Performans Hususları

- **Izgara boyutu**: Daha büyük ızgaralar daha iyi çözünürlük sağlar ancak hesaplama daha yavaştır.
- **Normalizasyon**: Normalleştirilmiş çizimler daha hızlı hesaplanır (büyüklük hesaplaması yoktur).
- **Nullcline'lar**: Ayrı bir ızgarada hesaplanır; yoğunluğu kontrol etmek için `nullcline_grid` kullanın.
- **JIT derleme**: Aynı modelin tekrarlanan değerlendirmeleri için `jit=True` ayarlayın.
- **Disk önbellekleme**: Çalıştırmalar arasında derlenmiş yapıları yeniden kullanmak için bir URI'den oluştururken `disk_cache=True` geçirin.

## Yaygın Desenler

### Yörüngelerle Faz Portresi

```python
ax = plot.fig.single()
handle = plot.vectorfield(
    model,
    ax=ax,
    normalize=True,
    nullclines=True
)

# Belirli yörüngeler ekle
from dynlib import Sim
sim = Sim(model)
sim.run(t_end=20.0, state_ic=[1.0, 0.0])
plot.series(sim, ax=ax, vars=("x", "y"))
```

### Bifurkasyon Analizi Kurulumu

```python
# Tarama aralığını numpy ile oluştur
import numpy as np

# Parametreyi tara ve vektör alanı yapısındaki nitel değişiklikleri gözlemle
plot.vectorfield_sweep(
    model,
    param="r",
    values=np.linspace(0, 1, 9),
    normalize=True,
    speed_color=True,
    title="Vektör alanı yapısında bifurkasyon"
)
```

### Özel Parametre Fonksiyonları ile Animasyon

```python
# Parametre güncellemeleri salınımlar için NumPy'ı yeniden kullanabilir
import numpy as np

# Salınan parametre
def param_func(frame_idx):
    return {"a": 2.0 * np.sin(2 * np.pi * frame_idx / 50)}

plot.vectorfield_animate(
    model,
    frames=50,
    params_func=param_func,
    fps=10
)
```