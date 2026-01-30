# Çatallanma (Bifurcation) Diyagramları

Çatallanma diyagramları, dinamik bir sistemin uzun vadeli davranışının bir parametre değiştirildiğinde nasıl değiştiğini görselleştirir. Dynlib, çatallanma analizi için tipik olan yoğun nokta bulutları için optimize edilmiş, saçılım (scatter) tarzı çizimler oluşturmak için `bifurcation_diagram()` fonksiyonunu sağlar.

## Temel Kullanım

`bifurcation_diagram()` fonksiyonu çatallanma verilerini iki formatta kabul eder:

1. Bir `BifurcationExtractor`/`BifurcationResult` (genellikle `SweepResult.bifurcation()` tarafından döndürülür)
2. `(parametre_değerleri, durum_değerleri)` dizilerinden oluşan bir demet (tuple)

Ayıklayıcı yardımcıları ve ham diziler hakkında daha fazla bilgi için aşağıdaki **Girdi Veri Formatları** bölümüne bakın.

```python
from dynlib.plot import bifurcation_diagram, theme, fig, export

# Analizden elde edilmiş çatallanma verileriniz olduğu varsayılıyor
# result = sweep_result.bifurcation("x")  # bir ayıklayıcı (extractor) döndürür (varsayılan .all())

# Çizimi oluştur
ax = fig.single(size=(10, 6))
bifurcation_diagram(
    result,  # BifurcationResult veya (p, y) demeti
    xlabel="r",
    ylabel="x*",
    title="Çatallanma Diyagramı",
    ax=ax
)

export.show()
```

## Girdi Veri Formatları

### Çatallanma ayıklayıcıları ve sonuçları

`SweepResult.bifurcation("x")` bir `BifurcationExtractor` döndürür; bu nesne `BifurcationResult` ile aynı ince arayüzü uygular (`.p`, `.y`, `.param_name`, `.meta` ve `.mode` özelliklerini dışa açar). Ayıklayıcıyı doğrudan `bifurcation_diagram()` fonksiyonuna iletebilir veya somut bir `BifurcationResult` elde etmek için `.all()`, `.tail()`, `.extrema()` veya `.final()` gibi yardımcı yöntemleri çağırabilirsiniz.

```python
from dynlib.analysis.sweep import traj_sweep

sweep_result = traj_sweep(sim, param="r", values=r_values, record_vars=["x"], ...)
result = sweep_result.bifurcation("x")             # ayıklayıcı varsayılan olarak "all" modundadır
# result = result.tail(50)                         # isteğe bağlı: son 50 noktaya odaklan

bifurcation_diagram(result)  # xlabel="r", ylabel="x", başlık moda göre belirlenir
```

`bifurcation_diagram()` şunları otomatik olarak çeker:
- `result.p` üzerinden parametre değerleri
- `result.y` üzerinden durum değerleri
- `result.param_name` ve `result.meta["var"]` üzerinden eksen etiketleri (meta veriler mevcut olduğunda)
- `result.mode` üzerinden başlık (örneğin, `"all"`, `"tail"`, `"extrema"`)

### Ham Diziler

Özel veriler veya harici çatallanma hesaplamaları için:

```python
import numpy as np

# Ham parametre ve durum değeri dizileri
r_values = np.array([...])  # parametre değerleri
x_values = np.array([...])  # karşılık gelen durum değerleri

bifurcation_diagram(
    (r_values, x_values),
    xlabel="r",
    ylabel="x*",
    title="Özel Çatallanma Verisi"
)
```

## Çizim Özelleştirme

### Stillendirme Seçenekleri

Çatallanma diyagramları, optimize edilmiş varsayılanlarla saçılım tarzı çizimi kullanır:

```python
bifurcation_diagram(
    result,
    color="blue",           # İşaretçi rengi
    marker=",",             # Piksel işaretçisi (varsayılan)
    ms=0.5,                 # İşaretçi boyutu (piksel işaretçileri için yok sayılır)
    alpha=0.5,              # Saydamlık (varsayılan)
    label="Lojistik Harita" # Lejant etiketi
)
```

Farklı görsel stiller için varsayılanları geçersiz kılın:

```python
# Daha büyük, daha görünür işaretçiler
bifurcation_diagram(
    result,
    marker=".",
    ms=1.0,
    alpha=1.0,
    color="darkred"
)
```

### Eksen Kontrolü

Eksen sınırlarını ve etiketlerini ayarlayın:

```python
bifurcation_diagram(
    result,
    xlim=(2.5, 4.0),       # Parametre aralığı
    ylim=(0, 1),            # Durum aralığı
    xlabel="r",             # Parametre eksen etiketi
    ylabel="x*",            # Durum eksen etiketi
    title="Lojistik Harita Çatallanmaları"
)
```

### Yazı Tipi Boyutları ve Düzen

Metin görünümünü özelleştirin:

```python
bifurcation_diagram(
    result,
    xlabel_fs=12,           # X ekseni etiketi yazı tipi boyutu
    ylabel_fs=12,           # Y ekseni etiketi yazı tipi boyutu
    title_fs=14,            # Başlık yazı tipi boyutu
    xtick_fs=10,            # X ekseni tik yazı tipi boyutu
    ytick_fs=10             # Y ekseni tik yazı tipi boyutu
)
```

## Ek Açıklamalar ve Vurgulamalar

### Dikey Çizgiler

Önemli parametre değerlerini işaretlemek için dikey çizgiler ekleyin:

```python
bifurcation_diagram(
    result,
    vlines=[
        3.0,                                    # r=3 noktasında basit çizgi
        (3.449, "Periyot-4 çatallanması"),      # Etiketli çizgi
        (3.5699, "Feigenbaum noktası")          # Başka bir etiketli çizgi
    ],
    vlines_color="red",
    vlines_kwargs={
        "linestyle": "--",
        "alpha": 0.7,
        "linewidth": 1
    }
)
```

### Gelişmiş Stillendirme

Dikey çizgi görünümünü kontrol edin:

```python
bifurcation_diagram(
    result,
    vlines=[(3.0, "r=3"), (3.57, "Kaos başlangıcı")],
    vlines_kwargs={
        "linestyle": ":",
        "alpha": 0.5,
        "label_rotation": 90,      # Etiketleri döndür
        "label_position": "top"    # Etiketleri yukarı/aşağı konumlandır
    }
)
```

## Tam Örnek

İşte birden fazla özelleştirme seçeneğini gösteren kapsamlı bir örnek:

```python
from dynlib.plot import bifurcation_diagram, theme, fig, export

# Temayı yapılandır
theme.use("notebook")
theme.update(grid=True)

# Şekil oluştur
ax = fig.single(size=(12, 8))

# Tam özelleştirme ile çiz
bifurcation_diagram(
    result,
    color="black",
    alpha=0.8,
    xlim=(2.5, 4.0),
    ylim=(0, 1),
    xlabel="r",
    ylabel="x*",
    title="Lojistik Harita: Periyot İkiye Katlama Kaskadı",
    xlabel_fs=14,
    ylabel_fs=14,
    title_fs=16,
    vlines=[
        (3.0, "Periyot-2"),
        (3.449, "Periyot-4"),
        (3.5699, "Feigenbaum noktası")
    ],
    vlines_kwargs={
        "color": "red",
        "linestyle": "--",
        "alpha": 0.6,
        "linewidth": 1.5
    },
    ax=ax
)

export.show()
```

## Etkili Çatallanma Çizimleri İçin İpuçları

1. **Çözünürlük**: Pürüzsüz diyagramlar için yüksek çözünürlüklü parametre taramaları (10.000+ nokta) kullanın.
2. **Geçici Rejimler (Transients)**: Çekerlere ulaşmak için yeterli geçici sürenin (transient time) tanındığından emin olun.
3. **İşaretçiler**: Piksel işaretçileri (`,`) yoğun veriler için iyi çalışır; seyrek veriler için daha büyük işaretçiler kullanın.
4. **Alfa (Alpha)**: Düşük alfa değerleri, yoğun bölgelerdeki nokta yoğunluğunu görselleştirmeye yardımcı olur.
5. **Ek Açıklamalar**: Çatallanma noktalarını ve geçişleri vurgulamak için dikey çizgiler kullanın.
6. **Yakınlaştırma (Zooming)**: Karmaşık kaskadlar için, yakınlaştırılmış bölgeleri ayrı ayrı çizmeyi düşünün.

`bifurcation_diagram()` fonksiyonu, dynlib'in analiz iş akışıyla sorunsuz bir şekilde bütünleşerek yörünge taramalarından görsel çatallanma diyagramlarına dönüşümü otomatik olarak halleder.