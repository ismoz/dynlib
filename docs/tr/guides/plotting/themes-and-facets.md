# Temalar & Facet'ler (Bölümlendirme)

Dynlib'in çizim sistemi, tutarlı ve yayına hazır figürler oluşturmak için güçlü tema ve facet (bölümlendirme) yetenekleri sağlar. Bu rehber, stil oluşturmak için `plot.theme` kullanımını ve çok panelli düzenler için `plot.fig`/`plot.facet` kullanımını kapsar.

## Temalara Genel Bakış

Dynlib'deki temalar, tüm çizimlerin görsel görünümünü kontrol ederek figürler arasında tutarlılık sağlar. Tema sistemi şunları yönetir:

- Yazı tipi boyutları ve aileleri
- Çizgi genişlikleri ve işaretçi (marker) stilleri
- Renk paletleri
- Izgara (grid) ve arka plan ayarları
- Boşluklar ve kenar boşlukları

Temalar küresel olarak uygulanır ve değiştirilene kadar sonraki tüm çizimleri etkiler.

### Yerleşik Hazır Ayarlar (Presets)

Dynlib, farklı kullanım durumları için optimize edilmiş birkaç önceden tanımlı tema içerir:

- **notebook**: Dengeli stillendirme ile etkileşimli Jupyter not defterleri için varsayılan tema.
- **paper**: Yayınlar için temiz tema; ince ızgaralar devre dışı bırakılmış ve yazı tipi boyutları optimize edilmiştir.
- **talk**: Sunumlar için yüksek kontrastlı tema; daha büyük öğeler ve daha kalın çizgiler içerir.
- **dark**: Daha iyi görünürlük için ayarlanmış renklere sahip koyu arka planlı tema.
- **mono**: Gri tonlamalı renkler kullanan monokrom tema.

### Temaları Kullanma

Çizim senaryonuzun başında bir tema ayarlayın:

```python
from dynlib.plot import theme

# Bir hazır ayar kullan
theme.use("paper")

# Veya bir hazır ayarın üzerine özelleştirme yap
theme.use("notebook", tokens={"scale": 1.2, "grid": False})
```

### Temaları Özelleştirme

Hazır ayarları değiştirmeden tema ayarlarını modifiye edebilirsiniz:

```python
# Ayarları geçici olarak düzenle
theme.update(tokens={"fontsize_title": 16, "line_w": 2.0})

# Veya kapsamlı değişiklikler için push/pop kullan
theme.push(tokens={"palette": "mono"})
# ... çizimleri oluştur ...
theme.pop()  # Önceki temaya geri döner
```

### Renk Paletleri

Dynlib birden fazla renk paletini destekler:

- **classic**: Standart Matplotlib renkleri
- **cbf**: Renk körü dostu palet (erişilebilirlik için önerilir)
- **mono**: Gri tonlamalı palet

Özel paletleri kaydedin:

```python
theme.register_palette("my_colors", ["#ff0000", "#00ff00", "#0000ff"])
theme.use("notebook", tokens={"palette": "my_colors"})
```

### Tema Jetonları (Tokens)

Temalar, bireysel stil özelliklerini belirten jetonlar tarafından kontrol edilir. Temel jetonlar şunlardır:

- **scale**: Genel boyut çarpanı
- **fontsize_***: Farklı öğeler için yazı tipi boyutları (temel, etiket, başlık vb.)
- **line_w**: Çizgi genişliği
- **marker_size**: İşaretçi boyutu
- **grid**: Izgara çizgilerinin gösterilip gösterilmeyeceği
- **palette**: Renk paleti adı
- **background**: "light" (açık) veya "dark" (koyu)

Mevcut jeton değerlerine erişim:

```python
current_scale = theme.get("scale")
font_size = theme.get("fontsize_title")
```

## Figür Izgaraları ve Düzenleri

Dynlib, temalarla sorunsuz çalışan figür düzenleri oluşturmak için üst düzey yardımcılar sağlar.

### Temel Figür Oluşturma

```python
from dynlib.plot import fig

# Tek çizim
ax = fig.single(title="Grafiğim")

# Alt çizimlerden (subplots) oluşan ızgara
axes = fig.grid(rows=2, cols=3, title="Parametre Taraması")

# 3B çizim
ax_3d = fig.single3D(title="3B Yörünge")

# Alt çizimleri saran (wrap) esnek ızgara
axes = fig.wrap(n=7, cols=3)  # 3x3 ızgara oluşturur, son 2 ekseni gizler
```

Tüm `fig` yardımcıları özelleştirme için parametreler kabul eder:

- `size`: (genişlik, yükseklik) demeti olarak figür boyutu
- `scale`: Boyut çarpanı
- `sharex`/`sharey`: Eksenlerin paylaşılıp paylaşılmayacağı
- `title`: Figür başlığı

### Çizim ile Entegrasyon

Oluşturulan eksenleri çizim fonksiyonlarına iletin:

```python
from dynlib.plot import fig, series

ax = fig.single()
series.plot(x=time, y=signal, ax=ax, label="Sinyal")
```

## Çok Panelli Figürler için Facetleme (Faceting)

Facetleme, ızgaraları otomatik olarak oluşturur ve veri kategorileri üzerinde yineleme yapar; bu parametre taramaları veya gruplandırılmış veriler için mükemmeldir.

### Temel Facetleme

```python
from dynlib.plot import facet, series

# Farklı parametreler için veriler
data = {
    "r=2.5": trajectory_r25,
    "r=3.0": trajectory_r30,
    "r=3.5": trajectory_r35,
}

# Facet'lenmiş çizim oluştur
for ax, param in facet.wrap(data.keys(), cols=2, title="Bifurkasyon Analizi"):
    traj = data[param]
    series.plot(x=time, y=traj, ax=ax, title=param)
```

`facet.wrap` fonksiyonu:

- Bir anahtar (kategoriler) yinelenebilir nesnesi alır
- Belirtilen sayıda sütuna sahip bir ızgara oluşturur
- Yineleme için (eksen, anahtar) çiftleri üretir
- Düzeni otomatik olarak yönetir ve kullanılmayan eksenleri gizler

### Facetleme Parametreleri

- `cols`: Izgaradaki sütun sayısı
- `title`: Genel figür başlığı
- `size`, `scale`: Figür boyutlandırma
- `sharex`/`sharey`: Eksen paylaşımı

### İleri Seviye Facetleme Örneği

```python
import numpy as np
from dynlib.plot import facet, series, theme

# Parametre taraması
r_values = np.linspace(2.8, 4.0, 12)

theme.use("paper")
for ax, r in facet.wrap(r_values, cols=4, title="Lojistik Harita Bifurkasyonları"):
    # Bu r değeri için yörünge simüle et
    x = 0.1
    traj = [x]
    for _ in range(100):
        x = r * x * (1 - x)
        traj.append(x)
    
    series.plot(x=range(len(traj)), y=traj, ax=ax, title=f"r={r:.1f}")
```

## En İyi Uygulamalar

1. **Temaları erken ayarlayın**: Herhangi bir figür oluşturmadan önce senaryonuzun başında temaları uygulayın.

2. **Tutarlı paletler kullanın**: Daha iyi erişilebilirlik için "cbf" gibi renk körü dostu paletleri tercih edin.

3. **Facetlemeden yararlanın**: Parametre taramaları veya gruplandırılmış veriler için facetleme, tekrar eden kodları (boilerplate) azaltır.

4. **Düşünerek özelleştirin**: Tamamen yeni temalar oluşturmak yerine küçük ayarlamalar için `theme.update()` kullanın.

5. **Değişiklikleri kapsamlandırın**: Geçici tema modifikasyonları için `theme.push()`/`pop()` kullanın.

6. **Uygun şekilde boyutlandırın**: Farklı çıktılar (ekran veya baskı) için figür boyutlarını ayarlamak adına `scale` parametresini veya tema jetonlarını kullanın.

Temalar ve facetleme, ister keşifsel analiz ister yayın figürleri için olsun, dynlib çizimlerinizin profesyonel ve tutarlı olmasını sağlamak için birlikte çalışır.