# Çatallanma (Bifurcation) Diyagramı: Lojistik Harita

## Genel Bakış

Bu örnek, `dynlib` kütüphanesinin parametre tarama ve çizim yeteneklerini kullanarak çatallanma (bifurcation) diyagramlarının nasıl oluşturulacağını açıklamaktadır. Çatallanma diyagramları, bir sistem parametresi değiştikçe sistemin uzun vadedeki davranışını görselleştirir. Bu sayede sabit noktalar, periyodik yörüngeler ve kaotik davranışlar arasındaki geçişler net bir şekilde gözlemlenebilir.

Bu kavramı incelemek için klasik bir örnek olan lojistik harita (`x_{n+1} = r·x_n·(1-x_n)`) kullanılacaktır. Bu modelde, `r` parametresi 2.5'ten 4.0'a doğru artırılırken gözlemlenen **periyot katlanmalarıyla kaosa geçiş** süreci incelenecektir.

## Temel Kavramlar

- **Parametre Taraması**: Bir model parametresinin belirli bir aralıktaki farklı değerleri için sistemin davranışını (yörüngelerini) hesaplama işlemidir.
- **Diyagram için Veri Çıkarımı**: Parametre taraması sonucu elde edilen yörünge verilerinden, diyagramı çizmek için kullanılacak noktaların (saçılım verisinin) ayıklanmasıdır.
- **Veri Çıkarma Stratejileri**: `dynlib`, yörüngelerden veri ayıklamak için farklı stratejiler sunar. Örneğin, yörüngenin tamamını (`all`), son kısmını (`tail`), sadece son noktasını (`final`) veya yerel ekstremumlarını (`extrema`) kullanabilirsiniz.
- **Periyot Katlanması ile Kaosa Geçiş**: Sistemin kararlı durumdan periyodik davranışa, ardından periyotların ikiye katlanarak giderek daha karmaşık hale geldiği ve sonunda kaosa ulaştığı klasik bir senaryodur (Feigenbaum senaryosu).

## Lojistik Harita Modeli

Lojistik harita denklemi aşağıdaki gibidir:

$$
x_{n+1} = r \cdot x_n \cdot (1 - x_n)
$$

Bu modelde `r` parametresinin kritik değerleri ve sistemin davranışı:
- **r < 3.0**: Sistem, tek bir kararlı sabit noktaya yakınsar.
- **r = 3.0**: İlk periyot katlanması gerçekleşir ve 2-periyotlu bir yörünge ortaya çıkar.
- **r ≈ 3.449**: Bir sonraki katlanma ile 4-periyotlu yörünge başlar.
- **r ≈ 3.57**: Kaosun başlangıcı olarak kabul edilen Feigenbaum noktası (r∞ ≈ 3.5699).
- **r = 4.0**: Sistemin tamamen kaotik davranış sergilediği durum.

## Temel Örnek

Aşağıdaki kod, `dynlib` kütüphanesinin yerleşik lojistik harita modelini kullanarak basit bir çatallanma diyagramı oluşturur.

**Adımlar:**
1.  **Modeli Kur**: `setup` fonksiyonu ile lojistik harita modeli hazırlanır.
2.  **Parametre Aralığını Belirle**: `numpy.linspace` ile taranacak `r` değerleri oluşturulur.
3.  **Parametre Taraması Yap**: `sweep.traj_sweep` fonksiyonu, her bir `r` değeri için modelin yörüngesini hesaplar.
    - `transient`: Sistemin kararlı hale gelmesi için başlangıçta atlanacak adım sayısı.
    - `N`: `transient` adımlarından sonra kaydedilecek adım sayısı.
4.  **Veriyi Çıkar**: `bifurcation` metodu, yörünge sonuçlarından diyagram için gerekli veriyi ayıklar.
5.  **Diyagramı Çiz**: `bifurcation_diagram` fonksiyonu ile sonuçlar görselleştirilir.

```python
from dynlib import setup
from dynlib.analysis import sweep
from dynlib.plot import bifurcation_diagram, theme, fig, export
import numpy as np

# 1. Yerleşik lojistik harita modelini kur
sim = setup("builtin://map/logistic", stepper="map", jit=True)

# 2. Taranacak 'r' parametresi için değer aralığı
r_values = np.linspace(2.8, 4.0, 5000)

# 3. Parametre taramasını çalıştır
sweep_result = sweep.traj_sweep(
    sim,
    param="r",           # Taranacak parametre
    values=r_values,     # Parametre değerleri
    record_vars=["x"],   # Kaydedilecek değişken
    N=100,               # Her bir 'r' değeri için kaydedilecek adım sayısı
    transient=500,       # Kararlı yörüngeye ulaşmak için atlanacak adım sayısı
)

# 4. Diyagram için yörüngelerden 'x' verisini çıkar
result = sweep_result.bifurcation("x")

# 5. Sonuçları çizdir
theme.use("notebook")
ax = fig.single(size=(10, 6))

bifurcation_diagram(
    result,
    xlabel="r",
    ylabel="x*",
    title="Lojistik Harita için Çatallanma Diyagramı",
    ax=ax
)

export.show()
```

## Detaylı Örnekler

Proje içerisindeki `examples/` dizininde, bu konuda daha detaylı çalışan kodları bulabilirsiniz.

### 1. Yüksek Çözünürlüklü Çatallanma Diyagramı

Bu örnek, standart iş akışını kullanarak yüksek çözünürlüklü bir diyagram oluşturur.

```python
--8<-- "examples/bifurcation_logistic_map.py"
```

**Özellikleri:**
- Daha detaylı bir görüntü için yüksek çözünürlüklü parametre taraması (20.000 nokta).
- Varsayılan veri çıkarma modu kullanılır (`all`, yani yörüngedeki tüm noktalar).
- Minimum özelleştirme ile sade bir çizim yapılır.

### 2. Veri Çıkarma Stratejilerinin Karşılaştırılması

Bu örnek, farklı veri çıkarma stratejilerinin diyagram üzerindeki etkisini gösterir.

```python
--8<-- "examples/bifurcation_logistic_map_comparison.py"
```

Örnekte üç farklı strateji yan yana karşılaştırılır:

```python
extractor = sweep_result.bifurcation("x")

# Strateji 1: Sadece yörüngenin son değeri kullanılır
result_final = extractor.final()

# Strateji 2: Yörüngenin son 50 noktası kullanılır (çekicinin yapısını gösterir)
result_tail = extractor.tail(50)

# Strateji 3: Yörüngenin son 100 noktasındaki yerel ekstremumlar (maksimum ve minimumlar) kullanılır
result_extrema = extractor.extrema(tail=100, max_points=30)
```
Bu farklı stratejiler, sistemin periyodik veya kaotik davranışını farklı açılardan vurgulamaya yardımcı olur.
