# Çizim Örnekleri

## Genel Bakış

Bu sayfada, `dynlib`'in çizim araçlarının nasıl kullanılacağını gösteren demolar yer almaktadır. Bu araçlar, `dynlib.plot` API'si altında toplanmış olup; zaman serileri, faz portreleri, dönüş haritaları (return maps), vektör alanları, histogramlar ve animasyonlar oluşturmak için tutarlı ve basit bir arayüz sunar.

## Zaman Serileri ve Faz Portreleri

### Lojistik Harita Tanılamaları

Aşağıdaki örnek, yerleşik lojistik harita modelini oluşturur, bir başlangıç geçiş sürecini (transient) atladıktan sonra simülasyonu çalıştırır. Sonrasında `series.plot`, `return_map` ve `cobweb` yardımcı fonksiyonlarını kullanarak zaman serisini, dönüş haritasını ve örümcek ağı diyagramını (cobweb diagram) çizer. Ayrıca, analitik tahminleri sayısal olarak bulunan çekerlerle (attractors) karşılaştırmak için `sim.model.fixed_points(seeds=...)` ile bulunan sabit noktaları da ekrana yazar.

```python
--8<-- "examples/logistic_map.py"
```

### Van der Pol Osilatörü

Bu örnek, `builtin://ode/vanderpol` gibi katı (stiff) bir diferansiyel denklem modelinin `tr-bdf2a` gibi özel bir çözücü ile nasıl çalıştırılacağını gösterir. Simülasyon süresini `dynlib.utils.Timer` ile ölçer ve hem zaman serisini hem de faz portresini çizer. Bu problem $\mu=1000$ parametre değeriyle diğer numerik çözücüler için simülasyonu son derece zordur.

```python
--8<-- "examples/vanderpol.py"
```

## Çizim Yardımcıları Galerisi

### Temel Çizim Fonksiyonları

Bu örnek, `series.stem`, `series.step`, `utils.hist`, `phase.xy` ve `series.plot`'un `map` ve `mixed` stilleri gibi daha az kullanılan çizim fonksiyonlarını altı farklı alt grafikte gösterir. Her bir fonksiyonun farklı veri türlerini (sürekli, ayrık, dağılım) nasıl işlediğini hızlıca görmek için bir referans niteliğindedir.

```python
--8<-- "examples/plot/plot_demo.py"
```

### Tema Ayarları

Aşağıdaki betik, `notebook`, `paper`, `talk`, `dark` ve `mono` gibi mevcut tüm tema ön ayarlarını dener. Her bir tema için örnek bir figür oluşturur ve PNG dosyası olarak kaydeder. Bu sayede her ön ayarın renkleri, ızgara çizgilerini ve tipografiyi nasıl etkilediğini inceleyebilirsiniz.

```python
--8<-- "examples/plot/themes_demo.py"
```

### Facet (Bölümlenmiş) Grafikler

`plot.facet.wrap` fonksiyonu, farklı kategorilerdeki veriler için bir alt grafik ızgarası oluşturmayı kolaylaştırır. Her bir eksen, kendi veri dilimini ve başlık/etiketleri otomatik olarak alır. Bu, veri dağılımlarını manuel olarak `plt.subplots` oluşturmadan keşfetmek için kullanışlıdır.

```python
--8<-- "examples/plot/facet.py"
```

## Vektör Alanları

### Temel Vektör Alanı Çizimi

Bu örnek, `plot.vectorfield` aracını kullanarak bir spiral modelin vektör alanını çizer. Örnekte, `nullcline`'ların (sıfır büyüme çizgileri) gösterimi, akış hızına göre renklendirme (`speed_color`) ve geri dönen `handle` nesnesi aracılığıyla parametrelerin (`a`, `b`) güncellenerek grafiğin yeniden çizilmesi gösterilmektedir.

```python
--8<-- "examples/plot/vectorfield_demo.py"
```

### Yüksek Boyutlu Sistemlerin Vektör Alanı Kesitleri

Bu örnekte, 3-boyutlu Lorenz sisteminin vektör alanı, seçilen 2-boyutlu düzlemlere (`x/y` ve `y/z`) yansıtılır. `fixed` parametresi ile sabitlenen durum değişkenlerinin (`z` ve `x`) değerleri ayarlanabilir. Ayrıca, `interactive=True` seçeneği aktifleştirildiği için panellerden herhangi birine tıklayarak o kesit üzerinden geçen kısa bir yörüngeyi anında çizebilirsiniz.

```python
--8<-- "examples/plot/vectorfield_highdim_demo.py"
```

### Parametre Değerlerine Göre Vektör Alanı Taraması

`plot.vectorfield_sweep` fonksiyonu, bir model parametresinin (`a`) farklı değerleri için vektör alanlarını tek bir ızgara üzerinde otomatik olarak çizer. Bu, farklı parametre rejimlerinin sistemin dinamiğini nasıl değiştirdiğini bir bakışta karşılaştırmanızı sağlar.

```python
--8<-- "examples/plot/vectorfield_sweep_demo.py"
```

## Animasyonlar

### Vektör Alanı Animasyonları

Aşağıdaki örnek, `plot.vectorfield_animate` kullanarak bir parametrenin (`a`) belirli değerler arasında nasıl değiştirileceğini ve bu değişimin vektör alanını nasıl etkilediğini gösteren bir animasyon oluşturur. Oluşturulan `anim` nesnesi, bir değişkene atanmalıdır; aksi takdirde Python'un çöp toplayıcısı (garbage collector) tarafından silinebilir ve animasyon görüntülenmez.

```python
--8<-- "examples/plot/vectorfield_animate_demo.py"
```

Bu örnekte ise, bir sin/cos tabanlı vektör alanının frekans parametresi `k`, 300 kare boyunca taranarak bir animasyon oluşturulur.

```python
--8<-- "examples/plot/vectorfield_animation.py"
```
