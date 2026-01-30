# Çizim Temelleri

Dynlib'in çizim modülü (`dynlib.plot`), dynlib'in analiz iş akışlarıyla uyumlu olması için Matplotlib'i sarmalayan bir dizi üst düzey yardımcı araç sağlar. Bu araçlar, tutarlı stil ve düzen sağlarken zaman serileri, faz portreleri ve manifoldlar gibi dinamik sistemler için yaygın çizim görevlerini yerine getirir. Tüm yardımcılar NumPy dizilerini, listeleri, pandas Series'leri veya `Results` nesnelerinden alınan dilimleri kabul eder ve bunları gerektiği gibi otomatik olarak dönüştürür.

Bu rehber, şekiller oluşturmaktan çizimleri özelleştirmeye kadar bu çizim yardımcılarını kullanmanın temellerini kapsar.

## Başlarken

Dynlib'in çizim araçlarını kullanmak için gerekli modülleri içe aktarın:

```python
from dynlib.plot import fig, series, phase, utils
```

İşte bir alt çizim (subplot) ızgarası oluşturmayı ve farklı türde verileri çizmeyi gösteren hızlı bir örnek:

```python
# 2x2'lik bir alt çizim ızgarası oluştur
axes = fig.grid(rows=2, cols=2, size=(10, 8))

# Bir zaman serisi çiz
series.plot(x=t, y=x_traj, label="x(t)", ax=axes[0, 0])

# Ayrık verileri stem (çubuk) olarak çiz
series.stem(x=k, y=impulse_response, ax=axes[0, 1])

# Denge noktalarıyla birlikte bir faz portresi çiz
phase.xy(x=x_traj, y=y_traj, equil=[(x_eq, y_eq)], ax=axes[1, 0])

# Renk çubuğu (colorbar) ile 2D bir görüntü sergile
utils.image(Z, extent=[0, 10, 0, 1], colorbar=True, ax=axes[1, 1])
```

Dynlib'in `fig` yardımcıları şekil oluşturma ve yerleşim düzenini otomatik olarak yönetir, böylece Matplotlib'in basmakalıp kodları yerine verilere odaklanabilirsiniz.

## Şekiller ve Alt Çizimler Oluşturma

Dynlib, tutarlı stile sahip şekiller ve alt çizimler oluşturmak için kullanışlı fonksiyonlar sağlar:

- `fig.single()`: Tek bir alt çizim oluşturur.
- `fig.grid(rows=2, cols=2)`: Bir alt çizim ızgarası oluşturur.
- `fig.wrap(n=5, cols=3)`: Belirli sayıdaki alt çizimi sütunlara saran ve kullanılmayan eksenleri gizleyen bir ızgara oluşturur.
- `fig.single3D()`: Tek bir 3D alt çizim oluşturur.

Bu fonksiyonlar, özelleştirme için `title`, `size`, `scale`, `sharex` ve `sharey` gibi parametreleri kabul eder. Çizim yardımcılarına iletebileceğiniz Matplotlib eksen nesnelerini döndürürler.

Çizimleri veri kategorilerine göre fasetlemek (bölümlere ayırmak) için, eksenleri ve yineleme anahtarlarını veren `plot.facet.wrap(keys, cols=3)` fonksiyonunu kullanın.

Örnek:

```python
# Tek bir alt çizim oluştur
ax = fig.single(size=(8, 6))
series.plot(x=t, y=data, ax=ax)

# Birden fazla çizim için bir ızgara oluştur
axes = fig.grid(rows=1, cols=3)
for i, dataset in enumerate(datasets):
    series.plot(x=t, y=dataset, ax=axes[i])
```

## Stillendirme ve Süslemeler

Dynlib'in çizim yardımcıları, ön ayarlar ve süslemeler aracılığıyla tutarlı stillendirmeyi destekler.

### Stil Ön Ayarları (Style Presets)

Stil ön ayarları, verilerin nasıl görselleştirileceğini tanımlar (örneğin; çizgiler, işaretçiler veya her ikisi). Mevcut ön ayarlar arasında `"continuous"`, `"discrete"`, `"line"`, `"scatter"` ve diğerleri bulunur. Bir ön ayar adı veya özel bir stil sözlüğü iletebilirsiniz.

Örnek:

```python
# Bir ön ayar kullan
series.plot(x=t, y=data, style="continuous", ax=ax)

# Geçersiz kılmalarla özelleştir
series.plot(x=t, y=data, style={"ls": "--", "marker": "x"}, color="red", ax=ax)
```

### Süslemeler (Decorations)

Özellikleri vurgulamak için dikey veya yatay çizgiler ve bantlar ekleyin:

- `vlines`: Dikey çizgiler (örneğin, `vlines=[5, (10, "eşik")]`)
- `hlines`: Yatay çizgiler
- `vbands`: Dikey bantlar
- `hbands`: Yatay bantlar

Etiketler otomatik olarak konumlandırılır ve eksen sınırlarına uyar.

Örnek:

```python
series.plot(x=t, y=data, vlines=[(5, "başlangıç"), 10], hbands=[(0, 1, "bölge")], ax=ax)
```

### Eksen Kontrolü

`xlim`, `ylim` ve `zlim` (3D için) ile eksen sınırlarını kontrol edin. Yardımcılar tutarlı etiketleri, yazı tiplerini ve döndürmeleri otomatik olarak uygular.

## Zaman Serilerini Çizme

Zaman tabanlı çizimler için `series` yardımcılarını kullanın:

- `series.plot(x, y, ...)`: Sürekli veya ayrık veriler için standart çizgi grafiği.
- `series.stem(x, y, ...)`: Ayrık örnekler için "stem" (çubuk) grafiği.
- `series.step(x, y, ...)`: Parçalı sabit veriler için basamak grafiği.
- `series.multi(data, ...)`: Aynı anda birden fazla seriyi çizer.

Bu yardımcılar tüm stillendirme ve süsleme seçeneklerini destekler.

Örnekler:

```python
# Basit zaman serisi
series.plot(x=t, y=x_traj, label="Konum", ax=ax)

# Çoklu seriler
data = {"x": x_traj, "y": y_traj}
series.multi(data, styles={"x": "continuous", "y": "discrete"}, ax=ax)

# İmpulslar için stem grafiği
series.stem(x=k, y=impulse, ax=ax)
```

## Faz Portrelerini Çizme

Faz uzayı çizimleri, durum değişkenleri arasındaki ilişkileri görselleştirir:

- `phase.xy(x, y, ...)`: 2D faz portresi.
- `phase.xyz(x, y, z, ...)`: 3D faz portresi.
- `phase.multi(x_list, y_list, ...)`: Tek bir çizimde birden fazla yörünge.
- `phase.return_map(x, step, ...)`: Haritalar (maps) için geri dönüş haritası.

Denge noktalarını `equil` ile işaretleyin ve etiketleri özelleştirin.

Örnekler:

```python
# 2D faz portresi
phase.xy(x=x_traj, y=y_traj, equil=[(0, 0)], ax=ax)

# Geri dönüş haritası (Return map)
phase.return_map(x=trajectory, step=1, equil=[(fixed_point,)], ax=ax)
```

## Yardımcı Çizimler

Yaygın görselleştirmeler için ek yardımcılar:

- `utils.hist(data, ...)`: 1D verilerin histogramı.
- `utils.image(data, ...)`: 2D görüntü grafiği.

Her ikisi de stillendirmeyi destekler ve renk çubukları (colorbars) içerebilir.

Örnek:

```python
# Histogram
utils.hist(data, bins=50, density=True, ax=ax)

# Renk çubuğu ile görüntü
utils.image(matrix, extent=[0, 1, 0, 1], colorbar=True, ax=ax)
```

## Şekilleri dışa aktarma ve görüntüleme

Bir şekli sunmanız veya kaydetmeniz gerektiğinde `dynlib.plot` içinden `export` modülünü içe aktarın. Bu modül, Matplotlib'in `savefig` yardımcılarını dynlib'e duyarlı varsayılanlarla yeniden dışa aktarır, böylece ekstra kod yazmadan tüm eksen ızgaralarını veya şekil tanıtıcılarını (handles) iletebilirsiniz. Çizim senaryonuz tamamlandığında `export.show()` fonksiyonunu çağırın (not defterleri veya betikler için kullanışlıdır) veya birden fazla formatta yazmak için `export.savefig(fig_or_ax, "plots/my-fig", fmts=("png", "pdf"))` fonksiyonunu kullanın. Format seçimi, meta veriler ve ızgara kapları (grid containers) ile çalışma hakkında ayrıntılar için özel [Çizimleri dışa aktarma rehberi](export.md)'na bakın.

## Manifoldları Çizme

1D manifoldlar için `plot.manifold(segments, ...)` kullanın veya dalları olan `result` nesnelerini iletin. Farklı gruplar için bileşenleri ve stilleri belirtin.

Örnek:

```python
plot.manifold(result.branches, components=(0, 1), ax=ax)
```

## İpuçları ve En İyi Uygulamalar

- Çizimlerin nerede görüneceğini kontrol etmek için çok panelli şekiller oluştururken her zaman `ax=` kullanın.
- Hızlı stillendirme için stil ön ayarlarından yararlanın, ardından gerektiğinde bunları geçersiz kılın.
- Çizgiler ve bantlar gibi süslemeler, tutarlı ek açıklamalar (annotations) için çoğu yardımcıda çalışır.
- Yardımcılar veri dönüşümünü otomatik olarak yönetir, bu nedenle NumPy dizilerini, listeleri ve dynlib sonuçlarını serbestçe karıştırabilirsiniz.
- Renk çubuğu olan görüntülerde, daha fazla özelleştirme için renk çubuğuna `ax._last_colorbar` üzerinden erişin.

Bu bilgiler dynlib'in çizim araçlarıyla başlamanızı sağlayacaktır. Daha gelişmiş özellikler için API referansına bakın.