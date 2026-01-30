# Havza (Basin) Çizimleri

Havza çizimleri, parametre uzayının 2D ızgarasını renklendirerek her bir başlangıç koşulunun hangi çekere (attractor) yerleştiğini ortaya koyar. `plot.basin_plot()`, çekim havzası analiz araçları tarafından üretilen kategorik etiketleri bir `pcolormesh`'e dönüştürür, böylece havzaların yapısını, özel sonuçları ve renk lejantının her bir çekerle nasıl ilişkili olduğunu anında görebilirsiniz.

## Çizim ne gösteriyor

Her ızgara hücresi, analiz sırasında tanımlanan bir başlangıç koşuluna karşılık gelir. O hücre için saklanan değer bir tam sayı etiketidir:

- **Çeker (Attractor) Kimlikleri** (0, 1, …), bilinen bir çekere yakınsayan yörüngeleri işaretler.
- **Özel değerler** (`BLOWUP`, `OUTSIDE`, `UNRESOLVED`), ıraksayan yörüngeleri, ilgi alanından kaçışları veya hesaplama bütçesi içinde bir karara varamayan başlangıç koşullarını işaretler.

`basin_plot()`, bu tam sayıları renklere eşler ve önce özel sonuçları, ardından indeks sırasına göre çekerleri etiketleyen bir renk çubuğu (colorbar) oluşturur.

## Verilerinizi hazırlama

`analysis.basin_auto()` (veya `analysis.basin_known()`) tarafından döndürülen `BasinResult` nesnesini doğrudan `basin_plot()` fonksiyonuna iletin. Yardımcı araç, kategorik ızgara için `res.labels` verisini okur ve ızgara boyutlarını (`ic_grid`), sınırları (`ic_bounds`), gözlemlenen değişkenleri (`observe_vars`) ve çeker meta verilerini (`attractor_labels`/`attractor_names`) çıkarmak için `res.meta` verisini kullanır.

Etiketleri kendiniz hesapladıysanız, bunları `labels=` ile iletin; 1D diziler, yardımcının 2D'ye yeniden şekillendirebilmesi için `grid=(nx, ny)` şeklini gerektirirken, önceden şekillendirilmiş 2D diziler doğrudan sağlanabilir. Alternatif olarak, etiket dizisiyle eşleşen açık `x` ve `y` koordinatlarını sağlayın.

```python
from dynlib import setup
from dynlib.analysis import basin_auto
from dynlib.plot import basin_plot

sim = setup("models/henon.map", stepper="map")
res = basin_auto(sim, ic_grid=[256, 256], ic_bounds=[(-2, 2), (-2, 2)])
basin_plot(res)
```

## Renk haritasını (colormap) kontrol etme

`basin_plot()`, özel sonuçlarla (varsayılan sıra `[BLOWUP, OUTSIDE, UNRESOLVED]`) başlayan ve ardından çeker kimlikleri gelen tek bir renk haritası oluşturur. Varsayılanları şunlarla geçersiz kılabilirsiniz:

- `special_order`: Özel kimliklerin sırasını değiştirin veya çıkarın.
- `special_colors`: Her özel etiket için bir renk sağlayın; daha fazla giriş isterseniz varsayılanlar mevcut Matplotlib paletinden türetilen gri tonlamalı çiftlere geri döner.
- `special_labels`: Renk çubuğunda görünen özel girişleri yeniden adlandırın (örneğin, `{"blowup": "Iraksadı"}`).
- `attractor_cmap`: Çekerler için `"hsv"` yerine herhangi bir Matplotlib renk haritasına (veya bir `Colormap` örneğine) geçiş yapın.
- `attractor_colors`: Bir renk haritasını örneklemek yerine açık renkler verin.
- `attractor_labels`: Renk çubuğundaki çeker adlarını özelleştirin; varsayılan olarak `meta["attractor_labels"]`, `meta["attractor_names"]` veya `A0`, `A1`, … kullanılır.

```python
basin_plot(
    res,
    special_colors=["#1a1a1a", "#444444", "#777777"],
    attractor_cmap="viridis",
    attractor_labels=["Periyot-1", "Periyot-2"],
    colorbar_label="Sonuç",
)
```

`basin_plot()`, etiketlerden daha az renk isterseniz hata verir, bu nedenle paletinizin sonuçtaki özel sonuç ve çeker sayısıyla eşleştiğinden emin olun.

## Eksen, sınırlar ve ek açıklamalar

Eksen sınırları, etiketler ve tik (işaret) stillendirmesi olağan `plot` yardımcıları tarafından yönetilir:

- `bounds` veya `res.meta["ic_bounds"]`, başlangıç koşulları oluşturulurken kullanılan `(x_min, x_max)`/`(y_min, y_max)` aralıklarını belirtir. Eğer `x` ve `y` dizilerini sağlarsanız, `bounds` yok sayılır.
- `xlabel`, `ylabel` ve `title`, `matplotlib.axes.Axes` etiketleri gibi davranır. Sonuç meta verileri `observe_vars` içeriyorsa, bunları geçersiz kılmadığınız sürece bu isimler otomatik olarak `xlabel`/`ylabel` alanlarını doldurur.
- `xlim`, `ylim`, `aspect`, `xlabel_fs`, `ylabel_fs`, `xtick_fs`, `ytick_fs`, `xlabel_rot`, `ylabel_rot`, `title_fs`, `titlepad`, `xpad` ve `ypad`, görünümü ince ayar yapmanıza olanak tanır.
- Yardımcı, mevcut bir `ax=` parametresini kabul eder, böylece havza haritasını `plot.fig()`/`plot.theme()` veya doğrudan Matplotlib tarafından üretilen çok panelli bir şekle yerleştirebilirsiniz.

Çizim `pcolormesh` kullandığından, `shading` varsayılan olarak `"auto"`dur ve başka bir veri setinden konturlar veya havzalar bindirmeyi düşünüyorsanız ızgarayı soluklaştırmak için `alpha` kullanılabilir.

## Renk çubuğu (colorbar) ayarları

Renk çubuğunu kaldırmak için `colorbar=False` olarak ayarlayın. Aksi takdirde, yardımcı otomatik olarak özel girişler ve ardından çeker kimlikleri için tikleri ayarlar. Şunları kullanın:

- Eksen başlığını ayarlamak için `colorbar_label`, `colorbar_label_rotation` ve `colorbar_labelpad`.
- `plt.colorbar()` fonksiyonuna ekstra ayarlar (örneğin `{"fraction": 0.05}`) iletmek için `colorbar_kwargs`.

`plot.basin_plot()`, oluşturulan `Colorbar` nesnesini diğer çizim yardımcılarını yansıtacak şekilde `ax._last_colorbar` üzerinde saklar.

## İpuçları

- Yalnızca düzleştirilmiş etiketler iletirken `grid` parametresini sağlayın; yardımcı, `pcolormesh` için bunları `(ny, nx)` şeklinde nasıl yeniden boyutlandıracağını bilmelidir.
- `res.meta["ic_grid"]` ve `res.meta["ic_bounds"]` genellikle analiz rutinleri tarafından doldurulur, bu nedenle çizim yaparken bunları nadiren tekrar etmeniz gerekir.
- Çok parametreli bir sonucun farklı bir dilimini görselleştirmek için, `basin_plot()` çağrısından önce etiket dizisini dilimleyin ve `bounds` değerini buna göre güncelleyin (yardımcı, çok boyutlu dilimleri otomatik olarak yeniden şekillendirmez).
- Kimlikler yerine çeker adlarını vurgulamak istiyorsanız, her zaman `attractor_labels` sağlayın; böylece çeker kayıt defterinizin sırasına bakılmaksızın renk çubuğu tikleri net bir şekilde okunur.

`basin_plot()` her zaman `Axes` nesnesini döndürür, böylece Matplotlib komutları veya [Çizim Süslemeleri](decorations.md) bölümünde açıklanan paylaşılan süsleme argümanları (`vlines`, `hlines`, `vbands`, `hbands`) ile şekle ek açıklamalar yapmaya devam edebilirsiniz.