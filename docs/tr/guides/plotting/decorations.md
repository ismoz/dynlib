# Grafik Süslemeleri

Dynlib çizim yardımcıları, her üst düzey görselleştiricinin `src/dynlib/plot/_primitives.py` içindeki `_apply_decor()` aracılığıyla ilettiği tek bir süsleme argümanları setini kullanıma sunar. Bu yardımcı, `series.plot()`, `series.step()` ve benzeri giriş noktalarından gelen aynı parametreleri kabul eder, böylece süslemeler her yerde aynı şekilde davranır.

## Dikey Çizgiler (`vlines`)

- `vlines` parametresine ya skaler değerlerden oluşan bir liste ya da `(x, etiket)` demetleri (tuples) iletin. Demetler (tuples), ayrı bir `text()` çağrısına ihtiyaç duymadan, ilgili çizginin yanında etiketlerle birlikte oluşturulur.
- Görünümü ayarlamak için `vlines_kwargs` (veya çoğu yardımcıda bulunan kolaylık eşleşmesi `vlines_color`/`vlines_kwargs`) sağlayabilirsiniz. Varsayılan kwargs değerleri şöyledir: `color='black'`, `linestyle='--'`, `linewidth=1`, `alpha=0.7`.
- Etiket konumlandırması, `_apply_decor()` tarafından yakalanıp geri kalanı `ax.axvline()` fonksiyonuna iletilmeden önce dört özel kwargs ile kontrol edilir:
  - `label_position`: `'top'`, `'bottom'`, `'center'` seçeneklerinden biri. Metnin ofsetler uygulanmadan önce eksenin üstüne/altına/ortasına mı sabitleneceğini belirler.
  - `placement_pad`: Metin bağlantı noktasını hesaplarken eksen boyunca ek ofset ekler (`<1` ise eksen yüksekliğinin oranı, aksi takdirde veri birimi olarak).
  - `label_pad`: Etiketi çizgiye dik olarak (yani yatay olarak) hareket ettirir; yine `<1` değerlerini eksen oranı ve `>=1` değerlerini veri birimi olarak yorumlar.
  - `label_rotation`/`label_color`: Döndürme (varsayılan 90°) ve metin rengini (varsayılan olarak çizgi rengi) geçersiz kılar.

Örnek:
```python
series.plot(
    x=t,
    y=x_traj,
    vlines=[(3.0, "period-2"), 3.57],
    vlines_kwargs={
        "label_position": "bottom",
        "placement_pad": 0.08,
        "label_pad": 0.05,
        "label_rotation": 90,
        "linestyle": ":",
        "color": "firebrick",
    },
)
```

## Yatay Çizgiler (`hlines`)

- `vlines` ile aynı şekilde çalışır, ancak y koordinatları içindir. Etiketler `(y, etiket)` demetleri (tuples) aracılığıyla sağlanabilir.
- `hlines_color`, yalnızca rengi değiştirmenin yaygın olduğu durumlar için mevcuttur; `_apply_decor()` çalışmadan önce `hlines_kwargs` ile birleştirilir.
- Özel kwargs benzerdir ancak yatay geometriyi yansıtır:
  - `label_position`: Etiketin eksenin hangi tarafına yaslanacağını seçmek için `'left'`, `'right'` veya `'center'`.
  - `placement_pad`: Bağlantı noktasını x ekseni boyunca kaydırır (`<1` = eksen oranı, `>=1` = veri birimi).
  - `label_pad`: Etiketi çizgiye dik olarak öteler (dikey olarak hareket ettirir), aynı eksen-vs-veri-birimi yorumu geçerlidir.
  - `label_rotation` yatay metin için varsayılan olarak `0` derecedir ve `label_color` yine varsayılan olarak çizgi rengini kullanır.

Örnek:
```python
series.plot(
    x=t,
    y=x_traj,
    hlines=[(0.25, "low"), (0.75, "high")],
    hlines_kwargs={
        "label_position": "left",
        "placement_pad": 0.1,
        "label_pad": 0.02,
        "label_color": "navy",
        "linestyle": "-",
        "alpha": 0.6,
    },
)
```

## Dikey Bantlar (`vbands`)

- Dikey bölgeleri gölgelendirmek için bir `(başlangıç, bitiş)` demetleri listesi (isteğe bağlı olarak renk için üçüncü bir girişle) iletin: `(başlangıç, bitiş)` varsayılan renk `C0`'ı kullanır, `(başlangıç, bitiş, "teal")` bunu geçersiz kılar.
- `_apply_decor()`, `başlangıç < bitiş` koşulunu zorunlu kılar ve `ax.axvspan(start, end, color=color, alpha=0.1)` kullanarak oluşturur.

Örnek:
```python
series.plot(
    x=t,
    y=x_traj,
    vbands=[(2.5, 2.9, "gold"), (3.4, 3.6)],
)
```

## Yatay Bantlar (`hbands`)

- `vbands` gibi davranır ancak yatay şeritleri doldurmak için `ax.axhspan()` kullanır. Demetler (tuples) isteğe bağlı bir renk içerebilir.
- Yardımcı fonksiyon çizimden önce `başlangıç < bitiş` kontrolü yapar ve `alpha=0.1` ile `color='C0'` varsayılanını kullanır.

Örnek:
```python
series.plot(
    x=t,
    y=x_traj,
    hbands=[(0, 0.2), (0.8, 1.0, "lightcoral")],
)
```

## Özel kwargs özeti (tüm süsleme yardımcıları)

- `vlines_kwargs` / `hlines_kwargs`, olağan Matplotlib çizgi argümanlarına ek olarak şu etiket yerleştirme yardımcılarını kabul eder: `label_position`, `placement_pad`, `label_pad`, `label_rotation`, `label_color`.
- `placement_pad` ve `label_pad`, `<1` değerlerini ilgili eksen aralığının kesri/oranı olarak, `>=1` değerlerini ise veri birimi olarak ele alır; böylece göreceli ve mutlak ofsetler arasında geçiş yapabilirsiniz.
- Demet (tuple) tabanlı çizgi tanımları (değer + etiket) otomatik metin oluşturmayı tetikler; her etiketi global olarak özelleştirmek için yine de `label_color` ve `label_rotation` sağlayabilirsiniz.
- Bantlar (`vbands`, `hbands`) yalnızca `(başlangıç, bitiş)` veya `(başlangıç, bitiş, renk)` kabul eder ve demet uzunluğu yanlış olduğunda veya `başlangıç >= bitiş` olduğunda `ValueError` verir.

Başka eksen notasyonlarına (örneğin, manuel `text()` çağrıları veya ekstra artistler) ihtiyacınız varsa, bunları bu süslemelerle karıştırabilirsiniz; `_apply_decor()` eksen başına yalnızca bir kez çalışır ve diğer artistlere dokunmaz.