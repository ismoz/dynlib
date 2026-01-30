# Grafikleri Dışa Aktarma

`dynlib.plot.export` yardımcısı, Matplotlib `savefig`/`show` iş akışını sarmalar; böylece tek bir panel, ızgara düzeni veya daha üst düzey bir kapsayıcı çiziyor olmanız fark etmeksizin betikleriniz aynı tutarlı varsayılanları korur.

## Çekirdek Fonksiyonlar

### `export.savefig(fig_or_ax, path, *, fmts=("png",), dpi=300, transparent=False, pad=0.01, metadata=None, bbox_inches="tight")`

- `fig_or_ax`, bir figür, eksen (axes) veya herhangi bir dynlib düzen nesnesini (`fig.grid`, `AxesGrid` vb.) kabul eder. Yardımcı fonksiyon alttaki figürü otomatik olarak bulur, böylece `fig.figure` nesnesini çağırmadan doğrudan çizim işleminden hemen sonra kullanabilirsiniz.
- `path` bir uzantı içerebilir (örneğin, `"plots/phase.png"`) veya içermeyebilir (örneğin, `"plots/phase"`). Bir uzantı ilettiğinizde, yardımcı fonksiyon yalnızca o formatı yazar; uzantıyı atlayıp `fmts` belirtirseniz birden fazla formatı tek seferde kaydedebilirsiniz.
- `fmts`, yoldan format çıkarımı yapılmadığı sürece varsayılan olarak `("png",)` değerini alır. Yardımcı fonksiyon, sağladığınız değerleri normalize eder, kopyaları temizler ve küçük harfe çevirir; böylece ekstra ayrıştırma yapmadan `(".PNG", ".pdf")` geçebilirsiniz.
- Geri kalan anahtar kelime argümanları Matplotlib'in `savefig` fonksiyonunu yansıtır. Çözünürlük için `dpi`, alfa arka planlar için `transparent`, boşluk eklemek için `pad` ve arama dostu etiketler yerleştirmek için `metadata` kullanın. `constrained_layout=True` ile dynlib'in `fig` yardımcılarına güvendiğinizde, bu yardımcı fonksiyon süslemeleri kesebilecek sıkı (tight) bir sınırlayıcı kutu uygulanmasını otomatik olarak önler.

### `export.show()`

Matplotlib'in `plt.show()` fonksiyonunu tetiklemek için bir betiğin, not defteri hücresinin sonunda veya herhangi bir etkileşimli oturumda `export.show()` çağrısı yapın. Dynlib'in stilini takip eder, bu nedenle CLI kullanıyor veya yardımcıları bir betikte içe aktarıyor olmanız fark etmeksizin figür numaralandırması ve düzenler aynı şekilde davranır.

## En İyi Uygulamalar

- `export` importlarını çizim yardımcılarınızın yanında tutun: `from dynlib.plot import fig, series, export`. Bu sayede her figür setinden sonra tutarlı bir şekilde `export.show()` çağırabilirsiniz.
- Birden fazla format kaydederken, uzantıyı `path`'ten çıkarın ve `fmts`'ye güvenin. Örneğin, `export.savefig(ax, "figures/lorenz", fmts=("svg","png"))` çağrısı, aynı dpi/pad ayarlarıyla `lorenz.svg` ve `lorenz.png` dosyalarını yazar.
- `axes = fig.grid(...)` gibi dynlib kapsayıcılarını veya `plot.vectorfield()` gibi yardımcıların dönüş değerlerini doğrudan `savefig`'e iletin; `export`, figürü otomatik olarak bulmak için kapsayıcıyı tarar.
- Yayıma hazır görseller oluştururken aranabilir anahtar kelimeler veya yazar bilgisi için `metadata` (string anahtar/değerlerinden oluşan bir sözlük) kullanın.

Uygulamalı bir örnek ve daha fazla biçimlendirme notu için [Temel çizim rehberine](basics.md) dönün veya çizim dokümanlarının geri kalanını inceleyin.