# Örneklere Genel Bakış

İlgilendiğiniz konularla en alakalı çalıştırılabilir örnekleri kolayca bulmak için bu sayfaları kullanabilirsiniz. Her sayfa, ilgili örneğin ne yaptığını açıklar, kullandığı `dynlib` özelliklerini vurgular ve `examples/` klasöründeki orijinal koda bir bağlantı içerir.

## Katalog
- [`Analiz`](analysis.md) – Çekim havzalarının (basin of attraction) hesaplanması, Lyapunov üstel spektrumunun analizi, kararsız periyodik yörüngelerin manifoldlarının izlenmesi, homoklinik ve heteroklinik yörüngelerin bulunması ve parametre taraması gibi analiz araçlarının kullanımını gösteren örnekler.
- [`Çizim`](plotting.md) – Zaman serisi ve faz uzayı çizimleri, vektör alanı görselleştirmeleri, animasyonlar ve temalar gibi `dynlib`'in çizim yeteneklerini gösteren örnekler.
- [`Çalışma Zamanı`](runtime.md) – Simülasyonu belirli bir koşul sağlandığında erken sonlandırma, yörüngedeki geçişleri tespit etme, derlenmiş denklemleri inceleme ve nümerik hassasiyeti değerlendirme gibi çalışma zamanı yeteneklerini gösteren örnekler.
- [`Durum Yönetimi`](state-management.md) – Simülasyon durumunu anlık görüntüler (snapshots) ile kaydedip yükleme, modelin kaynak kodunu dışa aktarma, hazır ayarları (presets) kullanma ve URI'ler aracılığıyla modellere erişme gibi özellikleri gösteren örnekler.
- [`Çatallanma`](bifurcation.md) – Parametre uzayını tarayarak (sweeping) lojistik harita gibi sistemlerin çatallanma (bifurcation) diyagramlarının nasıl oluşturulduğunu gösteren örnekler.
- [`Tamsayı Harita`](integer-map.md) – Tamsayı aritmetiği ve `map` tipi adımlayıcı (stepper) kullanarak Collatz dizisi gibi yinelemeli sistemlerin nasıl modelleneceğini gösteren bir örnek.
- [`Izhikevich Nöron`](izhikevich.md) – Farklı parametreler ve uyaranlar (stimulus) altında Izhikevich nöron modelinin nasıl simüle edileceğini ve farklı ateşleme (spiking) desenlerinin nasıl elde edileceğini gösteren örnekler.