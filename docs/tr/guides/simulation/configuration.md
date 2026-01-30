# Simülasyon Yapılandırması

Bu rehber, çalışma zamanı varsayılanlarının nasıl ayarlanacağını, model durumu/parametre bankasının nasıl düzenleneceğini ve `Sim` ile çalışırken kayıt (recording) ve kapasite sınırlarının (caps) nasıl kontrol edileceğini açıklar.

## `Sim.config()` ile varsayılanları kalıcı hale getirme

`Sim.config()`, belirli bir argüman belirtilmediğinde `run()` komutunun kullanacağı *kalıcı* varsayılanlar ayarlamanıza olanak tanır. En yaygın simülasyon düğmelerini ve stepper'a özgü `Config` alanlarını kapsar.

```python
sim.config(
    dt=0.01,
    max_steps=5000,
    record=True,
    record_interval=10,
    cap_rec=2048,
    cap_evt=4,
    tol=1e-6,          # aktif stepper yapılandırmasına iletilir
)
```

Öne Çıkanlar:

- `dt`, nominal zaman adımını (veya ayrık modeller için etiket aralığını) ayarlar ve simülasyon durumunda `_nominal_dt` olarak saklanır. Pozitif olmalıdır.
- `max_steps`, `N`/`T` atlandığında varsayılan güvenlik sınırı (sürekli) veya hedef iterasyon sayısı (ayrık) olur.
- `record` ve `record_interval`, varsayılan günlük kaydı davranışını tanımlar ve geçersiz kılınmadıkça sonraki `run()` çağrıları tarafından devralınır.
- `cap_rec`/`cap_evt`, yörünge/olay tamponlarının başlangıç boyutlarını kontrol eder. Gerekirse otomatik olarak büyürler, ancak daha büyük başlangıç kapasiteleri yeniden tahsis işlemlerini (reallocations) azaltabilir.
- Ekstra anahtar kelime argümanları `Sim.stepper_config()`'e iletilir, böylece `tol`, `max_iter` veya diğer stepper düğmelerini global olarak yapılandırabilirsiniz.

`run()`'a verilen açık argümanların bu varsayılanları her zaman geçersiz kıldığını unutmayın; tekrarlanabilir scriptler için `config()` ile `run(...)` geçersiz kılmalarını birlikte kullanın.

## `Sim.assign()` ile durum ve parametreleri ayarlama

`Sim.assign()`, hiçbir şeyi yeniden derlemeden oturumun mevcut durum ve parametre vektörlerini isme göre günceller.

```python
sim.assign({"v": -65.0, "I": 5.0})
sim.assign(I=8.0, clear_history=True)
```

Temel davranışlar:

- Bir eşleme (mapping) ve/veya anahtar kelime argümanları kabul eder; anahtar kelimeler harita girdilerini geçersiz kılar.
- İsimleri önce `states`, sonra `params` içinde arar; bilinmeyen isimler "bunu mu demek istediniz?" önerileriyle net bir `ValueError` verir.
- Girdileri model veri tipine dönüştürür, hassasiyet kaybı olacaksa bir uyarı verir.
- `clear_history=True`, zamanı, çalışma alanını (workspace), snapshot'ları veya stepper yapılandırmasını değiştirmeden birikmiş `Results` verisini, segmentleri ve bekleyen etiketleri temizler.
- Değişiklikler, `run()` komutuna açık `ic`/`params` argümanları geçmediğiniz sürece bir sonraki `run()` için hemen geçerli olur.

`Sim`'i yeni koşullarla yeniden kullanmak, deneyler arasında parametreleri değiştirmek veya devam etmeden (resume) önce sistemi hazırlamak istediğinizde `assign()` kullanın.

## Kayıt (Recording) seçenekleri

`Sim.run()`, `config()` veya `[sim]` tablosu aracılığıyla belirlediğiniz varsayılanların yanı sıra günlük kaydı üzerinde hassas kontrol sunar.

- `record` (bool): kaydı aç/kapa. `False` olduğunda, yalnızca global zaman ekseni güncellenir; durum/aux tamponları büyümez.
- `record_interval` (int): her N'inci adımı yakalar (varsayılan `1`). Örnekleme sıklığını düşürmek (downsampling) veya hızlı simülasyonları ucuza yakalamak için kullanışlıdır.
- `record_vars`: seçici kayıt listesi. Kabul edilebilir girişler:
  - `None` (varsayılan) : Mevcut tüm durum değişkenleri.
  - Önek (prefix) olmayan isimler durumlara (states) atıfta bulunur.
  - `"aux.<isim>"` açıkça yardımcı değişkenleri hedefler, ancak `aux` isimlerini önek olmadan göndermek de kabul edilir ve ayırt edilir.
  - Boş bir liste (`[]`), zaman damgalarını, adımları ve bayrakları kaydetmeye devam ederken durum/aux kaydını devre dışı bırakır.

Seçici kayıt, aynı `Results` tampon düzenini korur ancak yalnızca istenen alt kümeleri doldurur, bu da büyük durum vektörleri için bellek/zaman tasarrufu sağlar.

Ayrıca, bir çalıştırmadan önce `cap_rec` ve `cap_evt` ile kayıt kapasitesini ayarlayabilir veya `record_interval`/`record_vars` neyin yakalanacağına karar verirken dynlib'in tamponları otomatik olarak büyütmesine izin verebilirsiniz.

## Simülasyon ufuklarını (horizons) yönetme

- `dt` nominal adım boyutunu kaydeder. Kalıcı bir varsayılan için `config(dt=…)` kullanın veya çalıştırma başına geçersiz kılın.
- `T` (sürekli) veya `N` (ayrık), runner'ın ne kadar ilerleyeceğini tanımlar; her ikisi de atlanırsa `max_steps` devreye girer (varsayılan yine de `[sim].max_steps`'e saygı duyar).
- Haritalarla (`kind="map"`) çalışırken, `N` iterasyonları belirler ve `T` türetilir; ODE'ler için `T` bitiş zamanıdır ve `N` çıkarım yoluyla bulunur.
- `max_steps`, sürekli modellerde bir güvenlik önlemi olarak uygulanır ve ayrık modellerde `N` atlandığında varsayılan iterasyon sayısı olarak işlev görür. Ufuk büyüdüğünde bunu artırın veya kontrolsüz döngülerden kaçınmak için küçültün. Bir runner `max_steps`'e ulaşırsa, sessiz ve beklenmedik davranışlardan kaçınmak için bir uyarı verilir.
- `transient`, saklanan `Results` verisini etkilemeden başlangıçtaki ısınma periyodunu (zaman veya iterasyon olarak) atlayabilir. `transient` periyodu boyunca hiçbir şey kaydedilmez. Zamanın `transient` periyodundan sonra `t0`'dan başladığını unutmayın; bu, bazı kullanıcılar için sezgisel olmayabilir.
- `resume=True`, son `SessionState`'den yeniden başlatmaya izin verir; devam etme (resume) modunda `ic`, `params`, `t0` ve `dt`'nin geçersiz kılınamayacağını unutmayın.

Çok aşamalı deneyleri yönetirken bu seçenekleri `Sim.reset()`/`Sim.create_snapshot()` ile eşleştirin, böylece ufukları değiştirseniz bile segmentler ve kaydedilen geçmiş üzerinde kontrolü elinizde tutarsınız.

## Özet

- `dt`, `max_steps`, kayıt, kapasiteler ve stepper ayarları için uzun ömürlü varsayılanları bildirmek üzere `Sim.config()` kullanın.
- Modeli yeniden oluşturmadan durum/parametre değerlerini güncellemeniz gerektiğinde `Sim.assign()` çağırın.
- Doğruluk ve bellek arasında denge kurmak için `run(record_vars=…)`, `record_interval` ve tampon sınırlarını (caps) kullanın.
- Harita veya sürekli sistem simülasyonu yapıp yapmadığınıza bağlı olarak `T`, `N` ve `max_steps` değerlerini ayarlayın ve aşamalandırmayı kontrol etmek için `transient`/`resume` kullanın.
- Hem tekrarlanabilirlik hem de hıza ihtiyaç duyduğunuzda yapılandırma seçeneklerini snapshot'lar, segmentler ve `setup()` ile birleştirin.