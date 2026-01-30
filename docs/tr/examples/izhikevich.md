# Izhikevich Nöron Modeli

## Genel Bakış

Bu örnek, Izhikevich nöron modelinin nasıl simüle edileceğini ve farklı giriş akımları altında membran potansiyelinin nasıl görselleştirileceğini adım adım göstermektedir. Örnek, `dynlib`'in bir simülasyonu devam ettirme, model ön ayarlarını değiştirme ve grafikleri açıklayıcı notlarla zenginleştirme yeteneklerini öne çıkarır. Bu sayede, sürücü akımındaki değişikliklerin nöronun ateşleme düzenini nasıl etkilediği net bir şekilde görülebilir.

## Temel Kavramlar

- **Kademeli Akım Simülasyonu**: Simülasyonun durumunu koruyarak ve enjekte edilen akımı güncelleyerek tek bir simülasyonu çalıştırma. Bu yöntem, hem geçici dinamikleri hem de yeni çekerleri (attractors) yakalamayı sağlar.
- **Ön Ayarları Uygulama**: `sim.apply_preset("bursting")` komutu ile nöronun uyarılabilirliğini belirleyen `c`, `d` gibi içsel parametreleri değiştirme.
- **Anlık Görüntü (Snapshot) Araçları**: Farklı rejimlerden geçtikten sonra kaydedilen yapılandırmayı incelemek için `sim.list_snapshots()` komutu ve `source="snapshot"` parametresi ile `sim.param_vector` / `param_dict` kullanma.
- **Açıklamalı Zaman Serisi Grafikleri**: `series.plot`, enjekte edilen akımın değiştiği rejimleri etiketlemek için `vbands` (dikey şeritler) ve `vlines` (dikey çizgiler) gibi görsel elemanları destekler.

## Izhikevich Modeli

`dynlib` içerisindeki yerleşik model, aşağıdaki 2-boyutlu sistemi içerir:

$$
\begin{align}
\frac{dv}{dt} &= 0.04v^2 + 5.0v + 140.0 - u + I \\
\frac{du}{dt} &= a(bv - u)
\end{align}
$$ 

Membran potansiyeli `v` bir eşik değerine (`v_th`, varsayılan 30.0) ulaştığında, bir sıfırlama olayı tetiklenir ve değişkenler şu şekilde güncellenir:
- `v`'nin yeni değeri `c` olur.
- `u`'nun yeni değeri `u + d` olur.

Varsayılan parametreler (`a=0.02`, `b=0.2`, `c=-65`, `d=8`, `I=10`) düzenli ateşleme (regular spiking) davranışı üretir. `bursting` (patlama) ön ayarı ise `c` ve `d` değerlerini değiştirerek, artan akımla birlikte ortaya çıkan hızlı ateşleme patlamaları oluşturur.

## Örnek Kod

Aşağıdaki kod, farklı akım seviyelerinde simülasyonu çalıştırır ve "bursting" ön ayarını uygulayarak sonuçları görselleştirir.

```python
--8<-- "examples/izhikevich.py"
```

Kodda `series.plot` fonksiyonu, her bir akım adımını işaretlemek için dikey şeritler ve çizgiler ekler. `run` fonksiyonundaki `resume=True` parametresi, akım değeri değiştikçe simülasyonun durumunu koruyarak kesintisiz devam etmesini sağlar. Grafik çizdirildikten sonra, anlık görüntü (snapshot) yardımcı fonksiyonları, daha sonraki analizler için kaydedilen parametre setlerini yazdırır.

```