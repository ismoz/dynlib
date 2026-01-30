# Tamsayı Haritası Örneği: Collatz Dizisi

## Genel Bakış

Bu örnek, tamsayı durum değişkeni kullanan bir **harita (map)** simülasyonunun nasıl oluşturulacağını gösterir. Dokümantasyonun diğer bölümlerinde gösterilen kayan noktalı (floating-point) aritmetiğe dayalı lojistik harita örneklerinin aksine, bu model Collatz yinelemesini tamsayılar üzerinde gerçekleştirerek, tamsayı aritmetiğinin kullanımını sergiler.

## Öne Çıkan Kavramlar

- **Ayrık Zamanlı Simülasyon (`map`)**: Durumun her adımda bir kez güncellendiği ve sürekli zaman kavramının olmadığı ayrık zamanlı bir simülasyon türüdür.
- **Tamsayı Veri Tipi (`int64`)**: Collatz dizisinin elemanlarının tam olarak hesaplanmasını sağlayarak, önceden bilinen referans diziyle karşılaştırma yapılmasına olanak tanır.
- **`assert` ile Doğrulama**: `numpy.testing.assert_array_equal` fonksiyonu kullanılarak, simülasyon sonucu elde edilen yörüngenin, beklenen 1-4-2-1 döngüsüyle biten diziyle birebir aynı olup olmadığı kontrol edilir.
- **Seri Grafiği**: Tamsayı yörüngesi görselleştirilerek, dizinin ne zaman 4-2-1 döngüsüne yakınsadığı gösterilir.

## Collatz Konjektür Modelinin Tanımlanması

Model, başlangıç değeri 27 olan `n` adında tek bir tamsayı durum değişkeni ile tanımlanmıştır. Modelin denklemi, `n`'nin çift veya tek olmasına göre farklı hesaplama yapan `int64` türünde bir ifadedir. Bu tanım, aşağıda görüldüğü gibi TOML formatında satır içi (inline) olarak yapılmıştır.

```toml
[model]
type = "map"
dtype = "int64"
name = "Collatz Conjecture"

[states]
n = 27

[equations.rhs]
n = "n//2 if n % 2 == 0 else 3*n + 1"
```

## Simülasyonun Çalıştırılması

Betik, `setup(..., stepper="map")` fonksiyonu ile modeli oluşturur ve `len(expected) - 1` adım kadar simülasyonu çalıştırır. Simülasyon sonuçları `sim.results()` ile alınır. `series.plot` fonksiyonu, `n`'nin adımlara göre değişimini gösteren bir grafik çizer. Betik ayrıca, elde edilen dizinin, 27 ile başlayıp 1-4-2-1 döngüsüyle biten bilinen `expected` dizisiyle tam olarak eşleşip eşleşmediğini kontrol eder. Son olarak, yörüngenin son kısmı ve durum değişkeninin veri tipi ekrana yazdırılarak döngünün ve veri tipinin korunduğu doğrulanır.

## Grafik ve Görselleştirme

`theme.use("paper")` komutu, `matplotlib`'i yayın kalitesinde grafikler oluşturacak şekilde ayarlar. `export.show()` ise oluşturulan grafiği ekranda gösterir. Grafiğin eksen etiketleri (`iteration` ve `n`) ve başlığı (`Collatz Conjecture`), `n` değerlerinin meşhur 4-2-1 döngüsüne girmeden önceki değişimini kolayca incelemeyi sağlar.

## Referans Kod

Doğrulama ve çizim mantığını da içeren tam betik aşağıda verilmiştir:
```python
--8<-- "examples/collatz.py"
```
Bu betik, aynı zamanda bir regresyon testi işlevi görür. Çalıştırıldığında, simülasyonla üretilen her tamsayı değerinin, `expected` olarak tanımlanan NumPy dizisiyle birebir eşleştiğini doğrular.
