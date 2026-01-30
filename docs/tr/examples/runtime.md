# Çalışma Zamanı Araçları ve Tanılama

## Genel Bakış

Bu bölümdeki çalışma zamanı araçları, `dynlib`'in düşük seviyeli kontrol mekanizmalarını kullanır. `setup()` fonksiyonu ile çeşitli adım denetleyicileri (stepper), durma koşulları ve olay gözlemcileri yönetilebilir. Yardımcı API'lar ise derleyicinin ne derlediğini veya bir yörüngenin gerçek zamanlı olarak nasıl davrandığını anlamayı sağlar. Bu betikler, sayısal doğruluğun nasıl kontrol edileceğini, simülasyonların nasıl erken sonlandırılacağını, önemli geçişlerin nasıl tespit edileceğini ve `Sim` içinde tam olarak neyin çalıştığını denetlemek için DSL denklemlerinin nasıl yazdırılacağını gösterir.

## Örnek betikler

### Adım denetleyici (stepper) doğruluğunu sıralama

Bu betik, kayıtlı her bir ODE adım denetleyicisini, küçük ve sabit bir `dt` değeriyle iki analitik çözüme (üstel sönüm ve harmonik osilatör) karşı çalıştırır. Her çalıştırma için bir RMS (Kök Ortalama Kare) göreli hata hesaplanır. Bu sayede adım denetleyicileri doğruluğa göre sıralayabilir, hataları kaydedebilir ve zaman adımınız kararlılık sınırına yakınken her bir entegratörün nasıl davrandığını karşılaştırabilirsiniz.

```python
--8<-- "examples/accuracy_demo.py"
```

### Erken çıkış

Bu örnek, `cross_up`, `in_interval` ve `decreasing` gibi makroların DSL `stop` ifadesi ile beraber kullanımını gösterir. Betik, çeşitli lojistik harita varyantları için 100 adıma kadar çalışır ancak belirtilen koşul doğru olur olmaz durur. Çıkış nedenini, yürütülen adım sayısını ve tetikleyici koşuldan hemen önceki ve sonraki değerleri yazdırır. Bu özellik, erken durmanın zaman kazandırırken çalışmayı nasıl deterministik tuttuğunu vurgular.

```python
--8<-- "examples/early_exit_demo.py"
```

### Olaylar ile geçiş tespiti

Bu örnek, Lorenz sisteminin `x` değişkeni sıfırı aşağıdan yukarıya kestiğinde tetiklenen bir olay (event) ekler. Çalıştırma, her `detect` olayının zaman damgasını kaydeder, `x(t)` zaman serisi grafiğine dikey çizgiler çizer ve kaotik rejimlerde ek açıklamalar yapmak veya başka işlemleri yönlendirmek için `res.event("detect")` verisinin nasıl inceleneceğini gösterir.

```python
--8<-- "examples/detect_transition.py"
```

### Derlenmiş denklemleri yazdırma

Bu betik, Henon haritasını ve Lorenz sistemini `jit=False` seçeneği ile oluşturur ve derleyicinin TOML veya yerleşik (builtin) tanımlardan türettiği Sağ Taraf (RHS) ve Jakobiyen tablolarını yazdırır. Bu, maliyetli simülasyonları çalıştırmadan önce `dynlib`'in beklediğiniz modeli doğru bir şekilde yorumlayıp yorumlamadığını doğrulamak için kullanışlıdır. Eğer simülasyonun farklı sonuçlar verdiğinden şüpheleniyorsanız derlenmiş denklemleri kontrol edebilirsiniz.

```python
--8<-- "examples/print_equations_demo.py"
```
