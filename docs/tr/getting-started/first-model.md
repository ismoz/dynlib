# İlk Modelinizi Oluşturma

Bu sayfada, basit bir modeli hem bağımsız bir TOML dosyası olarak hem de Dynlib'in dosya sistemine yazmadan işleyebileceğiniz bir `inline:` ifadesiyle nasıl tanımlayacağınızı öğreneceksiniz. Daha ileri DSL konuları için Modelleme rehberlerine göz atın: TOML yapıları için [DSL temelleri](../guides/modeling/dsl-basics.md), inline içerikler için [inline modeller](../guides/modeling/inline-models.md) ve modeli kayıt defterine eklemek isterseniz [konfigürasyon dosyası](../guides/modeling/config-file.md).

## 1. Temel TOML tanımı

Köşeli parantez `[]` ile yazılan TOML ifadelerine tablo denilir. Bir model oluşturmak için en azından `[model]`, `[states]` ile birlikte bir denklem tablosu (`[equations]` vb.) belirtilmelidir. `[model]` tablosu altında modelin türüne göre `type = "map"` veya `type = "ode"` kullanın. `name` ifadesi ile modelin ismini; `dtype` (ing. data type kısaltması) ifadesi ile veri türünü belirleyebilirsiniz. 

`[params]` (ing. prameters kıslatması) tablosu altına parametreleri ve bunların değerlerini alt alta yazabilirsiniz. Benzer şekilde `[states]` (ing. state variables kısaltması) tablosu altına durum değişkenlerini ve bunların başlangıç değerlerini alt alta yazabilirsiniz.

`[equations]` veya `[equations.rhs]` tablosu altına ise sistemin denklemleri yazılır. İkisi arasındaki fark şu şekildedir:

```toml
[equations]
expr = """
x = r * x * (1 - x)
"""
``` 

!!! note "Buradaki `expr` ifadesi İngilizce expression ifadesinin kısaltmasıdır."

```toml
[equations.rhs]
x = "r * x * (1 - x)"
```
 
Aşağıdaki örneği proje klasörünüze `first-model.toml` (veya benzeri bir yol adı) olarak kaydedin:

```toml
[model]
type = "map"
name = "Simple Logistic Map"
dtype = "float64"

[params]
r = 3.9

[states]
x = 0.2

[equations.rhs]
x = "r * x * (1 - x)"
```

[DSL temelleri](../guides/modeling/dsl-basics.md), ekleyebileceğiniz sabitler, yardımcı değişkenler, olaylar, Jacobian tabloları gibi diğer tabloları ve Dynlib'in bu ifadelerin hepsini nasıl yorumladığını anlatır. Tanımınızın sözdiziminin (syntax) geçerli olup olmadığını doğrulamak için `dynlib model validate first-model.toml` (veya `python -m dynlib.cli model validate ...`) komutunu çalıştırın.

## 2. Inline (satır içi) tanımlarla hızlı prototiplendirme

Eğer farklı bir model dosyası oluşturmadan hızlıca bir model tanımlayıp hemen simülasyon yapmak isterseniz bir python dosyası içerisinde satır içi (inline) bir model tanımlayabilirsiniz. Satır içi model tanımı üç adet tek tırnak arasına aynı TOML modelinin yazılmasıyla oluşturulur. `setup()` veya `build()` araçlarının modelin satır içi olduğunu anlayabilmesi için model tablosunun üstüne `inline:` ifadesi eklenmelidir.

```python
from dynlib import setup

spec = '''
inline:
[model]
type = "map"
name = "Inline Logistic"
dtype = "float64"

[params]
r = 3.9

[states]
x = 0.3

[equations.rhs]
x = "r * x * (1 - x)"
'''

sim = setup(spec)
sim.run(N=30)
```

[Satır içi modeller](../../guides/modeling/inline-models.md), kabul edilen URI biçimlerini ve `inline:` kullanım senaryolarını listeler. Detaylar için ilgili dökümantasyonu inceleyebilirsiniz.

## 3. Basit bir ODE örneği

Sürekli zamanlı bir sistemi modellemek için `type = "ode"` seçip durum denklemlerini `[equations.rhs]` altında tanımlayın. Diğer tabloların kullanımı yukarıdaki modellemeyle aynıdır. Tek fark denklemlerin artık zamana göre türevi temsil etmesidir:

```toml
[model]
type = "ode"
name = "Simple Harmonic Oscillator"
dtype = "float64"

[params]
omega = 1.0

[states]
x = 1.0
v = 0.0

[equations.rhs]
x = "v"
v = "-omega ** 2 * x"
```

Bu dosyayı `dynlib model validate harmonic.toml` ile doğrulayabilir veya satır içi (inline) tanımlayarak çalıştırabilirsiniz:

```python
sim = setup("path/to/harmonic.toml")
sim.run(T=30)
```

## 4. Sonraki adımlar

Tanımınızı doğruladıktan sonra modeli çalıştırmak ve yeniden kullanmak için şu adımları izleyin:

- `setup("first-model.toml", ...)` ile bir `Sim` nesnesi oluşturup [simülasyonlar](../guides/simulation/index.md) veya [analizler](../guides/analysis/index.md) yapın.
- Bir yapılandırma dosyasıyla klasörü kaydederek `proj://first-model.toml` gibi etiketlerle modeli bilgisayarınızdaki herhangi bir konumda tekrar tekrar kullanın. Konfigürasyon dosyasıyla hangi dosyanın hangi etikete denk geleceğini ve Dynlib’in kayıt defterini nerede tuttuğunu netleştirmek için [konfigürasyon rehberini](../guides/modeling/config-file.md) okuyun.
- Simülasyon sonuçlarını elde etmek ve kullanmak için [sonuçlar](../guides/simulation/results.md) sayfasını inceleyin.
- Sonuçları çizdirmek için [çizim](../guides/plotting/index.md) sayfasını inceleyin.


