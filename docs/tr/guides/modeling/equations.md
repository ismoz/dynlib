# Denklemler

`[equations]` tablosu, durumların (states) her adımda nasıl değiştiğini tanımladığınız yerdir. Modelinize uygun stili seçebilmeniz için birbirinin yerine kullanılabilen birkaç alt formu kabul eder.

## Temel formlar

- `[equations.rhs]` (durum-başına) – `state = "expr"` girişlerinden oluşan bir TOML tablosu. Her ifade bir dize (string) olmalıdır, böylece DSL, o durum için sağ tarafı değerlendirmeden önce makroları ayrıştırabilir.
- `[equations].expr` (blok) – Her satırın bir duruma atama yaptığı (`x = ...`) veya ODE modelleri için türev notasyonu kullandığı (`dx = ...` veya `d(x) = ...`) çok satırlı tek bir dize. Cebirinizi düzenli tutan stili kullanın, ancak aynı durumu her iki yerde de tanımlamayın (yükleyici bunu zorunlu kılar).
- `[equations.inverse]` – Yalnızca `map` modelleri için mevcuttur; ana denklem formunu yansıtır ve çağrılabilir bir ters güncelleme sağlar. Bu tablo içinde `rhs` veya `expr` tanımlayabilirsiniz, ancak aynı durum için ikisini birden tanımlayamazsınız.
- `[equations.jacobian]` – Yoğun (dense) Jacobian'ı tanımlayan kare bir liste-içinde-liste (list-of-list) değişmezi içeren tek bir `expr` anahtarına sahip isteğe bağlı meta veriler (her giriş bir dize veya sayısal değişmez olabilir). Bu tablo yalnızca derleyici için özel türevler sağladığınızda kullanılır (örneğin, stiff (sert) çözücüler veya implicit stepper'lar için).

### Örnek

```toml
[equations.rhs]
x = "speed * cos(theta)"
theta = "speed * sin(theta)"

[equations.inverse]
expr = """
x = x - speed * cos(theta)
theta = theta - speed * sin(theta)
"""

[equations.jacobian]
expr = [
  ["0", "-speed * sin(theta)"],
  ["speed * cos(theta)", "0"]
]
```

## İfade bağlamı

Denklem ifadeleri, başka yerlerde mevcut olan aynı tanımlayıcıları paylaşır: states, parameters, constants, aux, functions, makrolar (`sin`, `clip`, `approx`, generator kapsamaları) ve `t`. ODE blokları ayrıca türev hedeflerini (`dx`, `d(x)`) kabul eder, ancak map modelleri `state = expr` atamalarına bağlı kalmalıdır.

## Ters (Inverse) denklemler

- `inverse` tablosu yalnızca map modelleri için vardır ve tersine çevirme araçları ve teşhisleri tarafından kullanılan bir `inv_rhs` çağrılabilir öğesi sağlar.
- Bunu bir durum-başına tablo (`[equations.inverse.rhs]`) veya bir blok dizesi (`[equations.inverse].expr`) olarak yazabilir ve birincil denklem formunu yansıtabilirsiniz.
- Her durum, ters formlar arasında yalnızca bir kez görünebilir; aynı durum için `rhs` ve `expr` karıştırmak bir hataya neden olur.
- Ters güncelleme, ileri denklemlerle aynı tanımlayıcı kümelerini (states, params, aux vb.) çözümlemelidir.

## Jacobian tablosu

- `[equations.jacobian].expr` bir satır listesidir; satır ve sütun sayısı, bildirilen durumların (states) sayısıyla eşleşmelidir (kare matris).
- Her matris girişi bir dize ifadesi veya sayısal bir değişmez (tamsayılar/float'lar) olabilir. Derleyici bunu, çözücü desteği için kullanılan açık Jacobian'a düzleştirir.
- Yoğun bir Jacobian'a ihtiyacınız varsa ancak bunu düzenli tutmayı tercih ediyorsanız, paylaşılan ifadeleri aux değişkenleri ile önceden hesaplayabilir ve bunlara matris girişleri içinde referans verebilirsiniz.

## Doğrulama ipuçları

- Ayrıştırıcı, `[equations]`, `[equations.inverse]` ve `[equations.jacobian]` içindeki bilinmeyen anahtarları yasaklar, böylece yazım hataları erken yakalanır.
- Durumlar, `[equations.rhs]` ve `[equations].expr` arasında yalnızca bir kez tanımlanabilir; ters tablo için de aynısı geçerlidir.
- Map modelleri türev notasyonu kullanamaz (yükleyici, map'ler için `[equations].expr` içindeki `d(x)` ifadesini açıkça reddeder).
- `[equations.jacobian].expr` bir satır listesi olarak sağlanmalıdır; çoğul `exprs` kullanmak veya tabloyu atlamak bir hataya neden olur.

## En iyi uygulamalar

1.  **Durum başına tek bir stile bağlı kalın** (ya `rhs` ya da blok), böylece gereksiz (tekrarlayan) mantıktan kaçınırsınız.
2.  **Aux/functions kullanın**, karmaşık sağ taraf ifadelerini çarpanlarına ayırarak denklem tablolarının okunabilir kalmasını sağlayın.
3.  **Ters tabloları net bir şekilde belgeleyin**—neden var olduklarını belirtin (örneğin, geriye doğru adım atmak veya teşhis için), çünkü bunlar normal çözücü yolunun dışında çalışır.
4.  **Yalnızca gerektiğinde bir Jacobian sağlayın** (implicit çözücüler, stiffness); aksi takdirde, derleyicinin türevleri sayısal olarak tahmin etmesine izin verin.