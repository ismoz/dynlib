# Matematik Fonksiyonları, Skaler Makrolar ve Generator Comprehension'lar

## Yerleşik Matematik Fonksiyonları

Aşağıdaki matematiksel fonksiyonlar kullanıma hazırdır ve doğrudan Python'un `math` modülüne veya yerleşik fonksiyonlarına eşlenir:

### Temel Fonksiyonlar
- `abs(x)` - Mutlak değer
- `min(x, y, ...)` - Argümanların en küçüğü
- `max(x, y, ...)` - Argümanların en büyüğü
- `round(x)` - En yakın tamsayıya yuvarlama

### Üstel ve Logaritmik Fonksiyonlar
- `exp(x)` - Üstel fonksiyon (e^x)
- `expm1(x)` - exp(x) - 1 (küçük x değerleri için daha hassas)
- `log(x)` - Doğal logaritma
- `log10(x)` - 10 tabanında logaritma
- `log2(x)` - 2 tabanında logaritma
- `log1p(x)` - log(1 + x) (küçük x değerleri için daha hassas)
- `sqrt(x)` - Karekök

### Trigonometrik Fonksiyonlar
- `sin(x)` - Sinüs
- `cos(x)` - Kosinüs
- `tan(x)` - Tanjant
- `asin(x)` - Ters sinüs (arksinüs)
- `acos(x)` - Ters kosinüs (arkkosinüs)
- `atan(x)` - Ters tanjant (arktanjant)
- `atan2(y, x)` - İki argümanlı ters tanjant

### Hiperbolik Fonksiyonlar
- `sinh(x)` - Hiperbolik sinüs
- `cosh(x)` - Hiperbolik kosinüs
- `tanh(x)` - Hiperbolik tanjant
- `asinh(x)` - Ters hiperbolik sinüs
- `acosh(x)` - Ters hiperbolik kosinüs
- `atanh(x)` - Ters hiperbolik tanjant

### Yuvarlama Fonksiyonları
- `floor(x)` - Taban (bir alt tamsayıya yuvarlar)
- `ceil(x)` - Tavan (bir üst tamsayıya yuvarlar)
- `trunc(x)` - Kırpma (kesirli kısmı atar)

### Özel Fonksiyonlar
- `hypot(x, y)` - Öklid uzaklığı (sqrt(x^2 + y^2))
- `copysign(x, y)` - y'nin işaretini x'in büyüklüğüne kopyalar
- `erf(x)` - Hata fonksiyonu
- `erfc(x)` - Tamamlayıcı hata fonksiyonu

## Skaler Makrolar

Skaler makrolar, yaygın matematiksel işlemleri gerçekleştiren özel fonksiyonlardır:

- `sign(x)` - İşaret fonksiyonu: negatif için -1, sıfır için 0, pozitif için 1 döndürür
- `heaviside(x)` - Heaviside basamak fonksiyonu: x < 0 için 0, x >= 0 için 1 döndürür
- `step(x)` - heaviside ile aynı (takma ad/alias)
- `relu(x)` - Rectified Linear Unit: max(0, x) döndürür
- `clip(x, min, max)` - x değerini [min, max] aralığına sıkıştırır/sınırlar
- `approx(x, y, tol)` - |x - y| <= tol kontrolü yapar (boolean döndürür)

## Generator Comprehension'lar

DSL, aralıklar üzerinde verimli toplama ve çarpma işlemleri için "generator comprehension" yapılarını destekler:

- `sum(expr for var in range(start, stop[, step]) [if condition])` - Bir aralıktaki ifadelerin toplamı
- `prod(expr for var in range(start, stop[, step]) [if condition])` - Bir aralıktaki ifadelerin çarpımı

Bu yapılar optimize edilmiş for döngülerine derlenir. Yineleyici (iterator) olarak sadece `range()` desteklenir ve sadece tek bir üreteç (generator) kullanılabilir. `if` ile koşullu filtreleme desteklenir.

Örnekler:
- `sum(i*i for i in range(10))` - 0'dan 9'a kadar karelerin toplamı (0+1+4+...+81)
- `prod((i+1) for i in range(1, 5))` - Çarpım 2×3×4×5 = 120
- `sum(x[i] for i in range(N) if i % 2 == 0)` - Çift indeksli elemanların toplamı (x'in bir dizi olduğu varsayılarak)

## Olay (Event) Makroları

Olay makroları, olay koşullarında durum değişikliklerini ve geçişleri algılamak için kullanılır. Bu makrolar karşılaştırma için otomatik olarak geçmiş (lag uygulanmış) durum değerlerini kullanır:

- `cross_up(state, threshold)` - Durum, eşiği aşağıdan yukarıya kestiğinde Doğru (True) olur
- `cross_down(state, threshold)` - Durum, eşiği yukarıdan aşağıya kestiğinde Doğru (True) olur
- `cross_either(state, threshold)` - Durum, eşiği herhangi bir yönde kestiğinde Doğru (True) olur
- `changed(state)` - Durum değeri önceki adıma göre değiştiğinde Doğru (True) olur
- `in_interval(state, lower, upper)` - Durum şu anda [lower, upper] aralığındaysa Doğru (True) olur
- `enters_interval(state, lower, upper)` - Durum [lower, upper] aralığına girdiğinde Doğru (True) olur
- `leaves_interval(state, lower, upper)` - Durum [lower, upper] aralığından çıktığında Doğru (True) olur
- `increasing(state)` - Durum artıyorsa (mevcut > önceki) Doğru (True) olur
- `decreasing(state)` - Durum azalıyorsa (mevcut < önceki) Doğru (True) olur