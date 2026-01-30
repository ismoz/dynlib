# Örümcek Ağı (Cobweb) Grafikleri

Örümcek ağı grafikleri, bir boyutlu ayrık dinamik sistemlerin (haritaların) davranışını analiz etmek için güçlü bir görselleştirme aracıdır. Bir fonksiyonun iterasyonlarının zaman içinde nasıl geliştiğini anlamak için sezgisel bir yol sağlarlar ve sabit noktalar, periyodik yörüngeler ve kaotik davranış gibi önemli dinamik özellikleri belirlemeye yardımcı olurlar.

## Örümcek Ağı Grafikleri Nasıl Çalışır?

Bir örümcek ağı grafiği, bir fonksiyonu tekrar tekrar uygulama (iterasyon) sürecini görselleştirir. Bir $f(x)$ fonksiyonu için, $x_0$ başlangıç değerinden başlayarak, iterasyon şu diziyi oluşturur:

$$x_{n+1} = f(x_n)$$

Örümcek ağı grafiği bu iterasyonu geometrik olarak şu şekilde temsil eder:

1.  **Fonksiyon eğrisi**: $y = f(x)$ çizilir.
2.  **Birim (Identity) doğrusu**: $y = x$ çizilir (kesikli çizgi olarak gösterilir).
3.  **İterasyon yolu**: Her iterasyonun $(x_n, x_n)$ noktasından $(x_n, x_{n+1})$ noktasına, oradan da $(x_{n+1}, x_{n+1})$ noktasına nasıl hareket ettiğini gösteren bir "merdiven" çizilir.

Merdiven yapısı şu şekilde oluşturulur:
- $(x_n, x_n)$ noktasından $(x_n, f(x_n))$ noktasına dikey bir çizgi çizilir.
- $(x_n, f(x_n))$ noktasından $(f(x_n), f(x_n))$ noktasına yatay bir çizgi çizilir.

Bu işlem, fonksiyon eğrisi ile birim doğrusu arasında zikzak çizen ve iterasyon sürecini görsel olarak temsil eden bir yol oluşturur.

## Temel Kullanım

```python
from dynlib.plot import cobweb

# Basit bir fonksiyon kullanarak
def logistic(x, r=4.0):
    return r * x * (1 - x)

cobweb(
    f=logistic,
    x0=0.1,      # başlangıç koşulu
    steps=50,     # iterasyon sayısı
    xlim=(0, 1),  # x-ekseni sınırları
)
```

## Dynlib Modelleriyle Çalışma

Örümcek ağı grafikleri dynlib modelleriyle sorunsuz çalışır:

```python
from dynlib import setup
from dynlib.plot import cobweb

model = """
inline:
[model]
type="map"
name="Logistic Map"

[states]
x=0.1

[params]
r=4.0

[equations.rhs]
x = "r * x * (1 - x)"
"""

sim = setup(model, stepper="map")
cobweb(
    f=sim.model,  # Modeli doğrudan iletin
    x0=0.1,
    xlim=(0, 1),
    steps=50,
)
```
DSL'i `setup()` (veya `build()`) fonksiyonuna satır içi (inline) olarak iletirken, dynlib'in bunu bir dosya yolu yerine gömülü bir model tanımı olarak algılaması için dizenin başına `inline:` ekleyin.

## Temel Parametreler

### Fonksiyon Belirtimi
- `f`: İterasyon yapılacak fonksiyon veya model. Şunlar olabilir:
  - Çağrılabilir bir fonksiyon `f(x)` veya `f(x, r)`
  - `map()` metoduna sahip bir dynlib Model nesnesi
  - Bir Sim nesnesi (`sim.model` kullanılır)

### İterasyon Kontrolü
- `x0`: İterasyon için başlangıç değeri
- `steps`: Çizilecek iterasyon adımı sayısı (varsayılan: 50)
- `t0`: Başlangıç zaman indeksi (varsayılan: 0.0)
- `dt`: Zaman adımı boyutu (varsayılan: 1.0)

### Modele Özgü Seçenekler
- `state`: Çok boyutlu modeller için, hangi durum değişkeninin kullanılacağını belirtir (isim veya indeks ile)
- `fixed`: Sabit parametre/durum değerleri sözlüğü (dictionary)
- `r`: 'r' parametresi için geçersiz kılma değeri (çatallanma analizinde yaygındır)

### Çizim Stili
- `xlim`/`ylim`: Eksen sınırları (belirtilmezse otomatik hesaplanır)
- `color`: Fonksiyon eğrisinin rengi
- `identity_color`: Birim doğrusunun (y=x) rengi
- `stair_color`: İterasyon merdiveninin rengi
- `lw`: Fonksiyon eğrisinin çizgi genişliği
- `stair_lw`: Merdivenin çizgi genişliği
- `alpha`: Şeffaflık

### Etiketler ve Görünüm
- `xlabel`/`ylabel`: Eksen etiketleri
- `title`: Grafik başlığı
- `legend`: Lejandın gösterilip gösterilmeyeceği (varsayılan: True)

## Örümcek Ağı Grafiklerini Yorumlama

### Sabit Noktalar
Sabit noktalar $x = f(x)$ eşitliğinin sağlandığı yerlerde oluşur. Örümcek ağı grafiğinde bunlar, fonksiyon eğrisi ile birim doğrusunun kesişim noktaları olarak görünür.

### Kararlılık (Stabilite)
- **Kararlı sabit noktalar**: Merdiven, sabit noktaya doğru içeriye spiraller çizer.
- **Kararsız sabit noktalar**: Merdiven, sabit noktadan dışarıya doğru spiraller çizer.

### Periyodik Yörüngeler
Periyodik davranış, örümcek ağı grafiğinde kapalı döngüler olarak görünür. Örneğin, periyot-2 yörüngesi, merdivenin tekrar tekrar takip ettiği dikdörtgen bir yol oluşturur.

### Kaos
Kaotik davranış, merdivenin düzenli bir desene oturmaması ve genellikle grafiğin bölgelerini yoğun bir şekilde doldurmasıyla kendini gösterir.

## İleri Seviye Örnekler

### Çoklu Parametre Analizi
```python
# Farklı r değerlerini analiz etme
r_values = [2.5, 3.2, 3.5, 4.0]

for r in r_values:
    cobweb(
        f=logistic,
        x0=0.1,
        r=r,  # r parametresini geçersiz kıl
        xlim=(0, 1),
        title=f"Lojistik Harita (r={r})",
    )
```

### Çok Durumlu Modeller (tek bir değişkeni yineleme)

Örümcek ağı grafikleri, altta yatan harita (map) birçok duruma sahip olsa bile tek bir durum yörüngesini görselleştirir. Takip etmek istediğiniz değişkeni `state` ile ayarlayın ve geri kalanını `fixed` aracılığıyla sabit tutun. Bu, diğer durumlar sağlanan değerlerde sabitlenirken sistemi seçilen durum için etkili bir şekilde 1B haritaya indirger.

```python
from dynlib import setup

multi_state_model = """
inline:
[model]
type = "map"
name = "Two-State Map"

[states]
x = 0.5
y = 1.0

[params]
a = 3.0

[equations.rhs]
x = "a * x * (1 - x) + 0.1 * y"
y = "0.9 * y"
"""

sim = setup(multi_state_model, stepper="map")
cobweb(
    f=sim.model,
    x0=sim.state["x"],
    state="x",
    fixed={"y": 1.0},
    steps=100,
)
```

Grafik hala yalnızca seçilen `state` değişkenini gösterdiğinden, örümcek ağı grafikleri tek boyutlu haritalarla sınırlı kalır; çok durumlu sistemler her seferinde bir koordinat kesiti alınarak görselleştirilir.

### Özel Stil Tanımlama
```python
cobweb(
    f=sim.model,
    x0=0.1,
    xlim=(0, 1),
    color="blue",
    identity_color="red",
    stair_color="green",
    stair_lw=1.5,
    alpha=0.8,
    title="Özel Stilli Örümcek Ağı Grafiği",
)
```

## Sınırlamalar

- Örümcek ağı grafikleri yalnızca 1B haritalar için tasarlanmıştır; çok durumlu modeller `state`/`fixed` kullanılarak tek bir iterasyon durumuna indirgenmelidir.
- `lag()` fonksiyonlarını kullanan modeller desteklenmez (güvenli bir şekilde değerlendirilemez).
- Ayrık harita modelleri gerektirir (`spec.kind == 'map'`).

## İlgili Fonksiyonlar

- `plot.return_map()`: Dönüş haritalarını çizer ($x_n$'e karşı $x_{n+1}$)
- `plot.series()`: İterasyonun zaman serisini çizer
- `plot.phase()`: Daha yüksek boyutlu sistemler için faz uzayı grafikleri