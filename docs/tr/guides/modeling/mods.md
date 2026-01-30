# Mods: Dynlib'de Model Modifikasyonları

Dynlib'deki Mods (modifikasyonlar), orijinal model dosyalarını değiştirmeden model spesifikasyonlarını dinamik olarak değiştirmenize olanak tanır. Bu özellik şu durumlar için yararlıdır:

- Model varyantları oluşturmak (örn. farklı parametre setleri, eklenen olaylar)
- Farklı model konfigürasyonlarını A/B testine tabi tutmak
- Mevcut modellere yamalar veya düzeltmeler uygulamak
- Daha basit temel modellerden karmaşık modeller oluşturmak

## Genel Bakış

Modlar, TOML tablo sözdizimi kullanılarak tanımlanır. Temel bir mod şöyle görünür:

```toml
[mod]
name = "modifikasyonum"
group = "opsiyonel_grup"
exclusive = false

[mod.remove.events]
names = ["silinecek_olay"]

[mod.add.events.yeni_olay]
phase = "post"
cond = "x > threshold"
action = "x = 0"

[mod.set.params]
alpha = 0.5
beta = 2.0
```

## Fiil (Verb) İşlemleri

Modlar, şu sırayla uygulanan dört ana işlemi (fiilleri) destekler: **remove (kaldır) → replace (değiştir) → add (ekle) → set (ayarla)**.

### 1. Remove (Kaldır)

Modelden mevcut bileşenleri kaldırır. Yalnızca halihazırda var olan bileşenler üzerinde çalışır.

**Desteklenen hedefler:** `events`, `params`, `aux`, `functions`

```toml
[mod]
name = "temizlik"

# Belirli olayları kaldır
[mod.remove.events]
names = ["debug_event", "gecici_tetikleyici"]

# Parametreleri kaldır
[mod.remove.params]
names = ["kullanilmayan_param", "eski_sabit"]

# Yardımcı değişkenleri (aux) kaldır
[mod.remove.aux]
names = ["gecici_degisken", "debug_cikti"]

# Fonksiyonları kaldır
[mod.remove.functions]
names = ["yardimci_fonk", "kullanilmayan_util"]
```

**Not:** `states` (durumlar) veya diğer desteklenmeyen hedefleri kaldırmaya çalışmak bir hata oluşturur.

### 2. Replace (Değiştir)

Mevcut bileşenleri yeni tanımlarla değiştirir. Bileşen zaten mevcut olmalıdır.

**Desteklenen hedefler:** `events`, `aux`, `functions`

**Not:** Parametre değerlerini güncellemek için `set.params` kullanın. Parametreleri tamamen değiştirmek için `remove` + `add` kullanın.

```toml
[mod]
name = "mantik_guncelleme"

# Bir olayı (event) değiştir
[mod.replace.events.mevcut_olay]
phase = "post"
cond = "x > yeni_esik"
action = "x = 0; counter = counter + 1"

# Yardımcı değişkenleri değiştir
[mod.replace.aux]
energy = "0.5 * m * v^2"  # Yeni ifade
power = "force * velocity"  # Yeni ifade

# Fonksiyonları değiştir
[mod.replace.functions.activation]
args = ["x", "gain", "offset"]
expr = "gain * tanh(x) + offset"
```

### 3. Add (Ekle)

Modele yeni bileşenler ekler. Bileşen daha önce mevcut olmamalıdır.

**Desteklenen hedefler:** `events`, `params`, `aux`, `functions`

```toml
[mod]
name = "ozellik_ekle"

# Yeni olaylar ekle
[mod.add.events.reset_mechanism]
phase = "post"
cond = "x > 10"
action = "x = 0"

[mod.add.events.spike_detector]
phase = "pre"
cond = "v > threshold"
action = "spike_count = spike_count + 1"
log = ["t"]  # Spike zamanlarını kaydet

# Yeni parametreler ekle
[mod.add.params]
gain = 2.5
offset = 0.1

# Yeni yardımcı değişkenler ekle
[mod.add.aux]
total_energy = "kinetic + potential"
efficiency = "output_power / input_power"

# Fonksiyonlar ekle
[mod.add.functions.sigmoid]
args = ["x"]
expr = "1 / (1 + exp(-x))"

[mod.add.functions.relu]
args = ["x"]
expr = "max(0, x)"
```

**Not:** `states` veya diğer desteklenmeyen hedefleri eklemeye çalışmak bir hata oluşturur.

### 4. Set (Ayarla)

Bileşen değerlerini ayarlar veya günceller. Bu bir "upsert" (güncelle veya ekle) işlemidir - yeni bileşenler oluşturabilir veya mevcut olanları güncelleyebilir.

**Desteklenen hedefler:** `states`, `params`, `aux`, `functions`

```toml
[mod]
name = "yapilandir"

# Durum (state) başlangıç değerlerini ayarla
[mod.set.states]
x = 5.0
y = -2.5

# Parametre değerlerini ayarla (zaten mevcut olmalı)
[mod.set.params]
alpha = 0.1
beta = 2.5

# Yardımcı değişkenleri ayarla (upsert - oluştur veya güncelle)
[mod.set.aux]
debug = "t"  # Yeni oluştur
energy = "0.5 * k * x^2"  # Mevcut olanı güncelle

# Fonksiyonları ayarla (upsert - oluştur veya güncelle)
[mod.set.functions.activation]
args = ["x"]
expr = "tanh(x)"  # Mevcut olanı güncelle

[mod.set.functions.new_func]
args = ["a", "b"]
expr = "a + b"  # Yeni oluştur
```

**Not:** `states` ve `params` için, `set` yalnızca mevcut değerleri günceller ve bileşen yoksa hata verir. Yeni parametreler oluşturmak için `add` kullanın.

## Olay (Event) Tanımlama Formatı

Modlardaki olaylar, model tanımlarıyla aynı TOML tablo formatını kullanır:

```toml
[mod.add.events.event_name]
phase = "pre" | "post"        # Koşulun ne zaman kontrol edileceği
cond = "expression"           # Tetiklenecek koşul
action = "code"               # Gerçekleştirilecek eylem (string)
log = ["var1", "var2"]        # Tetiklendiğinde kaydedilecek değişkenler (opsiyonel)

# Alternatif: anahtarlı eylem atamaları
[mod.add.events.event_name]
phase = "post"
cond = "x > 5"
action.dx = 1.0
action.dy = -0.5
log = ["t"]
```

## Fonksiyon Tanımlama Formatı

Fonksiyonlar `args` ve `expr` ile tanımlanır:

```toml
[mod.add.functions.function_name]
args = ["arg1", "arg2", "arg3"]  # Argüman isimleri dizisi
expr = "expression"              # Fonksiyon gövdesi ifadesi
```

## Grup ve Dışlayıcılık (Exclusivity)

Çakışan modifikasyonları önlemek için modlar gruplandırılabilir:

```toml
# Aynı gruptaki birbirini dışlayan (exclusive) modlar
[mods.fast]
name = "fast"
group = "speed"
exclusive = true

[mods.fast.set.params]
dt = 0.01

[mods.slow]
name = "slow"
group = "speed"
exclusive = true

[mods.slow.set.params]
dt = 0.1

# "speed" grubundan aynı anda sadece bir mod aktif olabilir
```

## Modları Kullanma

### Dosyalardan Mod Yükleme

Modlar genellikle TOML dosyalarında saklanır ve URI aracılığıyla yüklenir:

```python
from dynlib import build

# Modlarla birlikte modeli yükle
model = build("model.toml", mods=["mods.toml#mod=variant1"])
```

#### Tek Mod Dosyası

```toml
[mod]
name = "parameter_tune"
group = "tuning"

[mod.set.params]
alpha = 0.5
beta = 2.0

[mod.add.events.monitor]
phase = "post"
cond = "t % 1.0 == 0"
action = ""
log = ["x", "y"]
```

#### Tek Dosyada Birden Fazla Mod

```toml
[mods.variant1]
name = "variant1"

[mods.variant1.set.params]
gain = 1.0

[mods.variant2]
name = "variant2"

[mods.variant2.set.params]
gain = 2.0

[mods.variant2.add.events.noise]
phase = "pre"
cond = "true"
action = "x = x + 0.1 * randn()"
```

#### Modlar için URI Desenleri

- `"mods.toml"` - Dosyadan tek bir mod yükle
- `"mods.toml#mod=variant1"` - Koleksiyondan belirli bir modu yükle
- `"inline: [mod]\nname='patch'\n..."` - Satır içi (inline) mod tanımı

### Programatik Kullanım

İleri düzey kullanım durumları için, modları programatik olarak oluşturabilirsiniz:

```python
from dynlib.compiler.mods import ModSpec, apply_mods_v2
from dynlib.dsl.parser import parse_model_v2

# Modu Python sözlüğü (dict) olarak tanımla (yukarıdaki TOML'a eşdeğer)
mod = ModSpec(
    name="programmatic_mod",
    set={
        "params": {"alpha": 0.5},
        "aux": {"debug": "t"}
    }
)

# Ayrıştırılmış (parsed) modele uygula
normal = parse_model_v2(model_toml_string)
modified = apply_mods_v2(normal, [mod])
```

## Hata Yönetimi

Modlar işlemleri doğrular ve şunlar için `ModelLoadError` verir:

- **Desteklenmeyen hedefler**: Desteklenmeyen hedefler üzerinde işlem yapmaya çalışmak (örn. `add.states`, `remove.states`, `replace.params`)
- **Mevcut olmayan bileşenler**: Mevcut olmayan bileşenleri kaldırmaya veya değiştirmeye çalışmak
- **Yinelenen bileşenler**: Zaten mevcut olan bileşenleri eklemeye çalışmak
- **Bilinmeyen bileşenler**: Mevcut olmayan `states` veya `params` için değer ayarlamaya çalışmak
- **Geçersiz veri tipleri**: `aux` değişkenleri için string olmayan değerler kullanmak
- **Hatalı biçimlendirilmiş tanımlar**: Geçersiz fonksiyon tanımları (eksik args, expr vb.)
- **Grup dışlayıcılık ihlalleri**: Aynı gruptan birden fazla exclusive modu etkinleştirmek

### Fiile Göre Desteklenen Hedefler

| Fiil (Verb) | Desteklenen Hedefler                 | Notlar                                   |
|-------------|--------------------------------------|------------------------------------------|
| `remove`    | `events`, `params`, `aux`, `functions` | Bileşen mevcut olmalı                    |
| `replace`   | `events`, `aux`, `functions`         | Bileşen mevcut olmalı                    |
| `add`       | `events`, `params`, `aux`, `functions` | Bileşen mevcut olmamalı                  |
| `set`       | `states`, `params`, `aux`, `functions` | States/params mevcut olmalı; aux/functions upsert edilir |

### Yaygın Hatalar ve Çözümler

**Hata: `add.states: unsupported target`**
- **Neden**: Modlar aracılığıyla yeni durum (state) değişkenleri eklemeye çalışmak.
- **Çözüm**: Durumlar (states) dinamik olarak eklenemez. Bunları temel modelde tanımlayın.

**Hata: `remove.states: unsupported target`**
- **Neden**: Modlar aracılığıyla durum değişkenlerini kaldırmaya çalışmak.
- **Çözüm**: Durumlar kaldırılamaz. Onlar model yapısının temelidir.

**Hata: `replace.params: unsupported target`**
- **Neden**: `replace` fiilini kullanarak parametreleri değiştirmeye çalışmak.
- **Çözüm**: Değerleri güncellemek için `set.params` kullanın veya `remove.params` + `add.params` sırasını kullanın.

**Hata: `add.params.x: param already exists`**
- **Neden**: Zaten mevcut olan bir parametreyi eklemeye çalışmak.
- **Çözüm**: Değeri güncellemek için `set.params` kullanın veya değiştirmek istiyorsanız önce `remove.params` kullanın.

### Doğrulama, Sessiz Hataları Önler

Doğrulama iyileştirmelerinden önce, desteklenmeyen işlemler sessizce başarısız olur ve kullanıcıların modlarının neden çalışmadığı konusunda kafası karışırdı. Şimdi, desteklenmeyen hedefleri kullanma girişimi, desteklenen hedeflerin bir listesiyle birlikte anında net bir hata verecektir.

## En İyi Uygulamalar

1. **Açıklayıcı isimler kullanın**: Modlara, amaçlarını belirten açık, açıklayıcı isimler verin.

2. **İlgili modları gruplayın**: Karşılıklı olarak birbirini dışlayan seçenekler (örn. farklı parametre setleri) için grupları kullanın.

3. **Kapsamlı test edin**: Modlar model davranışını önemli ölçüde değiştirebilir - sonuçları dikkatlice doğrulayın.

4. **Modları belgeleyin**: Her modun ne yaptığını ve nedenini açıklayan yorumlar ekleyin.

5. **Sürüm kontrolü**: Mod dosyalarını modellerinizle birlikte sürüm kontrolü altında tutun.

6. **Basit başlayın**: Temel `set` işlemleriyle başlayın, ardından daha karmaşık `add`/`replace`/`remove` kombinasyonlarına ilerleyin.

## Örnekler

### Parametre Çalışması Modları

```toml
[mods.low_gain]
name = "low_gain"
group = "gain_study"
exclusive = true

[mods.low_gain.set.params]
k = 0.1

[mods.high_gain]
name = "high_gain"
group = "gain_study"
exclusive = true

[mods.high_gain.set.params]
k = 10.0
```

### Parametreleri Zamanla Değişen İfadelerle Değiştirme

Bu örnek, sabit bir parametrenin zamana bağlı bir yardımcı değişkene nasıl dönüştürüleceğini gösterir:

```toml
[mods.sine_drive]
name = "sine_drive"

# Not: Modlar aracılığıyla durum (state) eklenemez (hata verir)
# Bunun yerine, yardımcı gürültü değişkenleri ekleyin

[mods.sine_drive.add.params]
freq = 1000.0
Vmax = 4.0

# V'yi zamanla değişen bir yardımcı değişken olarak ekle
[mods.sine_drive.add.aux]
V = "Vmax*sin(2*pi*freq*t)"
```

### Model Varyantları

```toml
[mod.stochastic]
name = "stochastic"

# Not: Modlar aracılığıyla durum eklenemez (hata verir)
# Bunun yerine, yardımcı gürültü değişkenleri ekleyin

[mod.stochastic.add.params]
sigma = 0.1

[mod.stochastic.add.aux]
noise = "sigma * randn()"
noisy_x = "x + noise"
```

### Hata Ayıklama (Debugging) Yardımları

```toml
[mod.debug]
name = "debug"

[mod.debug.add.aux]
debug_t = "t"
debug_x = "x"
debug_dx = "dx_dt"

[mod.debug.add.events.log_state]
phase = "post"
cond = "t % 1.0 == 0"  # Saniyede bir kaydet
action = ""  # Eylem yok, sadece kayıt
log = ["debug_t", "debug_x", "debug_dx"]
```

### Fonksiyon Varyantları

Aşağıdaki örnekte h(phi,N) fonksiyonunun iki varyantı vardır. Biri aşağıdaki gibi seçilebilir:

```python
sim = setup("memristive_chua#mod=odd")
```

```toml
[model]
type="ode"
name="Flux Controlled Memristor"

[states]
phi=0.1

[params]
a=0.08
b=1
c=0.83
d=1.8
N=0
freq=1.0
Vmax=4.0

[functions.W]
args=["phi"]
expr="a+b*tanh(phi)**2"

[aux]
V = "Vmax*sin(2*pi*freq*t)"
I = "W(phi)*V"

[equations.rhs]
phi="c*V-d*h(phi,N)"

### MODS:

[mods.odd.add.functions]
h = {args = ["phi","N"], expr="""
phi if N==0 else phi-sum(sign(phi+(2*j-1))+sign(phi-(2*j-1)) for j in range(1,N+1))
"""}

[mods.even.add.functions]
h = {args = ["phi","N"], expr = """
phi-sign(phi) if N==0 else phi-sign(phi)-sum(sign(phi+2*j)+sign(phi-2*j) for j in range(1,N+1))
"""}
```

## Fiil (Verb) Sırası Önemlidir

Fiillerin şu sabit sırayla uygulandığını unutmayın:

1. **Remove** - Önce bileşenleri kaldır
2. **Replace** - Mevcut bileşenleri değiştir
3. **Add** - Yeni bileşenler ekle
4. **Set** - Değerleri ayarla/güncelle

Bu, eski bileşeni kaldır → farklı isimle yeni bir tane ekle, veya bileşeni değiştir → sonra parametrelerini modifiye et gibi karmaşık dönüşümlere olanak tanır.