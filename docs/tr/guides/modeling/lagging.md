# Lag Sistem Tasarımı & Uygulaması

## Genel Bakış

Lag sistemi (gecikme sistemi), dynlib modellerinde geçmiş durum (state) değerlerine şu notasyonla erişim sağlar:
- `lag_<isim>()` - `<isim>` adlı durumun bir adım önceki değerine erişir
- `lag_<isim>(k)` - `<isim>` adlı durumun k adım önceki değerine erişir

**Temel Özellikler:**
- İsteğe bağlı aktivasyon (sadece lag uygulanan durumlar hafıza tüketir)
- O(1) dairesel tampon (circular buffer) erişimi (Numba uyumlu)
- Sadece başarılı ve onaylanmış (committed) adımlardan sonra sayılır (tampon büyümesinden, erken kesilmelerden ve devam etme/resume işlemlerinden etkilenmez)
- Hem ODE hem de map modelleriyle çalışır
- Özel ayrılmış runtime çalışma alanı (stepper ABI'si `runtime_ws` parametresi ile genişletilmiştir)

---

## DSL Sözdizimi

### Desteklenen Kullanım

```toml
[model]
type = "map"

[states]
x = 0.1

[params]
r = 3.5
alpha = 0.3

[equations.rhs]
# Mevcut ve lag (gecikmeli) uygulanmış durumları karıştır
x = "r * (alpha * x + (1 - alpha) * lag_x(1)) * (1 - x)"

# Bir adım geri gitmek için argümansız kısa yazım kullanımı
x = "r * (alpha * x + (1 - alpha) * lag_x()) * (1 - x)"
```

### Lag Derinlikleri

```toml
[aux]
# Birden fazla lag derinliği kullanımı - maksimum değer otomatik algılanır
delayed_diff = "x - lag_x(5)"

[equations.rhs]
x = "v + 0.1 * lag_x(2)"  # x için maksimum lag = 5 (aux kısmından gelir)
v = "-x - lag_v(3)"       # v için maksimum lag = 3
```

### Kısıtlamalar

**Sadece durum (state) değişkenleri için:**
```toml
[states]
x = 0.1

[params]
a = 2.0

[equations.rhs]
x = "lag_x(1)"   # Geçerli - x bir durum değişkenidir
x = "lag_a(1)"   # Hata - a bir parametredir, durum değişkeni değildir
```

**Lag argümanı bir tamsayı sabiti olmalıdır:**
```toml
 x = "lag_x(2)"       # Geçerli
 x = "lag_x(k)"       # Hata - k bir sabit sayı değil
 x = "lag_x(2 + 1)"   # Hata - ifade kullanımına izin verilmez
```

**Mantık sınırı:**
```toml
x = "lag_x(1000)"    # Hata - mantık sınırını (1000) aşıyor
```

---

## Yardımcı (Aux) Değişkenlere Lag Uygulama

**Yardımcı değişkenlere (aux) doğrudan lag UYGULANAMAZ.** Bunun yerine, ifadelerin içinde lag uygulanmış durumları (lagged states) kullanın:

### Desteklenmeyen Yöntem:
```toml
[aux]
energy = "0.5 * v^2 + 0.5 * k * x^2"

[equations.rhs]
v = "-x - 0.1 * lag_energy(1)"  # HATA: energy bir aux değişkenidir, state değildir
```

### Doğru Yaklaşım:
```toml
[aux]
energy = "0.5 * v^2 + 0.5 * k * x^2"  # Mevcut enerji (isteğe bağlı)

[equations.rhs]
# Lag uygulanmış enerjiyi lag uygulanmış durumlardan hesapla
v = "-x - 0.1 * (0.5 * lag_v(1)^2 + 0.5 * k * lag_x(1)^2)"
```

**Mantık:** Aux değişkenleri geçici ve türetilmiş değerlerdir. `energy` değişkenine lag uygulamak matematiksel olarak `energy` değerini geçmiş durum (state) verilerinden hesaplamakla aynıdır.

### Alternatif: Aux'u State'e Yükseltme

Eğer türetilmiş bir niceliğe sık sık lag uygulanması gerekiyorsa:

```toml
[states]
x = 0.1
v = 0.0
energy = 0.005  # Aux'tan state'e yükseltildi

[params]
k = 2.0

[equations.rhs]
x = "v"
v = "-k * x - 0.1 * lag_energy(1)"  # Temiz erişim
energy = "0.5 * v^2 + 0.5 * k * x^2"  # ODE olarak takip edilir
```

**Takas:** Sistemin boyutunu 1 artırır.

---

## Depolama Mimarisi

### RuntimeWorkspace Yapısı

Lag tamponları (buffers) özel bir `RuntimeWorkspace` NamedTuple içinde saklanır:

```python
RuntimeWorkspace = namedtuple(
    "RuntimeWorkspace",
    ["lag_ring", "lag_head", "lag_info"],
)
```

**Bileşenler:**
- `lag_ring`: Tüm dairesel tamponları saklayan bitişik dizi (veri tipi modelle eşleşir)
- `lag_head`: Her bir lag uygulanmış durum için mevcut baş (head) indekslerinin dizisi (int32)
- `lag_info`: (state_idx, depth, offset) bilgisini içeren (n_lagged_states, 3) boyutunda meta veri dizisi

### Dairesel Tampon Düzeni

Her lag uygulanmış durum `lag_ring` içinde bitişik bir segment alır:

```
lag_ring düzeni (bitişik):
┌──────────────────┬──────────────────┬─────────────┐
│ x için lag tamp. │ y için lag tamp. │ (kullanım-  │
│  (derinlik 5)    │  (derinlik 3)    │   dışı)     │
└──────────────────┴──────────────────┴─────────────┘
   offset=0           offset=5          son
```

**Tahsisat:**
- `k` derinliğine sahip her lag uygulanmış durum, `lag_ring` içinde `k` adet ardışık eleman alır
- Toplam `lag_ring` boyutu = tüm lag derinliklerinin toplamı
- `lag_head` her lag uygulanmış durum için bir girdiye sahiptir
- `j` numaralı lag durumu için `lag_info[j] = (state_idx, depth, offset)`

---

## Dairesel Tampon Mekaniği

### Erişim Modeli

`x` durumu için `lag_x(k)` örneğinde:
- `depth = 5` (maksimum lag)
- `offset = 0` (lag_ring[0]'dan başlar)
- `head_index = 0` (baş kısmı lag_head[0]'da)

**İndirgenmiş ifade:**
```python
runtime_ws.lag_ring[offset + ((runtime_ws.lag_head[head_index] - k) % depth)]
#  runtime_ws.lag_ring[0 + ((runtime_ws.lag_head[0] - k) % 5)]
```

### Başlatma (t=t0 anında)

```python
# Her lag uygulanmış durumu başlangıç koşulu (IC) ile doldur
for j, (state_idx, depth, offset) in enumerate(lag_info):
    value = y_curr[state_idx]
    runtime_ws.lag_ring[offset : offset + depth] = value
    runtime_ws.lag_head[j] = depth - 1  # baş son pozisyonda
```

**Neden depth-1?** Böylece ilk adım onaylandıktan (commit) sonra baş (head) 0'a döner.

### Onaylanmış Adım Sonrası Güncelleme

**KRİTİK:** Güncellemeler rededilen adımlardan veya tampon büyümelerinden sonra değil, SADECE başarılı adım onaylarından (commits) sonra gerçekleşir.

```python
# runner.py içinde, commit işleminden sonra:
for j in range(n_lagged_states):
    state_idx, depth, offset = lag_info[j]
    head = int(lag_head[j]) + 1
    if head >= depth:
        head = 0
    lag_head[j] = head
    lag_ring[offset + head] = y_curr[state_idx]
```

**x için derinlik=3 ile örnek izleme:**

```
Adım 0 (IC=0.1):
lag_ring = [0.1, 0.1, 0.1], head=2

Adım 1 (y_curr=0.2):
head = (2+1) % 3 = 0
lag_ring[0] = 0.2 → lag_ring = [0.2, 0.1, 0.1], head=0

Adım 2 (y_curr=0.3):
head = (0+1) % 3 = 1
lag_ring[1] = 0.3 → lag_ring = [0.2, 0.3, 0.1], head=1

Adım 3 (y_curr=0.4):
head = (1+1) % 3 = 2
lag_ring[2] = 0.4 → lag_ring = [0.2, 0.3, 0.4], head=2

Adım 3'te lag_x(1) erişimi:
lag_ring[0 + ((2 - 1) % 3)] = lag_ring[1] = 0.3 ✓ (adım 2 değeri)

Adım 3'te lag_x(2) erişimi:
lag_ring[0 + ((2 - 2) % 3)] = lag_ring[0] = 0.2 ✓ (adım 1 değeri)
```

---

## Güvenlik & Doğruluk

### Tampon Büyümesi (GROW_REC, GROW_EVT)

**Lag tamponları** kayıt/olay tamponu büyümesi sırasında **yeniden tahsis EDİLMEZ**:
- Runtime çalışma alanı boyutları lag derinliklerine göre belirlenir (modele özgüdür, yörüngeye bağlı değildir)
- Wrapper (sarmalayıcı) `rec`/`ev` tamponlarını iki katına çıkarır ama runtime çalışma alanını olduğu gibi bırakır
- Lag durumu, sisteme yeniden girişlerde (re-entry) korunur

### Erken Kesilmeler (STEPFAIL, NAN_DETECTED, USER_BREAK)

- Runner, kesilmeden **önce** durumu commit eder (onaylar)
- Lag tamponları son başarılı commit işlemine kadar olan değerleri içerir
- Devam etme (Resume) işlemi, tam lag durumunu geri yüklemek için `workspace_seed` kullanır

### Devam Etme (Resume) & Anlık Görüntüler (Snapshots)

- `RuntimeWorkspace`, `snapshot_workspace()` ve `restore_workspace()` işlevlerini destekler
- Lag tamponları otomatik olarak çalışma alanı anlık görüntülerine dahil edilir
- Özel bir işlem yapılmasına gerek yoktur

**Doğruluk garantisi:** Lag'ler sadece onaylanmış (committed) adımlardan sonra sayılır.

---

## Örnek: Gecikmeli Lojistik Map

```toml
[model]
type = "map"
name = "Delayed Logistic Map"

[states]
x = 0.1

[params]
r = 3.8
alpha = 0.7  # Mevcut ve gecikmeli geri beslemenin karışımı

[equations.rhs]
# Gecikme-bağlaşımlı lojistik map
x = "r * (alpha * x + (1 - alpha) * lag_x()) * (1 - x)"

[sim]
t0 = 0.0
dt = 1.0
stepper = "map"
```

**Yürütme izi:**
```
n=0: x=0.1, lag_x(1)=0.1 (IC - Başlangıç Koşulu)
n=1: x = 3.8*(0.7*0.1 + 0.3*0.1)*(1-0.1) = 0.342
     lag_x(1)=0.1
n=2: x = 3.8*(0.7*0.342 + 0.3*0.1)*(1-0.342) = 0.627
     lag_x(1)=0.342
n=3: x = 3.8*(0.7*0.627 + 0.3*0.342)*(1-0.627) = 0.788
     lag_x(1)=0.627
...
```

---

## Performans

### Hafıza Maliyeti

`k` derinliğine sahip her lag uygulanmış durum için:
- Depolama: `RuntimeWorkspace.lag_ring` içinde `k * sizeof(dtype)` bayt
- Baş (Head) indeksleri: `RuntimeWorkspace.lag_head` içinde her durum için 1 int32
- Meta veri: `RuntimeWorkspace.lag_info` içinde her durum için 3 int32
- Örnek: 3 durum, derinlik 10, float64 → 3 * 10 * 8 = 240 bayt + ek yük

### İşlem Maliyeti

- **Adım başına:** Dairesel tampona O(n_lagged_states) yazma işlemi
- **Lag erişimi:** O(1) modulo + dizi indeksi
  - Numba, eğer derinlik 2'nin kuvvetiyse `(x - k) % depth` işlemini bitwise AND işlemine optimize eder

---

## Referanslar

- **Runtime Workspace:** `runtime/workspace.py`
- **DSL Spec:** `dsl/spec.py`
- **İfade İndirgeme (Lowering):** `compiler/codegen/rewrite.py`
- **Runner ABI:** `runtime/runner_api.py`
- **Stepper Base:** `steppers/base.py`