# DSL Model Dosyası Şablonu

Bu, TOML formatında DSL model dosyaları oluşturmak için hızlı bir başvuru şablonudur.
Mevcut tüm tabloları ve anahtarlarını listeler.

## Gerekli Tablolar

### [model]
- `type` (gerekli): "ode" | "map"
- `name` (isteğe bağlı): string (dize)
- `dtype` (isteğe bağlı): veri tipi, varsayılan "float64"

### [states]
- `state_name = initial_value` (sıralama, yetkili durum vektörü sırasını belirler)
- 8/3 gibi değer ifadeleri için tırnak işareti kullanın: "8/3".

## İsteğe Bağlı Tablolar

### [constants]
- `constant_name = value` (skalerler, sayısal ifadelere izin verilir, önceki sabitlere referans verebilir)
- Sabitlere atama yapılamaz ve bunlar salt okunur değişmezlerdir (literals).

### [params]
- `param_name = value` (skalerler veya diziler, model veri tipine dönüştürülür)

### Denklemler (bir form seçin veya karıştırın)
#### [equations.rhs] (durum-başına form)
- `state_name = "expression"`

#### [equations] (blok form)
- `expr = """dx = expression 
 dy = expression"""

### [equations.jacobian] (isteğe bağlı yoğun Jacobian)
- `expr = [[ "...", "...", ... ], [...], ...]` (ifadelerin n × n matrisi)
- Durum vektörü sırası, [states] bildirim sırasıdır (mod'lardan sonra). `state_names = (s0, s1, ...)` için, `expr[i][j]`, ∂f_state_names[i]/∂state_names[j] anlamına gelir. [states] sırasını değiştirmek anlamsal bir değişikliktir ve matris değişmezlerinin nasıl yorumlanacağını değiştirir.

### [aux]
- `aux_name = "expression"`

### [functions.function_name]
- `args = ["arg1", "arg2", ...]`
- `expr = "expression"

### [events.event_name]
- `phase` (isteğe bağlı): "pre" | "post" | "both" (varsayılan "post")
- `cond = "expression"
- `action = "expression"` veya `action.state_name = "expression"
- `tags` (isteğe bağlı): ["tag1", "tag2", ...]
- `log` (isteğe bağlı): ["var1", "var2", ...]

### [sim]
- `t0 = value`
- `t_end = value`
- `dt = value`
- `stepper = "euler" | "rk4" | ...`
- `record = true/false`
- `stepper_config = value (stepper'a özgü yapılandırma değerleri)`

### [meta]
- `title = "string"
- Meta tablosu şu anda yoksayılır. [meta] içine her şey yazılabilir.

## Özel Değişkenler
- `t` - Şimdiki zaman (tüm ifadelerde kullanılabilir)