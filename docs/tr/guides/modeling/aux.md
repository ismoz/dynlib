# Yardımcı (Aux) Değişkenler

Yardımcı değişkenler (DSL dosyalarında `[aux]`), ara veya türetilmiş ifadeleri isimlendirmenizi sağlar; böylece mantıklarını tekrarlamadan denklemler, olaylar ve fonksiyonlar boyunca yeniden kullanabilirsiniz. Her adımdan (step) sonra durum güncellemelerinin ardından değerlendirilirler ve her ifade model çalışmadan önce Python'a derlenir.

## Sözdizimi (Sentaks)

```toml
[aux]
energy = "0.5 * mass * velocity^2"
gain = "baseline_gain * exp(-t / tau)"
# Döngü olmadığı sürece önceden tanımlanmış herhangi bir aux'u referans alabilirsiniz.
```

- Her değer bir dize (string) sabiti olmalıdır çünkü bir ifade olarak ayrıştırılır ve tür kontrolü yapılır. İfade `^` içerdiğinde, derleyici Python uyumluluğu için bunu `**` olarak yeniden yazar.
- Aux ifadeleri; durumlara (states), parametrelere, zamana (`t`), diğer aux değişkenlerine (döngü olmamalı), kullanıcı tanımlı fonksiyonlara, matematik makrolarına ve üreteç (generator) ifadelerine başvurabilir. Olay (event) bağlamlarında çalışmadıkları için olay makrolarını kullanamazlar.
- `t` (zaman), denklemlerde olduğu gibi kullanılabilir, bu nedenle zamana bağlı aux değişkenleri yazmak kolaydır.

## İfade Bağlamı

- **Durumlar (States)**: Yalnızca mevcut değerler (gecikme/lag notasyonu durumları açıkça referans almalıdır).
- **Parametreler**: `[params]` içinde tanımlanan sayısal sabitler.
- **Yardımcı (Auxiliary) değişkenler**: Dosyanın daha önceki kısımlarında tanımlanmış başka bir aux kullanabilirsiniz.
- **Yerleşik matematik fonksiyonları ve skaler makrolar**: `dynlib`'in DSL kütüphanesindeki her şey (`sin`, `cos`, `clip`, `approx`, üreteç ifadeleri vb.).
- **Kullanıcı tanımlı fonksiyonlar**: `[functions]` içinde beyan edildikten sonra isimleriyle çağrılabilirler (özyineleme/recursion yoktur).
- **Zaman (`t`)**: Tıpkı denklemlerde olduğu gibi ifadelerde her zaman mevcuttur.
- **Gecikme (Lag) notasyonu**: Yalnızca referans verilen sembol bir durum (state) ise kullanılabilir; aux değişkenleri doğrudan gecikmeli (lagged) olamaz.

## Örnek Kullanım

- Enerji, kuvvetler veya kayıt (logging) yardımcıları gibi paylaşılan ifadeleri hesaplamak için aux kullanın, böylece uzun hesaplamaları tekrarlamazsınız.
- İfadeleri yeniden yazmak yerine `cond`, `action` veya `log` listelerinde referans vererek aux'ları olaylarla (events) eşleştirin.
- Aux, aynı alt ifade birden fazla denklemde göründüğünde Jacobian girdilerini veya üreteç ifadelerini basitleştirebilir.

## Mod'lar (Mods) ile Etkileşim

Mod'lar; `remove` (kaldır), `replace` (değiştir), `add` (ekle) ve `set` (ayarla) fiillerini kullanarak yardımcı değişkenleri manipüle edebilir.

- `mod.remove.aux`, aux'un var olmasını gerektirir ve onu modelden basitçe düşürür.
- `mod.replace.aux`, aynı ismi koruyarak tanımı değiştirmenize olanak tanır.
- `mod.add.aux`, yeni bir yardımcı değişken ekler (isim zaten varsa hata verir).
- `mod.set.aux`, ifadeyi ekler veya günceller (eksikse oluşturur, varsa günceller).

Bu fiiller mod fiil sırasına (`remove → replace → add → set`) saygı duyar, böylece aynı mod içinde bir aux'u kaldırabilir ve daha sonra aynı isimle yeni bir tanım ekleyebilirsiniz.

## En İyi Uygulamalar

1. **Okunabilirliğe öncelik verin**: Aux isimlerini niceliği tanımlayacak şekilde verin (`kinetic_energy`, `normalized_voltage`), böylece karmaşıklığı gizlemek yerine yardımcı olsunlar.
2. **Döngülerden kaçının**: Aux değişkenleri arasında karşılıklı bağımlılıklar oluşturmayın; derleyici yönlü döngüsüz bir yapıyı (DAG) zorunlu kılar.
3. **İfadeleri odaklı tutun**: Gecikmeli (lagged) değerlere veya türevsel davranışlara ihtiyaç duymaya başlarsanız, niceliği aux mantığına aşırı yüklemek yerine bir duruma (state) terfi ettirmeyi düşünün.
4. **Niyeti belgeleyin**: Aux tanımının yanındaki kısa bir TOML yorumu, gelecekteki okuyuculara türetilen niceliğin neden var olduğunu hatırlatmak için yeterlidir.
5. **Dikkatli yeniden kullanım**: Aux, tekrarlanan matematik için harikadır, ancak matematiği sadece karmaşıklaştıran önemsiz takma adlarla modeli aşırı doldurmayın.