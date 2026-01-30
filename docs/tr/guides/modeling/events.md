# Olay Yönetimi (Event Handling)

Olaylar (events), simülasyon sırasında `cond` (koşul) ifadesi doğru olduğunda eylemleri çalıştırarak ve günlük (log) tutarak model koşullarına tepki vermenizi sağlar. Olaylar adımdan önce (`phase = "pre"`), sonra (`phase = "post"`) veya her ikisinde de çalışır ve hata ayıklama veya analiz için teşhisleri yakalamak üzere günlük kaydı ekleyebilirsiniz.

NOT: Analiz için bazı hızlı yol (fast-path) çalıştırıcıları, olayı olmayan modelleri tercih eder.

## Temel Şablon

```toml
[events.reset_on_threshold]
phase = "post"
cond = "x > threshold"
action = "x = 0; spike_count = spike_count + 1"
log = ["t", "x", "spike_count"]
```

- `phase`, koşulun ne zaman değerlendirileceğini kontrol eder (varsayılan `post`).
- `cond`, boolean döndüren bir dize olmalıdır. Her zaman adımında yeniden değerlendirilir.
- `action`, atama ifadelerinden oluşan bir dizedir; netlik için atamaları `action.var = "expr"` şeklinde de kapsamlandırabilirsiniz.
- `log` isteğe bağlıdır ve olay tetiklendiğinde değerleri kaydedilen değişkenleri listeler.

## Koşul Bağlamı

- **States/parameters**: Bildirilen herhangi bir duruma veya parametreye referans verin.
- **Aux değişkenleri**: Koşulları okunabilir tutmak için `[aux]`'tan türetilmiş ifadeleri yeniden kullanın.
- **Zaman (`t`)**: Zaman tabanlı tetikleyiciler için her zaman mevcuttur.
- **Kullanıcı tanımlı fonksiyonlar**: Bunları tıpkı denklemlerde veya aux tanımlarında olduğu gibi çağırın.
- **Yerleşik matematik & skaler makrolar**: `sin`, `cos`, `clip`, `approx` vb.
- **Generator (Üreteç) kapsamaları**: İndirgemelere (reductions) ihtiyacınız olduğunda `sum(...)` veya `prod(...)` kullanın.
- **Olay makroları**: `cross_up`, `cross_down`, `changed`, `in_interval`, `enters_interval`, `leaves_interval`, `increasing`, `decreasing` ve `cross_either` gecikmeli durum değerlerini otomatik olarak karşılaştırır, böylece manuel `lag_` ifadeleri yazmanıza gerek kalmaz.
- **Lag (Gecikme) notasyonu**: Koşullar içinde `lag_state(k)` çağırabilirsiniz, ancak bu yalnızca gerçek durum değişkenleri (states) içindir, asla aux için kullanılamaz.

### Olay Makroları Örneği

```toml
[events.detect_spike]
phase = "pre"
cond = "cross_up(v, 1.0)"
action = "spike_count += 1"
```

Makro, gecikmeli erişimi sizin için halleder, böylece koşul, `v` eşiği aşağıdan yukarıya geçtiği anda ekstra kayıt tutmaya gerek kalmadan tetiklenir.

## Eylem (Action) Detayları

- Eylemler; durumları, parametreleri (izin veriliyorsa), aux veya izleyici (tracker) değişkenlerini yeni ifadeler atayarak değiştirebilir.
- Birden fazla ifadeyi ayırmak için noktalı virgül kullanın veya `action.var = "expr"` sözdizimi ile bireysel atamalar tanımlayın.
- Eylemler, koşul değerlendirildikten sonra atomik (bütünsel) olarak çalıştırılır; yan etkiler bir sonraki zaman adımı için model durumunun bir parçası olur.
- Eylemleri kısa tutun; ağır hesaplamalar aux değişkenlerine veya yardımcı fonksiyonlara aittir.

## Günlük Kaydı (Logging)

- `log`, olay her tetiklendiğinde listelenen ifadeleri yakalar.
- Günlükler; durumları, aux veya hesaplanan ifadeleri içerebilir (`log = ["t", "energy", "debug_flag"]`).
- Olay zamanlamasını incelemek, sahte tetikleyicileri tespit etmek veya analiz için sayaçları kaydetmek amacıyla günlük kaydını kullanın.

## Mod'lar ile Olay Yaşam Döngüsü

Mod'lar, başka yerlerde mevcut olan aynı fiilleri kullanarak olayları manipüle edebilir:

- `mod.remove.events`, mevcut olayları isme göre siler.
- `mod.replace.events.name`, zaten var olan bir olayın phase/cond/action/log değerlerini yeniden tanımlar.
- `mod.add.events.new_name`, yeni bir olay ekler (isim zaten varsa hata verir).
- `mod.set.events` desteklenmez; bunun yerine `add` veya `replace` kullanın.

Her zaman küresel fiil sırasını hatırlayın: remove → replace → add → set; böylece aynı tanımlayıcıya sahip başka bir olay eklemeden önce bir olayı kaldırabilir veya değiştirebilirsiniz.

## En iyi uygulamalar

1.  **Olayları açıklayıcı bir şekilde isimlendirin** (`events.detect_refractory_start`), böylece niyetleri açık olur.
2.  **Karmaşık yüklemleri (predicates) aux değişkenlerine veya fonksiyonlara çıkarın**, böylece koşulları okunabilir tutarsınız.
3.  **Eylemleri küçük ve belirlenimci (deterministik) tutun** ve türetilmiş miktarları satır içi (inlined) ifadeler yerine aux aracılığıyla güncellemeyi tercih edin.
4.  **Geçişleri veya değişiklikleri izlerken olay makrolarını kullanın**, böylece manuel lag (gecikme) kaydı tutmaktan kaçınırsınız.
5.  **Bilinçli günlük tutun**—çok fazla günlük girişi performansı düşürebilir, bu nedenle yalnızca hata ayıklama veya analiz bağlamları için ihtiyacınız olanı kaydedin.