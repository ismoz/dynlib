# DSL Fonksiyonları

Kullanıcı tanımlı fonksiyonlar, tekrar kullanılabilir mantığı bir kez kapsülleyip; denklemlerden, aux (yardımcı) değişkenlerden, olaylardan (events) ve diğer fonksiyonlardan (özyineleme olmadan) çağırmanıza olanak tanır. İfadelerin yönetilebilir kalmasını sağlar ve yaygın hesaplamaları parametrelerle kullanmanıza izin verir.

## Sözdizimi (Syntax)

```toml
[functions.sigmoid]
args = ["x", "gain", "offset"]
expr = "gain / (1 + exp(-x)) + offset"
```

- `args` parametre adlarının bulunduğu bir dizidir; yalnızca basit tanımlayıcılar kullanın.
- `expr`, diğer ifadelerle aynı makro genişletmeli DSL/Python kuralları kullanılarak değerlendirilen bir dize ifadesidir (`^` → `**`, döngülere derlenen üreteç ifadeleri vb.).
- Fonksiyonlar bir dönüş tipi bildirmez; ifadenin değeri dönüş değeridir.

## Fonksiyonların İçindeki Bağlam

- **Argümanlar:** Fonksiyonlar kendi parametrelerini değişken olarak kullanabilir (`x`, `gain`).
- **Zaman (`t`):** Yalnızca çevreleyen bağlam zamanı sağlıyorsa kullanılabilir (örneğin, saf bir matematik yardımcısından değil, bir denklemden veya aux'tan çağrıldığında).
- **Durumlar ve parametreler (States & parameters):** Modelde mevcutlarsa isimleriyle referans verilebilir.
- **Aux değişkenleri:** `[aux]` içinde tanımlanan aux değişkenlerini çağırabilir, ancak bağımlılık döngüleri oluşturmadığınızdan emin olun.
- **Diğer kullanıcı tanımlı fonksiyonlar:** Normal şekilde çağırabilirsiniz, ancak özyinelemeyi (recursion) önleyin.
- **Yerleşik matematik ve makrolar:** `sin`, `cos`, `clip`, `approx`, üreteç ifadeleri vb. kullanın.
- **Gecikme (Lag) notasyonu:** Fonksiyon, `lag_` erişiminin geçerli olduğu bir yerde çağrılırsa izin verilir (yalnızca durumlar için).
- **Olay (Event) makroları:** Fonksiyonların içinde doğrudan erişilemez; bunları bunun yerine olay koşullarında kullanın.

## Fonksiyonları Çağırma

- Fonksiyonlara tıpkı yerleşikler gibi adıyla referans verin: `sigmoid(x, gain, offset)`.
- Argüman olarak ifadeler, durumlar, aux veya sabit değerler (literals) geçirebilirsiniz.
- Fonksiyonlar; denklemler veya olaylar arasında kullanılan tekrarlanan matematiği, koşullu mantığı veya karmaşık dönüşümleri basitleştirebilir.
- Jacobian'lar, günlükleme (logging) ifadeleri veya özel aktivasyon şekilleri için araçları ayırmak adına yardımcı fonksiyonlar kullanın.

## Modlar ve Fonksiyonlar

Fonksiyon tanımları modlar aracılığıyla değiştirilebilir:

- `mod.remove.functions` adlandırılmış fonksiyonları siler (bileşen zaten mevcut olmalıdır).
- `mod.replace.functions.name` tanımlayıcıyı korurken gövdeyi üzerine yazar.
- `mod.add.functions.name` yeni bir fonksiyon ekler (zaten varsa başarısız olur).
- `mod.set.functions.name` fonksiyon tanımını ekler veya günceller (oluşturur veya günceller).

Bu eylemler küresel sil/değiştir/ekle/ayarla (rem/replace/add/set) sırasına uyar, böylece başka bir sürüm eklemeden önce kaldırabilir veya değiştirebilirsiniz.

## En İyi Uygulamalar

1. **Yardımcıları açıklayıcı bir şekilde adlandırın** (`functions.normalize_input`), böylece sonraki denklemler niyeti açıkça belli eder.
2. **Argüman listelerini kısa tutun**; çok fazla argüman, fonksiyonun bunun yerine aux veya durum demetleri üzerinde çalışması gerektiğini gösterir.
3. **Yan etkilerden kaçının**—fonksiyonlar yalnızca değer döndürmeli ve durumları veya aux değişkenlerini değiştirmemelidir.
4. **Varsayımları belgeleyin** (örneğin, beklenen aralıklar); entegratörlerin kısıtlamalardan haberdar olması için yakınlardaki yorumlarda veya belgelerde belirtin.
5. **Mantıklı bir şekilde yeniden kullanın**: Okunabilirliğe yardımcı olmadıkça veya karmaşık matematiği gizlemedikçe basit ifadeleri sarmalamayın.