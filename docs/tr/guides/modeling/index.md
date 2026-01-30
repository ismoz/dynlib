# Modelleme rehberi

Bu rehber; TOML DSL'ini ve spesifikasyonları yapılandırılmış, okunabilir ve yeniden üretilebilir tutan yardımcı araçları kullanarak dynlib modellerini nasıl bildireceğinizi, genişleteceğinizi ve ince ayar yapacağınızı açıklar. DSL genel bakışı ile başlayın, ardından yeniden kullanılabilir bileşenleri ve bir spesifikasyonu çalışma zamanına bağlayan iş akışı yardımcılarını keşfedin.

## Model yapısının temelleri

- [DSL temelleri](dsl-basics.md) — yazabileceğiniz her tabloyu listeleyen kurallı TOML şablonu (`[model]`, `[states]`, `[params]`, `[constants]`, `[equations]`, `[aux]`, `[functions]`, `[events]`, `[sim]`, vb.).
- [Denklemler](equations.md) — `rhs` (sağ taraf), blok, ters ve Jacobian formlarını karşılaştırır, her birini hangi bağlamların kabul ettiğini açıklar ve sağ taraflarınızı düzenli tutmak için en iyi uygulamaları ana hatlarıyla belirtir.
- [Matematik ve makrolar](math-and-macros.md) — her ifadenin içinde bulunan yerleşik matematik fonksiyonlarını, skaler makroları (`clip`, `approx`, `relu`, vb.), üreteç ifadelerini ve olay yardımcı programlarını listeler.
- [Üçlü (Ternary) `if`](ternary-if.md) — Python tarzı üçlü ifadenin, sizi tam `if`/`else` bloklarına çekmeden küçük dallanmaları nasıl kolaylaştırdığını gösterir.
- [Model kayıt defteri](model-registry.md) — etiket URI'lerini (`builtin://`, özel etiketler, satır içi modeller), `DYNLIB_CONFIG`/`DYN_MODEL_PATH` davranışını ve kayıt yollarını doğrulayan veya geçersiz kılan CLI yardımcılarını tanımlar.

## Yeniden kullanılabilir yapı taşları

- [Auxiliary (Yardımcı) değişkenler](aux.md) — türetilmiş ifadeleri isimlendirin, böylece matematiği tekrarlamadan denklemler, olaylar veya Jacobian'lar arasında paylaşabilirsiniz.
- [DSL fonksiyonları](functions.md) — argümanları, ifade gövdeleri ve DSL'i bildirimsel tutan temiz çağrı yerleri olan yeniden kullanılabilir fonksiyonlar tanımlayın.
- [Olaylar (Events)](events.md) — `cond` (koşul), `action` (eylem) ve günlükleme meta verilerini `pre`/`post` aşamalarına bağlayın, olay makrolarını kullanın ve hızlı yol çalıştırıcılarını kararsızlaştırmadan olay günlüklerini yönetin.
- [Gecikme (Lagging)](lagging.md) — `lag_<state>(k)` yardımcılarını etkinleştirin, tampon derinliğini kontrol edin ve gecikmeli durumların ODE'ler, haritalar ve NumPy uyumlu çalışma zamanları ile nasıl etkileşime girdiğini anlayın.
- [Satır içi (Inline) modeller](inline-models.md) — modelleri tamamen testlerin veya not defterlerinin (notebooks) içinde prototiplemek için bir Python dizesine bir TOML parçası gömün.

## İş akışı yardımcıları

- [Yapılandırma dosyası](config-file.md) — kayıt yollarını, önbellek köklerini ve eklenti davranışını `~/.config/dynlib/config.toml` veya `DYNLIB_CONFIG` ortam değişkeni aracılığıyla özelleştirin.
- [Modlar (Mods)](mods.md) — `remove`, `replace`, `add` ve `set` eylemleriyle modelleri dinamik olarak yamalayın; böylece temel spesifikasyonu klonlamadan varyantlar oluşturabilir, parametreleri geçersiz kılabilir veya yeni olaylar ekleyebilirsiniz.
- [Ön ayarlar (Presets)](presets.md) — yeniden kullanılabilir durum/parametre anlık görüntülerini yakalayın, bunları diskten yükleyin/kaydedin ve simülasyon bankası aracılığıyla yeniden oynatın.
- [Simülasyon varsayılanları](sim.md) — `[sim]` tablosunu belgeleyin, `Sim.run` geçersiz kılmalarıyla nasıl birleştiğini açıklayın ve erken çıkış, kayıt ve tolerans ayarlarını vurgulayın.