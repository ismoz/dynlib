# Simülasyon ([sim])

`[sim]` tablosu, herhangi bir model için varsayılan çalışma zamanı ayarlarını (knobs) belirtir. Alanların çoğu `Sim` arayüzüne (facade) (`Sim.run`) ve entegrasyonları veya haritaları (maps) yürüten runner'a beslenir. Bu tabloyu, modelinizi yükleyen kullanıcılar ve araçlar için "önerilen varsayılanlar" olarak düşünün.

## Bilinen Anahtarlar

- `t0` – Başlangıç zamanı. Varsayılan `0.0`'dır. Hem çalışma zamanı saatini hem de türetilmiş `Nominal dt / T` aritmetiğini besler.
- `t_end` – Sürekli (ODE benzeri) modeller için bitiş zamanı. Varsayılan `1.0`'dır. `Sim.run` arayüzü, `T` değerini geçersiz kılmadığınızda bunu kullanır.
- `dt` – Nominal zaman adımı (veya ayrık harita aralığı). Varsayılan `1e-2`'dir. Runner bunu, hem adımlama için kullanılan "nominal dt" olarak hem de `Sim.run` içinde `dt` atlandığında geri dönüş değeri olarak önbelleğe alır.
- `stepper` – Varsayılan stepper'ın (adımayıcı) adı (örn. ODE'ler için `"rk4"`, map'ler için `"map"`). Derleyici, model türüne göre mantıklı bir varsayılan seçer ancak belirli bir entegratörü sabitlemek için bunu buradan geçersiz kılabilirsiniz.
- `record` – Varsayılan kayıt davranışını kontrol eden boolean değer. Varsayılan `true`'dur. `Sim.run`, `record` olmadan çağrıldığında, durum/yardımcı örneklerinin biriktirilip biriktirilmeyeceğine bu değer karar verir.
- `atol`, `rtol` – Adaptif-stepper toleransları (varsayılan `1e-8`/`1e-5`). Bunlar yalnızca yapılandırılmış stepper, `Config` veri sınıfı aracılığıyla adaptif kontrol sunuyorsa geçerlidir.
- `max_steps` – Runner durmadan önceki maksimum adım sayısı (varsayılan `1_000_000`). Ayrık modeller için, `N` veya `T` sağlamadığınızda varsayılan iterasyon sayısı (`N`) olarak da hizmet eder.
- `stop` – Genellikle `post` aşamasında, her adımda değerlendirilen erken çıkış koşulu. Basit bir string `stop = "x > threshold"` veya bir tablo yazabilirsiniz:
  ```toml
  [sim.stop]
  cond = "max_energy > threshold"
  phase = "post"  # şimdilik sadece "post" destekleniyor
  ```
  Koşul doğru olduğunda runner `EARLY_EXIT` fırlatır ve `Results.status` bu durumu yansıtır.
- **Ekstra anahtarlar** – Diğer tüm girişler (yukarıda listelenen anahtarlar dışındaki her şey) `SimDefaults._stepper_defaults` içine iletilir ve otomatik olarak aktif stepper'ın `Config` alanlarına eşlenir. Bu, kodda stepper adını tekrarlamadan `tol`, `max_iter` veya herhangi bir enum tipli seçenek gibi `stepper-specific` (stepper'a özgü) varsayılanları geçirmenize olanak tanır.

## Çalıştırmalar (Runs) ile Etkileşim

1. **Çalışma zamanı geçersiz kılmaları (overrides) kazanır** – Her halka açık `Sim.run` argümanı (`t0`, `T`/`N`, `dt`, `max_steps`, `record` vb.) `[sim]` içindeki değerleri geçersiz kılar. Buna `**stepper_kwargs` aracılığıyla geçirilen ve `[sim]` ekstralarına göre önceliği olan stepper kwarg'ları da dahildir.
2. **Öncelik zinciri** – Stepper yapılandırma varsayılanları şuradan gelir: Stepper sınıfı varsayılanı < `[sim]` ekstra alanları < `Sim.run(... stepper_kwargs...)`. Bu birleştirme `ConfigMixin.default_config` aracılığıyla otomatik olarak gerçekleşir, bu nedenle anahtarları yalnızca bir kez bildirmeniz gerekir.
3. **Ayrık vs. Sürekli** – `Sim.run`, model türüne (`map` vs `ode`) bağlı olarak `[sim].t_end`/`dt` değerlerini farklı yorumlar. Map'ler için, `N`/`T` atlandığında `[sim].max_steps` varsayılan iterasyon sayısı olur; ODE'ler için `t_end` varsayılan entegrasyon ufkudur.
4. **Durdurma koşulu değerlendirmesi** – `[sim].stop` mevcut olduğunda, derleyici bunu uygun runner'a (`phase`'e göre pre/post) bağlar. Koşul, olağan ifade bağlamını (states, params, aux, functions, yerleşik makrolar) paylaşır ve taahhüt edilen her adımda kontrol edilir.
5. **Kayıt varsayılanları** – `[sim].record` yalnızca varsayılanı ayarlar; `Sim.run(record=False)` o belirli çağrı için loglamayı yine de devre dışı bırakır. Seçici kayıt (`record_vars`) kullanırsanız, seçim çalışma zamanına özgü olduğu için `[sim]` tablosu değişmeden kalır.

## Örnekler

```toml
[sim]
t0 = 0.0
t_end = 5.0
dt = 0.01
stepper = "rk4"
record = true
atol = 1e-9
rtol = 1e-6
max_steps = 500_000
tol = 1e-8        # stepper yapılandırmasına iletilen ekstra alan
stop = "energy > 100"  # erken çıkışı tetikler
```

Eğer stepper `tol` adında bir alana sahip bir `Config` sunuyorsa, bu değer varsayılanını geçersiz kılar ancak yine de daha sonra `Sim.run(t, dt, tol=acik_deger)` yoluyla geçersiz kılınabilir.

## En İyi Uygulamalar

1. **Yalnızca ihtiyacınız olan stepper'a özgü anahtarları ekleyin**, böylece ekstra listesi odaklı kalır.
2. **Erken çıkış koşulunuzu belgeleyin**; `[sim].stop` kullandığınızda okuyucular simülasyonun ne zaman ve neden iptal edildiğini bilmelidir.
3. **Varsayılanları fiziksel zaman birimleriyle (saniye, iterasyon) tutarlı tutun**, böylece modelinizi kullanan script'lerin sürekli olarak `t0`, `dt` veya `t_end` değerlerini geçersiz kılması gerekmez.
4. **`[sim]`i çoğunlukla güvenli varsayılanlar için kullanın**; deneyleri yeniden üretirken veya kaşifleri yönlendirirken `Sim.run` argümanlarına güvenin.