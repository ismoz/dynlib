# Model Registry

Model dosyaları disk üzerinde bulunur, ancak dynlib bu dosyalara mutlak yollarla (absolute paths) uğraşmak yerine sabit bir URI ile referans vermenizi sağlayan küçük bir registry (kayıt sistemi) aracılığıyla erişim sunar. Registry, dynlib ile birlikte gelen yerleşik (builtin) modelleri yönetir, kendi etiketlerinizi (tags) tanımlamanıza izin verir; ayrıca göreli yolları, parçaları (fragments) ve satır içi (inline) modelleri şeffaf bir şekilde çözümler.

## Yerleşik (Built-in) Modeller

Dynlib, `src/dynlib/models` paketini önceden yükler, böylece her zaman bir `builtin://` etiketi mevcuttur. Bu sayede, aşağıdaki modellerden herhangi birini `setup(...)`, `dynlib model validate` veya başka bir giriş noktasına, herhangi bir yapılandırma dosyası yazmadan ekleyebilirsiniz.

### Map modelleri
- `builtin://map/logistic`
- `builtin://map/henon`
- `builtin://map/henon2`
- `builtin://map/ikeda`
- `builtin://map/lozi`
- `builtin://map/sine`
- `builtin://map/standard`

### ODE modelleri
- `builtin://ode/duffing`
- `builtin://ode/eto-circular`
- `builtin://ode/expdecay`
- `builtin://ode/exp-if`
- `builtin://ode/fitzhugh-nagumo`
- `builtin://ode/hodgkin-huxley`
- `builtin://ode/izhikevich`
- `builtin://ode/leaky-if`
- `builtin://ode/lorenz`
- `builtin://ode/quadratic-if`
- `builtin://ode/resonate-if`
- `builtin://ode/vanderpol`

Registry, bu builtin dizinini otomatik olarak ekler (kesin mantık için `dynlib/compiler/paths.py` dosyasına bakınız), bu nedenle `builtin://` altındaki yollar hakkında endişelenmenize nadiren gerek kalır — sadece `builtin://ode/vanderpol` (`.toml` olmadan) yazın; dynlib dosyayı kontrol eder ve bulamazsa yararlı bir `ModelNotFoundError` hatası verir.

Bir builtin modeli incelemeniz veya doğrulamanız gerektiğinde CLI'ı kullanın:

```bash
dynlib model validate builtin://ode/expdecay
```

Bu komut URI'yi ayrıştırır, dosyayı çözer, DSL'i doğrular ve bir simülasyon çalıştırmadan önce herhangi bir ayrıştırma hatasını rapor eder.

## URI Kullanımı

`resolve_uri` (CLI ve `setup(...)` arkasındaki aynı mantık) çeşitli URI biçimlerini anlar:

1. **Satır içi (Inline) bildirimler**: Bir dizeye `inline:` ile başlarsanız, dynlib DSL parçasını hafızada tutar. Notebook'lar veya testlerdeki "kullan-at" modeller için kullanışlıdır.
2. **Etiket (Tag) URI'leri**: `TAG://relative/path`, `TAG` için kaydedilmiş herhangi bir kök dizin altında modeli arar. Builtin modeller `TAG=builtin` kullanır, ancak kendi etiketlerinizi özel dizinlerle ekleyebilirsiniz (bir sonraki bölüme bakın).
3. **Mutlak veya göreli yollar**: Gerçek bir dosya yolu da çalışır ve dynlib bunu normalize eder (`~`, ortam değişkenleri ve `.toml` uzantısını genişletir; mutlak yolu `cwd`'ye göre çözer).

Etiket URI'leri, bir dosya içindeki modları veya bölümleri seçmek için parçalar (fragments) da taşıyabilir:

```
builtin://ode/duffing#mod=odd
```

Ayrıştırıcı (parser), dosyayı çözümlemeden önce `#mod=...` kısmını ayırır ve parçayı geri verir; böylece derleyici `build(..., mods=[...])` çağrısı yaparken bunu kullanabilir.

`resolve_uri`, sağlanan yolun bir soneki yoksa `.toml` eklemeyi de dener, bu nedenle hem `builtin://ode/vanderpol` hem de `builtin://ode/vanderpol.toml` kabul edilir. Güvenlik kontrolleri, kayıtlı kökün dışına çıkılmasını engeller, yani `TAG://../foo.toml`, herhangi bir dosya okunmadan önce bir `PathTraversalError` hatası verir.

## Etiket Köklerini (Tag Roots) Yapılandırma

Dynlib, registry yapılandırmasını küçük bir TOML dosyasında tutar ve bunu ortam değişkenleri ile destekler:

- `DYNLIB_CONFIG` yapılandırma yolunu geçersiz kılar (varsayılan: Linux'ta `~/.config/dynlib/config.toml`, macOS'ta `~/Library/Application Support/dynlib/config.toml`, Windows'ta `%APPDATA%/dynlib/config.toml`).
- `DYN_MODEL_PATH`, etiket köklerini anında (on-the-fly) kabuk dostu bir sözdizimiyle eklemenizi sağlar. POSIX sistemlerinde `TAG=/yol/bir,/yol/iki:DIGER=/yol/uc` kullanın, Windows'ta etiketler arasında `;` kullanın.

Bir `config.toml` şuna benzer:

```toml
[paths]
myproj = ["~/repos/dynlib-models", "/opt/models"]
builtin = ["/custom/builtin/overrides"]  # dynlib builtin'lerini erişilebilir tut
cache_root = "~/Library/Caches/dynlib"
```

`load_config()` bu dosyayı ayrıştırır, ardından `DYN_MODEL_PATH` girdilerini başa ekler; böylece birden fazla dizin aynı etiketi paylaştığında ortam değişkeni kökleri kazanır. Bundan sonra, `builtin://` URI'lerinin başka bir yerde geçersiz kılsanız bile çözümlenmesini garanti etmek için builtin modeller klasörü `builtin` etiket listesine eklenir.

## Kendi Yollarınızı Ekleme

1. Bir etiket seçin (örneğin `myproj`) ve etiket URI yapısını yansıtan bir dizin ağacı oluşturun. Örneğin, `myproj://circuit/srn.toml`, `.../<root>/circuit/srn.toml` yoluna çözümlenir.
2. Kökü `DYNLIB_CONFIG` içindeki `[paths]` tablosuna ekleyin veya geçici geçersiz kılmalar için kabuğunuzda `DYN_MODEL_PATH="myproj=~/models/myproj"` ayarlayın.
3. Kurulumu `dynlib model validate myproj://circuit/srn` ile doğrulayın.
4. URI'yi betikler, `setup(...)` veya kendi araçlarınız içinde kullanın — dynlib etiketleri çözer, `.toml` dener ve eksik dosyaları aday listesiyle birlikte rapor eder.

Birden fazla registry yönetiyorsanız, `DYN_MODEL_PATH` girdilerinin yapılandırma dosyası girdilerine göre önceliği olduğunu ve bunların her ikisinin de builtin klasöründen önce arandığını unutmayın. Bu sıralama, `builtin` etiket listesinde daha önceye aynı yapıya sahip bir dizin koyarak `builtin://` modellerini geçersiz kılmanıza (override) olanak tanır.

## İpuçları

- Bir simülasyonu çalıştırmadan önce registry'nin dosyayı gerçekten çözdüğünden emin olmak için `dynlib model validate <uri>` çalıştırın.
- Ayrı `[[mods]]` tablolarında saklanan varyantlara sahip modelleri oluştururken `mytag://path/to/model#mod=variant` kullanın.
- Yeniden kullanılabilir modelleri bilinen bir etiket dizini altında tutun, böylece iş arkadaşlarınız yerel yapılandırmalarını düzenlemeden aynı URI'lere güvenebilirler.

Bu registry mevcutken, dynlib'in arama semantiğini sizin yerinize halletmesine izin vererek builtin modelleri, paylaşılan kütüphaneleri ve projeye özgü dosyaları özgürce karıştırabilirsiniz.