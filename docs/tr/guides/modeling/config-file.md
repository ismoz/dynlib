# Model kayıt yapılandırması

Dynlib, model kayıt defteri detaylarını küçük bir TOML dosyasında tutar; böylece dizinlere etiketler atayabilir, yerleşik modelleri geçersiz kılabilir (override) ve JIT önbelleğinin nerede yaşayacağını kontrol edebilirsiniz. `load_config()`, dosyayı `DYN_MODEL_PATH` girişleri ile birleştirir ve nihai `PathConfig` (etiket haritası + isteğe bağlı önbellek kökü) nesnesini her çözümleyiciye teslim eder.

## Yapılandırma dosyası nerede bulunur

- **Varsayılan yol** (`DYNLIB_CONFIG` ayarlanmadığında):
  - Linux/Unix: `${XDG_CONFIG_HOME:-~/.config}/dynlib/config.toml`
  - macOS: `~/Library/Application Support/dynlib/config.toml`
  - Windows: `%APPDATA%/dynlib/config.toml`
- **Geçersiz Kılma (Override):** `DYNLIB_CONFIG` değişkenini özel bir TOML dosyasına ayarlayın, dynlib bunun yerine o dosyayı yükler.
- **Eksik dosya:** `load_config()` sessizce boş bir yapılandırma döndürür, böylece dynlib `DYNLIB_CONFIG` veya `DYN_MODEL_PATH` geçersiz kılmalarıyla çalışmaya devam eder.

## Dosya formatı

```toml
[paths]
custom = ["~/repos/dynlib-models", "/opt/dynlib/models"]
builtin = ["~/custom/builtin"] # Bu, yerleşik model yolunu genişletir, yerini almaz.

cache_root = "~/Library/Caches/dynlib"
# veya alternatif form
[cache]
root = "~/Library/Caches/dynlib"
```

- `[paths]`, bir etiket adını (`builtin` veya `custom` gibi) bir veya daha fazla dizin köküne eşler. Her giriş bir dize veya dizeler listesi olabilir. Dynlib, `custom://circuit/srn` gibi bir URI'yi her kökü sırayla arayarak çözümler.
- `[cache]` tablosu (veya üst düzey `cache_root`), `resolve_cache_root()` işlevine iletilen JIT önbellek konumunu sabitlemenizi sağlar. Mutlak veya `~/` ile genişletilmiş bir yol sağlayın; dynlib kullanmadan önce yazılabilirliğini doğrular.
- Dosya; TOML bozuksa, `[paths]` dize olmayan girişler içeriyorsa veya gerekli bir değer eksikse `ConfigError` ile korunur.

## Ortam değişkeni geçersiz kılmaları (Environment overrides)

- `DYN_MODEL_PATH`, dosyayı düzenlemeden etiket köklerini başa eklemenizi sağlar. Sözdizimi POSIX üzerinde `TAG=/path/one:/path/two` ve Windows üzerinde `TAG=C:\path1;TAG2=C:\path2` şeklindedir.
- Girişler bir haritaya ayrıştırılır ve yapılandırma dosyasında beyan edilenlerden önce eklenir, böylece birden fazla dizin aynı etiketi paylaştığında ortam yolları kazanır.
- Dynlib, etiketi yeniden tanımlasanız bile `builtin://` URI'lerinin her zaman çözümlenmesini sağlamak için yerleşik modeller klasörünü tüm geçersiz kılmalardan sonra `builtin` etiket listesine ekler.

## Çözümleme sırası ve davranışı

1. `load_config()` TOML dosyasını (varsa) yükler ve etiket haritasını oluşturur.
2. `DYN_MODEL_PATH` girişleri her etiketin başına eklenir, böylece geçici geçersiz kılmaların dosya destekli kökleri gölgelemesine izin verilir.
3. Yerleşik modeller dizini, etiketi yeniden tanımladığınızda bile `builtin://` URI'lerinin mevcut olmasını garanti etmek için `builtin` etiketine eklenir.
4. Sonuçta ortaya çıkan `PathConfig`, çözümleyici yardımcıları tarafından önbelleğe alınır, böylece CLI'yı veya işlemi yeniden başlatmak disk üzerindeki değişiklikleri yeniden okur.

Dynlib bir etiketi veya istenen model yolunu bulamadığında, denediği adayları listeleyerek bir `ConfigError` (bilinmeyen etiket) veya `ModelNotFoundError` (dosya araması başarısız) hatası yükseltir.

## Sorun giderme ipuçları

- Bir simülasyonu çalıştırmadan önce kaydın bir modeli çözümlediğini doğrulamak için `dynlib model validate <uri>` komutunu çalıştırın.
- Yazılabilir dizinleri işaret ettiklerinden emin olmak için `DYNLIB_CONFIG` ve `DYN_MODEL_PATH` değişkenlerini inceleyin.
- Yapılandırmadaki önbellek kökü yazılamaz durumdaysa, dynlib platform varsayılanına geri döner (Linux: `~/.cache/dynlib`, macOS: `~/Library/Caches/dynlib`, Windows: `%LOCALAPPDATA%/dynlib/Cache`). Bu gerçekleştiğinde bir `RuntimeWarning` yayar.