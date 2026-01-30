# Session state & yazdırma

`Sim`, derlenmiş modeli kapatıp yeniden kurmadan çalışan oturumu inceleyebilmen veya değiştirebilmen için canlı bir `SessionState` tutar. Bu rehber; “oturumda şu an ne var?”, bunu güvenli şekilde nasıl değiştirirsin ve ihtiyaç olduğunda alttaki DSL denklemlerini nasıl yazdırırsın gibi konulardaki yardımcıları anlatır. Daha detaylı introspection için `export_sources()` kullanımına bak.

## Session state inceleme

- **`session_state_summary()`** mevcut zamanı (`t`), adım sayısını, nominal `dt`’yi, stepper adını, durumu ve `resume=True`’ın hâlâ mümkün olup olmadığını (`can_resume`/`reason`) döndürür. Ayrıca saklanan stepper config özetini (digest) içerir; böylece gelecekteki bir `run()` çağrısının aynı konfigürasyonu kullanıp kullanmayacağını anlayabilirsin.
- **`can_resume()`** ve `compat_check()` çalışma zamanı pin’lerinin (spec hash, stepper, workspace imzası, dtype, dynlib sürümü) aktif model ile eşleşip eşleşmediğini kontrol eder; böylece bir oturumu devam ettirmeden önce doğrulayabilirsin.
- Bu özet, `run()` bittikten hemen sonra veya bir `Snapshot` oluşturmadan önce dashboard/log üretmek için çok kullanışlıdır.

### Örnek

```python
summary = sim.session_state_summary()
print(f"Current time {summary['t']} (step {summary['step']})")
if not summary["can_resume"]:
    print("Resume unavailable:", summary["reason"])
```

## Çalışma sırasında session değerlerini değiştirme

`Sim.assign(...)` state ve parametreleri isme göre günceller; zamanı, workspace’i veya sonuç geçmişini değiştirmez (sen açıkça temizlemezsen). Metot; isimleri modelden çıkarır, “bunu mu demek istedin?” önerileri verir, değerleri model dtype’ına çevirir ve bilinmeyen değişkenleri değiştirmeye çalışırsan hata verir.

- Bir mapping veya keyword ver: `sim.assign(v=-65.0, I=12.0)` aynı session zamanını korur ama bir sonraki run’ın başlangıç değerlerini değiştirir.
- `clear_history=True` ile birikmiş results/segment geçmişini düşürürsün, ama `SessionState` korunur. Bu, daha önceki snapshot’a dönmeden yeni bir kayıt başlatmak istediğinde işe yarar.

```python
sim.assign({"v": -70.0, "I": 10.0}, clear_history=True)
sim.run(T=1.0, record=True)
```

## Sayısal değerleri dışa aktarma ve yazdırma

Session’dan, model spec’inden veya isimli bir snapshot’tan state/parametre vektörlerini ya da sözlüklerini okumak için birkaç yardımcı var:

- **`state_vector(source='session', copy=True)`** / **`param_vector(...)`** DSL’deki tanım sırasına göre 1-boyutlu NumPy dizileri döndürür. `source` `"session"`, `"model"` veya `"snapshot"` olabilir. `copy=False` verirsen alttaki saklama alanına bir view alırsın; değerleri doğrudan değiştirmek istiyorsan kullanışlıdır.
- **`state_dict(...)`** / **`param_dict(...)`** yukarıdaki dizileri `isim -> float` sözlüğüne çevirir; hızlı log veya JSON çıktısı için pratik.
- **`state(name)`** / **`param(name)`** mevcut session’dan tek bir skalar okur; ismi yanlış yazarsan yardımcı öneriler verir.

```python
print(sim.state_dict())  # session değerleri dict olarak
print(sim.param_vector(source="model"))  # model varsayılanları ndarray olarak
snapshot_states = sim.state_vector(source="snapshot", snapshot="initial")

print(sim.state("v"))  # tek değer okuma
```

Bu yardımcılar; run ortasında debug yazdırma, UI paneli üretme veya snapshot/preset yanına checkpoint metadatası koyma gibi işlerde çok işe yarar.

## DSL denklemlerini yazdırma

`FullModel.print_equations()` üretilmiş runner’ı değil, orijinal DSL tanımını yansıtır. Böylece dokümana veya log’a “güzel yazdırılmış” denklemler ekleyebilirsin.

- `tables` hangi TOML table’larının yazdırılacağını seçer: varsayılan `"equations"` ana dinamikleri gösterir; ama diğer kayıtlı table’ları da isteyebilirsin (`"equations.inverse"`, `"equations.jacobian"` vb.) veya `tables="all"` ile hepsini yazdırabilirsin.
- `include_headers` bölüm başlıklarının yazdırılıp yazdırılmayacağını kontrol eder; `file=` ise çıktıyı yazılabilir herhangi bir stream’e yönlendirmeni sağlar.
- `FullModel.available_equation_tables()` tüm kayıtlı anahtarları listeler; ne isteyebileceğini kontrol edersin.

```python
model.print_equations()  # ana denklemleri başlıklarla yazdırır

with open("equations.txt", "w") as out:
    model.print_equations(tables="all", include_headers=False, file=out)

print(FullModel.available_equation_tables())
```

`print_equations()` ile `session_state_summary()` veya `state_dict()`’i birlikte kullanmak, sayısal state’i onu üreten denklemlerle bağlayan tekrarlanabilir raporlar üretmeni kolaylaştırır.
