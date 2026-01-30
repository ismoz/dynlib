# Derlenmiş Kaynakları Dışa Aktarma

dynlib'in neler ürettiğine göz atmanız gerektiğinde, `FullModel.export_sources()` (ve kardeş yardımcısı `dynlib.compiler.build.export_model_sources`), mevcut her çağrılabilir nesneyi bir dizine yazar. Böylece bunları editörünüzde açabilir, linting çalıştırabilir veya bir simülasyon derlemesinin kaydını tutabilirsiniz.

Kaynakları dışa aktarmak şu durumlarda faydalıdır:

- DSL denklemlerinin Python fonksiyonlarına (rhs, steppers, events vb.) nasıl dönüştüğünü anlamak istediğinizde
- Stepper'lar, olaylar veya çözücü seçenekleri arasında neyin değiştiğini denetlemeniz (audit) gerektiğinde
- Regresyon testleri, demolar veya ekip arkadaşlarınızla paylaşmak için çıktılar (artifacts) hazırladığınızda

## Adım adım dışa aktarma iş akışı

```python
from dynlib import build

model = build("decay.toml", stepper="euler", jit=True)
files = model.export_sources("compiled_sources")
print(files["rhs"])
```

1. Bir `FullModel` elde etmek için `build(...)` veya `setup(...)` çağırın. Zaten bir `Sim`'iniz varsa, derlenmiş yapıya ulaşmak için `sim.model` kullanın.
2. Yazılabilir bir `output_dir` (çıktı dizini) iletin. Dizin otomatik olarak oluşturulur (`mkdir -p` mantığıyla).
3. Dönüş değeri, bileşen isimlerini yazılan dosyanın `Path` nesnesine eşleyen bir sözlüktür.
4. Farklı seçeneklerle her yeniden derleme (rebuild) yaptığınızda, kaynak farklarını karşılaştırmak için yeni bir dizine tekrar dışa aktarım yapabilirsiniz.

Bağımsız bir yardımcı tercih ederseniz, `dynlib.compiler.build` modülünden `export_model_sources`'ı içe aktarın ve `FullModel` örneğini iletin.

## Neler yazılır

Dışa aktarma işlemi, model nesnesi üzerindeki kaynak metnini taşıyan her derlenmiş bileşen için bir `.py` dosyası yazar:

- `rhs.py`, `events_pre.py`, `events_post.py`, `update_aux.py`
- `stepper.py` ve (mevcut olduğunda) `jvp.py`, `jacobian.py`, `inv_rhs.py`

Ek olarak, `model_info.txt` spesifikasyonu özetler: spec hash, tür (kind), stepper adı, veri tipi (dtype), listelenen durumlar/parametreler, RHS denklemleri ve herhangi bir olayın kısa bir önizlemesi (faz, guard ve eylem bloğunun ilk ~50 karakteri). Kodun yanında bu metadatanın olması, derlenmiş bir snapshot'ı DSL girdisiyle ilişkilendirmeyi kolaylaştırır.

`FullModel` kaynakları `disk_cache` veya seçtiğiniz stepper'dan bağımsız olarak tuttuğu için, derleyici önbelleğe alınmış yapıları yeniden kullansa veya `euler`, `rk4` ya da özel stepper'lar arasında geçiş yapsanız bile dışa aktarma çalışır. Dosyalar UTF-8 kodlamasıyla yazılır, böylece herhangi bir standart editörde açabilirsiniz.

## İpuçları

- Hata ayıklama için oluşturulan runner/stepper'ın bir kaydına ihtiyacınız varsa, derledikten sonra ve uzun simülasyonları çalıştırmadan önce dışa aktarın.
- Her dışa aktarma dizinini bir snapshot olarak değerlendirin: Regresyonları izlemek veya belirli bir seçeneğin (örneğin; `jit=True` vs `jit=False`) üretilen kodu nasıl etkilediğini belgelemek için saklayın.
- Yardımcı metot `Path` nesneleri döndürür, böylece içerikleri hemen okuyabilirsiniz (örneğin; `files["stepper"].read_text()`).