# Simülasyon Temelleri

Bu rehber, bir dynlib simülasyonunu çalıştırmanın iki temel aşamasını ele alır:
1. Modelinizi bir `FullModel` olarak **Derlemek (Compile)**.
2. Derlenen bu yapıyı bir `Sim` örneği ile **Yürütmek (Drive)**.

Ayrıca, bir simülasyonu hızlıca ayağa kaldırmak için kullanılan `setup()` kısayolunu da vurgular.

## 1. Bir `FullModel` Derleme ve İnceleme

Her dynlib simülasyonu `build()` (veya daha fazla kontrole ihtiyaç duyduğunuzda alt seviye derleyici giriş noktaları) ile başlar. `build()`, model spesifikasyonunu (URI, dosya yolu veya satır içi DSL) ve isteğe bağlı stepper, mod ve JIT bayraklarını (flags) alarak bir `FullModel` döndürür.

```python
from dynlib import build

model = build("my_model.toml", stepper="rk4", jit=True)
```

Bir `FullModel` şunları içerir:

- Derlenmiş çağrılabilir nesneler (`rhs`, `stepper`, `runner`, `events_pre`, `events_post` vb.).
- Metadata (`spec`, `stepper_name`, `workspace_sig`, veri tipi, simülasyon varsayılanları, guard'lar).
- Üretilen Python kodunu dışarı aktarmak için `export_sources()` ve durumları (states), parametreleri, yardımcı değişkenleri (aux) ve `[sim]` varsayılanlarını incelemek için `full_model.spec` gibi yardımcı metotlar.

Derlenen `model` standart bir Python nesnesi olduğundan, onu bir çalışma zamanına (runtime) entegre etmeden önce inceleyebilir veya yeniden kullanabilirsiniz. Örneğin:

- Seçilen entegratörü doğrulamak için `model.stepper_name` özelliğini kontrol edebilirsiniz.
- Neleri kaydedeceğinizi planlamak için `model.spec.states`, `model.spec.aux` ve `model.spec.params` alanlarını inceleyebilirsiniz.
- Hata ayıklama (debugging) amacıyla kaynak kodunu `model.export_sources("./compiled")` kullanarak dışa aktarabilirsiniz.

`FullModel`'i yalnızca derlenmiş bileşenleri incelemeniz veya yeniden kullanmanız gerektiğinde doğrudan kullanın. Çoğu iş akışında `FullModel` doğrudan `Sim`'e aktarılır.

## 2. `Sim` ile Simülasyon Çalıştırma

`Sim`, bir `FullModel`'i sarmalar ve devam ettirilebilir (resumable) oturum durumunu, sonuç tamponlarını (buffers), snapshot'ları ve ön ayar (preset) bankalarını yönetir.

```python
from dynlib import Sim

sim = Sim(model)
sim.config(record_interval=5, max_steps=2000)
sim.run(T=10.0, record=True)
results = sim.results()
```

Temel çalışma zamanı kavramları:

- `run(...)` simülasyonu başlatır. `dt`, `T`/`N`, `record`, `record_interval`, `max_steps` ve `record_vars` aracılığıyla seçici kayıt gibi `[sim]` varsayılanlarını geçersiz kılabilirsiniz (override).
- `Sim.config(...)`, her `run()` çağrısında ayarları tekrarlamamak için kalıcı varsayılanlar belirler.
- `Sim.assign(...)`, çalıştırmadan önce durumları/parametreleri günceller veya geçmişi temizler.
- `Sim.results()`, isimlendirilmiş erişim için bir `ResultsView` sağlarken, `Sim.raw_results()` düşük seviyeli `Results` tamponu aracılığıyla doğrudan dizi görünümleri (array views) sunar.
- `Sim.reset()`, isimlendirilmiş bir snapshot'a geri döner (ilk çalıştırmada otomatik olarak bir `"initial"` snapshot oluşturulur) ve kaydedilen geçmişi temizler.
- `Sim.create_snapshot(...)`, `list_snapshots()` ve `name_segment()`, birden fazla senaryo için tekrarlanabilir segmentler üzerinde kontrol sağlar.
- `run(resume=True)`, mevcut durumdan devam ederek simülasyon segmentlerinin kesintisiz birleştirilmesine olanak tanır.

`Sim`, DSL içindeki `[sim]` tablosuna saygı duyar, bu nedenle simülasyon parametreleri mantıklı varsayılanlara sahiptir. `run()` yalnızca belirttiğiniz şeyleri değiştirir.

## 3. `setup()` ile Hızlı Kurulum

Derlemek ve çalıştırmak için hızlı bir yol arıyorsanız `setup()` yardımcısını kullanın. Bu, `build()` ve `Sim()` işlemlerini tek bir çağrıda birleştirir, aynı varsayılanları uygular ve derlenmiş modele `sim.model` üzerinden erişim sağlar.

```python
from dynlib import setup

sim = setup("my_model.toml", stepper="rk4", jit=True)
sim.run(T=10.0)
print(sim.results().t)
```

`setup()` size çalışmaya hazır bir `Sim` verdiğinden, simülasyonlara başlamanın en hızlı yoludur. Açık `build()` + `Sim()` yaklaşımını, önce `FullModel` üzerinde işlem yapmanız gereken durumlar (örneğin; guard'ları incelemek, kaynakları dışa aktarmak veya yeniden kullanım için önbelleğe almak) için saklayın.

## İş Akışı İpuçları

- Bir kez derleyin (build) ve `FullModel`'i farklı senaryolar veya veri şekilleri için yeniden kullanın.
- Snapshot'lar, ön ayarlar (presets) veya tekrarlanan yapılandırmalar için bir `Sim` örneğini canlı tutun; sıfırlayıp yeniden kullanmak, yeniden derlemekten daha verimlidir.
- Çoğu ayar için `[sim]` varsayılanlarına güvenin, gerektiğinde `Sim.config()` ve hedefli `run()` geçersiz kılmalarını kullanın.
- Hızlı prototipleme, test etme veya demolar için `setup()` kullanın.
- `jit=True` seçeneğini yalnızca numba yüklüyse ve simülasyonlar derleme maliyetini haklı çıkaracak kadar uzunsa etkinleştirin. Kısa çalıştırmalar için, yorumlayıcıda (interpreter) kalmak üzere `jit=False` kullanın.