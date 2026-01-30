# dynlib

Dynlib, **dinamik sistemleri modellemek, simüle etmek ve analiz etmek** için geliştirilmiş bir Python kütüphanesidir.  
Modeller TOML tabanlı bir DSL (alana özgü dil) ile tanımlanır ve ardından birleşik bir çalışma zamanı (runtime) üzerinden yürütülür. Böylece her deney için aynı Numpy/Matplotlib “tesisatını” yeniden yazmadan; çözücüleri, parametreleri ve analizleri hızlıca değiştirerek ilerleyebilirsiniz.

Dynlib ile bir modeli tanımlayıp ince ayar yapabilir, farklı çözücü/ayar kombinasyonlarını deneyebilir ve davranışı hızlıca görselleştirebilirsiniz. Derslerde ve gösterimlerde notebook’larla birlikte rahatça kullanılabilir. Oluşturduğunuz modelleri düzenli biçimde saklayabilir ve kolayca paylaşabilirsiniz.

## Proje durumu

Dynlib **alfa aşamasında** bir yazılımdır. API’ler değişebilir; sayısal köşe durumları veya hatalar ortaya çıkabilir. Bu nedenle, sonuçları doğrulayana kadar keşif amaçlı değerlendirin (ör. alternatif stepper’lar, daha sıkı toleranslar, daha küçük zaman adımları veya analitik kontroller). Şüpheli bir durum görürseniz, mümkün olduğunca küçük bir yeniden üretim örneği (minimal reproducer) ile issue açın.

## Öne çıkanlar

### Modelleme (TOML DSL)
- Bildirimsel (declarative) bir TOML tanımıyla **ODE**’leri ve **ayrık zamanlı haritaları (map)** tanımlayın.
- Denklemleri, parametreleri, başlangıç durumunu ve metaveriyi tutarlı bir formatta ifade edin.
- Uygun olduğunda **olaylar (events)**, **yardımcı değişkenler (aux)**, **fonksiyonlar/makrolar** ve **gecikmeli (lagging)** terimler desteği.
- Yerleşik **model kayıt sistemi** ve URI ile yükleme (ör. `builtin://...` modelleri).

### Simülasyon çalışma zamanı (runtime)
- Birden çok stepper ailesi:
  - ODE: Euler, RK4, RK45, Adams–Bashforth (AB2/AB3) ve örtük (implicit) yöntemler (örn. SDIRK/TR-BDF2).
  - Map: tamsayı güvenli modlar dâhil, ayrık sistemler için özel runner(lar).
- Yinelemeli iş akışları için runner varyantları ve oturum inceleme (session introspection) yardımcıları.
- Numba ile **JIT hızlandırma** (isteğe bağlı ama şiddetle önerilir) ve derlenmiş runner’lar için **disk önbelleği**.
- Uzun veya aşamalı simülasyonlar için **snapshot** ve **resume** desteği.
- Sonraki analizlere uygun olacak şekilde tasarlanmış seçici kayıt (recording) ve sonuç API’leri.

### Analiz
Dinamik sistemlerde sık kullanılan işler için yerleşik analiz yardımcıları:
- **Çatallanma (bifurcation)** ve ilgili post-processing araçları
- **Çekim havzaları (basins of attraction)** (auto/known varyantları)
- **Lyapunov üstel(leri)** analizi (runtime observer desteği dâhil)
- **Sabit nokta / denge (equilibria)** bulma
- **Manifold** izleme araçları (şimdilik 1B manifoldlarla sınırlı)
- **Homoklinik/Heteroklinik** yörünge izleme ve tespit
- **Parametre taraması (sweep)** yardımcıları ile trajektori ve post-analiz araçları

### Vektör alanları ve çizim (Matplotlib üzerinde)
Dynlib, ham Matplotlib kullanımındaki tekrar eden işleri azaltan ve dinamik sistemlere odaklı çizim yardımcıları içerir:
- Vektör alanı değerlendirme araçları ve **faz portresi (phase portrait)** yardımcıları
- **Basin**, **çatallanma diyagramı**, **manifold** ve genel dinamik çizimleri için modüller
- Daha üst seviye çizim kolaylıkları: **temalar**, **facets**, süsleme (decorations) ve dışa aktarma (export) yardımcıları
- Vektör alanı **animasyon** desteği

### CLI
Dynlib, model doğrulama, stepper listeleme ve önbellekleri inceleme gibi pratik işler için küçük bir CLI (Komut Satırı Arayüzü) ile gelir.  
Python API’sini kullanmak için CLI zorunlu değildir.

## Gereksinimler
- Python 3.10+
- Çizimler için Matplotlib
- Sayısal hesaplamalar için Numpy
- JIT yürütme için **Numba** şiddetle önerilir:
  - `python -m pip install numba`

## Kurulum
- `python -m pip install dynlib` veya
- Kaynaktan düzenlenebilir kurulum için: `python -m pip install -e .`

## Hızlı başlangıç

CLI’yi hızlıca kontrol edin ve paketle gelen bir modeli doğrulayın:

```bash
dynlib --version
dynlib model validate builtin://ode/lorenz.toml
```

Python’dan yerleşik bir modeli çalıştırın (Lorenz sistemi):

```python
from dynlib import setup
from dynlib.plot import fig, series, export

sim = setup("builtin://ode/lorenz.toml", stepper="rk4")

sim.run(T=15.0, dt=0.01)
res = sim.results()

print("States:", res.state_names)
print("Final z:", res["z"][-1])

ax = fig.single()
series.plot(x=res.t, y=res["x"], ax=ax, label="x")
series.plot(x=res.t, y=res["z"], ax=ax, label="z", xlabel="time")
export.show()
```

Sonraki adım: Kendi TOML modellerinizi tanımlamayı, URI etiketlerini (`proj://...`), kayıt (recording) seçeneklerini ve analiz iş akışlarını (basins, bifurcation, Lyapunov, fixed points) dokümantasyondan inceleyin.

## Dokümantasyon
Dokümantasyona şu adresten erişebilirsiniz: (#TODO: Bir bağlantı eklenecek ...)
