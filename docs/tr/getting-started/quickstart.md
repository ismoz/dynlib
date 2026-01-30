# Hızlı Başlangıç

Bu sayfa, her dynlib kullanıcısının yaptığı ilk üç işlemde size yol gösterir: paketi kurmak, CLI sağlamasını yapmak ve Dynlib içinde tanımlı yerleşik (built-in) bir modeli çalıştırmak. Bundan sonra, kendi kendinize ilk modelinizi tanımlamak ve kendi model kataloğunuzu oluşturmanız için sizi [İlk Modeliniz](first-model.md) rehberine yönlendireceğiz.

## Kurulum

Yeni bir çalışma alanı oluşturuyorsanız önce uygun bir sanal ortam kurup etkinleştirin (`python -m venv .venv` ardından Linux/macOS'ta `source .venv/bin/activate`, Windows PowerShell'de `.\.venv\Scripts\Activate.ps1`). Sanal ortam aktifken paket paylaşımını ve komut satırı giriş noktalarını yüklemek için:

```bash
python -m pip install dynlib
```

Eğer kaynak kodu değiştirip geliştirmeye katkıda bulunmak isterseniz paketi github üzerinden indirip kurabilirsiniz:

```bash
git clone https://github.com/your-username/dynlib.git
cd dynlib
python -m pip install -e .
```

İşlem boyunca sanal ortamı açık tutun; böylece dynlib ve bağımlılıkları diğer projelerden izole kalır.

## CLI Doğrulama

Paketle gelen CLI, kurulumun başarılı olup olmadığını test etmek için kullanılabilir. Daha fazla örnek ve açıklama için [CLI rehberi](../guides/cli/cli.md)'ne bakın.

- `dynlib --version` veya `python -m dynlib.cli --version` kurulu paket sürümünü gösterir ve giriş noktalarının çalıştığını doğrular.
- `dynlib model validate builtin://ode/lorenz.toml` yerleşik Lorenz modelini derler ve doğrular. Diğer yerleşik modellere de `builtin://ode/` veya `builtin://map/` URI ifadeleri ile erişilebilir.
- `dynlib steppers list --kind ode --jit_capable` JIT destekli stepper'ları listeler, `--stiff` veya `--jacobian optional` gibi filtrelerle listelenecek stepper'ları sınırlayabilirsiniz.
- `dynlib cache path` derlenmiş runner'ların yerini söyler; önbelleği incelemek için `dynlib cache list`, temizlemek için `dynlib cache clear --dry_run` komutunu kullanın.

Her komut `--help` seçeneğiyle ek bayrakları ve parametreleri gösterir (örneğin `dynlib steppers list --help`), böylece çalışma zamanı komutlarını kaynak koduna girmeden keşfedebilirsiniz.

## Python'dan yerleşik bir modeli çalıştırma

`dynlib.setup` bir modeli derlerken stepper seçimini, JIT kullanımını ve veri kayıt stratejilerini tek yerden yönetmenizi sağlar. Aşağıdaki örnek Lorenz sistemini yükler, runner'ı derler, 15 zaman birimi boyunca simülasyonu yürütür ve `x`/`z` durumlarını çizim yardımcılarıyla grafik olarak çizdirir:

```python
from dynlib import setup
from dynlib.plot import fig, series, export

sim = setup(
    "builtin://ode/lorenz.toml",
    stepper="rk4",
)

sim.run(T=15.0, dt=0.01)
res = sim.results()

print("Recorded states:", res.state_names)
print("Recorded steps:", len(res))
print("Final z value:", res["z"][-1])

ax = fig.single()
series.plot(x=res.t, y=res["x"], ax=ax, label="x")
series.plot(x=res.t, y=res["z"], ax=ax, label="z", xlabel="time")
export.show()
```

`res` bir `ResultsView` nesnesidir. `res.t`, `res["state_name"]`, `res.event_names()` ve `res.to_pandas()` (pandas gerektirir) gibi yardımcılar veri kopyalamadan altta yatan bellek alanlarını (buffers) okur. Simülasyona devam etmek için `resume=True` seçeneğiyle tekrar `run()` komutu çalıştırılabilir. `assign()` komutu ile simülasyon ve model parametreleri değiştirilebilir. 

## Kendi modellerinizi kullanma

`builtin://` URI'leri paketle gelen ODE'leri (`lorenz`, `vanderpol`, `izhikevich` vb.) ve haritaları (`logistic`, `henon`, `standard` vb.) keşfetmeyi kolaylaştırır. Kendi TOML dosyalarınızı kullanmak için dynlib'in sunduğu URI çözümleme seçeneklerine bakın:

1. Inline (satır içi) TOML metinleri ile model oluşturma: [Inline (satır içi) modelleme](../guides/modeling/inline-models.md)
2. Mutlak veya göreli yollar (`/path/model.toml`, `my_model.toml`).
!!! note ".toml uzantısı ihmal edilebilir."
3. `proj://my_model.toml` gibi etiket tabanlı URI'ler: [Konfigürasyon](../guides/modeling/config-file.md) ve [Kayıt Defteri](../guides/modeling/model-registry.md).
4. Konfigürasyon dosyasını düzenlemeden yeni URI etiketleri eklemek için `DYN_MODEL_PATH=proj=/extra/models:/more` (Windows'ta `;` ile ayırın) gibi bir ortam değişkeni tanımlayın. `=` işaretinden önceki dize etiket adını, virgülle ayrılan yolları arama kökü olarak kullanır.

Konfigürasyon dosyanız şöyle görünebilir:

```toml
[tags]
proj = ["/Users/you/dynlib-models", "~/projects/other-models"]
tests = ["~/src/dynlib/tests/data/models"]

[cache]
cache_root = "/tmp/dynlib-cache"
```

`proj` etiketi tanımlandığında artık `setup("proj://decay.toml")`, `dynlib model validate proj://decay.toml` komutlarını çalıştırabilir veya diğer tanımlarda bu URI'yi referans verebilirsiniz.

## Sonraki adımlar

1. [İlk Modeliniz](first-model.md) ile kendi DSL tanımınızı yazmaya devam edin ve doğrulama akışını pekiştirin.
2. [Rehberler → Modelleme](../guides/modeling/index.md) ve [Rehberler → Simülasyon](../guides/simulation/index.md) üzerinden stepper'lar, kayıt seçenekleri ve runner yapılandırmaları hakkında detaylı bilgi edinin.
3. [Rehberler → Analiz](../guides/analysis/index.md) ve [Örnekler](../examples/index.md) sayfalarında çizim, sabit nokta analizi, Lyapunov üsteli hesaplayıcıları gibi ek araçları keşfedin.
4. `dynlib cache list`, `dynlib cache clear --dry_run` ve `dynlib model validate` gibi CLI komutlarıyla cache ve model doğrulamasını pratik edin; detaylı CLI kullanımı için [CLI rehberi](../guides/cli/cli.md)'ni inceleyin.
