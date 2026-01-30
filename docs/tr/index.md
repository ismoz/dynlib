# Dynlib

Dynlib, dinamik sistemleri tanımlamak, simüle etmek ve analiz etmek için geliştirilmiş bir simülasyon kütüphanesidir. Modeller, TOML tabanlı bir DSL (Alana Özgü Dil) ile tanımlanır; böylece aynı model tanımı hem simülasyon hem de analiz için kullanılabilir. Şu anda dynlib, ayrık zamanlı haritaları (maps) ve adi diferansiyel denklemleri (ODE'ler) desteklemekte olup, giderek büyüyen bir analiz araçları setine sahiptir.

Dynlib'in amacı, dinamik sistemlerle çalışmayı pratik ve tekrarlanabilir hale getirmektir. Uygulama detayları elbette önemlidir; ancak çözücüleri, durum dizilerini (state arrays), parametre taramalarını, çizimleri ve veri yönetimini nasıl bağlayacağınızı bir kez öğrendikten sonra, her yeni model için bu tesisatı tekrar kurmak bir darboğaza dönüşür. Dynlib, tekrarlayan mekanik işleri soyutlar, böylece sayısal yöntemler ve konfigürasyon üzerindeki kontrolü kaybetmeden modele, parametre rejimlerine ve yorumlamaya odaklanabilirsiniz.

Sadece "NumPy + Matplotlib" kullanılarak yapılan simülasyonlarla karşılaştırıldığında dynlib, simülasyon ve analiz sürecini son derece pratikleştirir. Dynlib ile model kodunu yeniden yazmaksızın sayısal çözüm metodunu değiştirebilir; simülasyon ve analizleri çok kısa kodlarla gerçekleştirebilirsiniz. Bu özellikle dinamik sistemler derslerinin öğretim süreçlerinde faydalı olmaktadır. Öğrenciler döngü oluşturma, numpy ve matplotlib kullanımı, nümerik çözüm metodunu gerçekleştirme gibi pek çok detayla boğuşurken bir yandan da dinamik sistemleri öğretmeye çalışmak çoğu zaman asıl amaçtan saparak dersi bir python kursuna dönüştürmektedir. Dynlib ile ise bu tür gerçekleştirim detaylarıyla boğuşmadan hızlıca model oluşturularak simülasyon ve analize yoğunlaşılabilir. Basitleştirilmiş çizim araçları ile kolaylıkla grafikler elde edilebilir.

Simülasyonun yanı sıra dynlib; çatallanma diyagramları (bifürkasyon), çekim havzaları (basins of attraction), Lyapunov üssü tahmini, manifold izleme ve sabit nokta tespiti gibi yerleşik analiz araçlarını içerir. Ayrıca performans için JIT derleme (Just-In-Time Compilation) ve disk önbellekleme (caching) imkanı sağlar. Birden fazla sayısal sayısal çözüm ailesini (Euler, RK4, RK45, TR-BDF2A, vb.) destekler. Model kütüphanesi ile yeni modeller oluşturabilir ve bu modellere kolaylıkla erişebilirsiniz. Simülasyonları durdurup tekrar başlatabilir (resume) ve herhangi bir anda simülasyon durumunun kaydını alabilirsiniz (snapshot). Ayrıca basit bir CLI (komut satırı arayüzü) ile komut satırından da dynlib ile ilgili işlemler yapabilirsiniz.

!!! warning "Uyarı : Dynlib şu anda aktif geliştirme aşamasındadır. API'ler (uygulama arayüzleri) değişebilir ve sonuçları etkileyen hatalar veya sayısal uç durumlar olabilir. Dynlib'i araştırma veya kritik kararlar için kullanıyorsanız, sonuçları güvenilir referanslar ile (örneğin alternatif çözücüler, daha küçük adım boyutları, analitik kontroller) doğrulayın ve lütfen karşılaştığınız sorunları bildirin."

## Kullanılan Terimler
Dynlib çalışmasını anlayabilmek için aşağıdaki terimlerin bilinmesi gerekmektedir. Bazı terimler tamamen dynlib kütüphanesi özgü terimlerdir.

- **Harita/Map:** Ayrık zamanlı dinamik sistemler (lojistik harita gibi).
- **ODE:** Adi diferansiyel denklem sistemi (ing. Ordinary Differential Equations).
- **DSL:** Alana özgü dil (ing. Domain-Specific Language). Modelleri tanımlama için TOML formatında rahat anaşılır bir dil.
- **JIT:** Just-in-time derlemenin kısaltması. Numba paketi yardımıyla python kodlarının derlenmesi ile daha yüksek hızlarda simülasyon / analiz yapma imkanı sağlar.
- **Stepper:** Simülasyonların sonraki adımını (step) hesaplamakla görevli program. ODE stepper'ları sayısal integrasyon metodlarını (Euler, RK4, RK45 gibi) içerir.
- **Runner:** Simülasyonlar koşu (run) olarak nitelendirilirse, runner bir sümülasyon koşusunu yerine getiren program olarak açıklanabilir. Dynlib birden fazla özelleşmiş runner programı içerir. Runner ve stepper kombinasyonu JIT ile derlenebildiği için bir kernel gibi düşünülebilir.
- **Wrapper:** JIT ile derlenmiş runner'lar ile yapılabilecek işlemler kısıtlı olduğu için her runner programı bir wrapper kontrolünde kullanılır. Runner yetersiz kalırsa salt python programı olan wrapper kontrolü devralır.
- **API:** Uygulama programlama arayüzü (ing. Application Program Interface). Bir programı nasıl çağırmanız gerektiğini belirler.
- **CLI:** Komut satırı arayüzü (ing. Command-Line Interface).
- **URI:** Model TOML dosyalarının adresini temsil eder (ing. Uniform Resource Identifier).
- **RHS:** Bir eşitliğin sağ tarafta kalan kısmı (ing. Right-Hand Side).
- **Anlık görüntü (snapshot):** Simülasyonun herhangi bir anında durum değişkenleri, parametre değerleri ve simülasyon parametreleri gibi tüm değerlerin o anki kaydının alınıp saklanması.

## Öne Çıkanlar

- **Modeli bir kez tanımlayın (TOML DSL):** ODE’leri veya ayrık haritaları tek bir TOML tanımıyla yazın ve aynı modeli tüm simülasyonlarda ve analizlerde kullanın.
[Modelleme](guides/modeling/index.md)

- **Simülasyonları kolayca çalıştırın:** Detaylarla boğuşmadan pratik bir şekilde simülasyon oluşturun. Stepper seçerek kolayca nümerik çözüm metodunu değiştirin. İsterseniz JIT hızlandırma kullanın ve hızlı derleme için disk önbelleği kullanın. Anlık görüntü (snapshot) kaydedin ve simülasyona kaldığınız yerden devam edin (resume).
[Simülasyon](guides/simulation/index.md) / [Çalışma Zamanı](examples/runtime.md)

- **Temel analizleri yapın:** Çatallanma diyagramı, çekim havzası, Lyapunov üsteli hesabı, sabit nokta bulma, manifold izleme ve parametre taramaları için yerleşik araçları kullanın. Ayrıca Matplotlib tabanlı çizim yardımcıları ile sonuçları kolayca çizdirin.
[Analiz](guides/analysis/index.md) / [Çizim](guides/plotting/index.md)

- **CLI:** Sık yapılan işler için komut satırı üzerinden hızlı doğrulama ve inceleme komutları kullanın.
[CLI](guides/cli/cli.md)

## Buradan Başlayın

1. Proje hedeflerini, önerilen önkoşulları ve dokümanların nasıl düzenlendiğini anlamak için [Başlarken genel bakış](getting-started/overview.md) sayfasını okuyun.
2. Dynlib'i kurmak, CLI ile doğrulamak ve Python ile yerleşik (built-in) bir modeli çalıştırmak için [Hızlı Başlangıç](getting-started/quickstart.md) adımlarını izleyin.
3. DSL ile model oluşturmak ve modeli doğrulamak için [İlk Modeliniz](getting-started/first-model.md) bölümünü inceleyin.

## Daha Derine İnin

Dynlib dokümantasyonundaki diğer rehberlere bakarak daha detaylı kullanım hakkında fikir sahibi olabilirsiniz:

- **[Ana Sayfa](index.md)**

### Başlarken
- [Genel Bakış](getting-started/overview.md)
- [Hızlı Başlangıç](getting-started/quickstart.md)
- [İlk Modeliniz](getting-started/first-model.md)

### Rehberler

#### CLI
- [CLI](guides/cli/cli.md)

#### Modelleme
- [Modelleme rehberi](guides/modeling/index.md)
- [DSL temelleri](guides/modeling/dsl-basics.md)
- [Denklemler](guides/modeling/equations.md)
- [Matematik ve makrolar](guides/modeling/math-and-macros.md)
- [Üçlü if](guides/modeling/ternary-if.md)
- [Model kaydı](guides/modeling/model-registry.md)
- [Yardımcı değişkenler](guides/modeling/aux.md)
- [DSL fonksiyonları](guides/modeling/functions.md)
- [Olaylar](guides/modeling/events.md)
- [Gecikme](guides/modeling/lagging.md)
- [Satır içi modeller](guides/modeling/inline-models.md)
- [Yapılandırma dosyası](guides/modeling/config-file.md)
- [Modlar](guides/modeling/mods.md)
- [Ön ayarlar](guides/modeling/presets.md)
- [Simülasyon varsayılanları](guides/modeling/sim.md)

#### Simülasyon
- [Simülasyon rehberi](guides/simulation/index.md)
- [Temeller](guides/simulation/basics.md)
- [Yapılandırma](guides/simulation/configuration.md)
- [Just-In-Time derleme](guides/simulation/jit.md)
- [Ön ayarlar](guides/simulation/presets.md)
- [Runner varyantları](guides/simulation/runner-variants.md)
- [Oturum içgözlemi](guides/simulation/session-introspection.md)
- [Sonuçlar](guides/simulation/results.md)
- [Adımlayıcılar](guides/simulation/steppers.md)
- [Anlık görüntüler ve devam](guides/simulation/snapshots-and-resume.md)
- [Sarmalayıcı ve çalıştırıcı](guides/simulation/wrapper-and-runner.md)
- [Kaynakları dışa aktarma](guides/simulation/export-sources.md)

#### Çizim
- [Çizim rehberi](guides/plotting/index.md)
- [Çizim temelleri](guides/plotting/basics.md)
- [Dekorasyonlar](guides/plotting/decorations.md)
- [Çizimleri dışa aktarma](guides/plotting/export.md)
- [Örümcek ağ çizimleri](guides/plotting/cobweb.md)
- [Çekim havzası çizimleri](guides/plotting/basin-plot.md)
- [Vektör alanları](guides/plotting/vectorfields.md)
- [Manifold çizimleri](guides/plotting/manifold-plot.md)
- [Tema ve yüzeyler](guides/plotting/themes-and-facets.md)
- [Bifurkasyon diyagramları](guides/plotting/bifurcation-diagrams.md)

#### Analiz
- [Analiz rehberi](guides/analysis/index.md)
- [Çalışma zamanı gözlemcileri](guides/analysis/observers.md)
- [Lyapunov analizi](guides/analysis/lyapunov.md)
- [Tarama araçları](guides/analysis/sweep.md)
- [Analiz sonrası](guides/analysis/post-analysis.md)
- [Sabit noktalar](guides/analysis/fixed-points.md)
- [Çekim havzası analizi](guides/analysis/basin.md)
- [Bifurkasyon diyagramları](guides/analysis/bifurcation.md)
- [Manifold analizi](guides/analysis/manifold.md)

### Örnekler
- [Genel bakış](examples/index.md)
- [Analiz kataloğu](examples/analysis.md)
- [Çizim kataloğu](examples/plotting.md)
- [Çalışma zamanı kataloğu](examples/runtime.md)
- [Durum yönetimi](examples/state-management.md)
- [Bifurkasyon lojistik haritası](examples/bifurcation.md)
- [Collatz sanısı](examples/integer-map.md)
- [Izhikevich nöronu](examples/izhikevich.md)

### Referans
- [Genel bakış](reference/index.md)
- [Yerleşik modeller](reference/models/index.md)

### Proje
- [Değişiklik günlüğü](project/changelog.md)
- [Sorunlar](project/issues.md)
- [TODO](project/todo.md)
