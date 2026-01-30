# Runner Türleri

Runner katmanının nasıl yapılandığını anlamak; performans dengelerini, observer (gözlemci) desteğini ve fast-path API’nin normal wrapper döngüsünden nasıl ayrıldığını daha kolay anlamanı sağlar.

## Normal runner’lar

`run_with_wrapper` (bkz. `src/dynlib/runtime/wrapper.py`) varsayılan çalıştırma yolunu yönetir. Şunları yapar:

- `Sim` ayarlarını (t0/tend, adaptive bayraklar, discrete ufuklar, stop phase’ler, seçmeli kayıt seçenekleri, observer’lar vb.) dondurulmuş runner ABI’si ile uyumlu tamponlara ve skaler girdilere çevirir.
- kayıt/event dizilerini ve workspace’leri bir kez ayırır, sonra `runner_variants.get_runner` üzerinden `RunnerVariant.BASE` veya `RunnerVariant.ANALYSIS` runner’ını çağırır.
- `GROW_REC`, `GROW_EVT`, `DONE`, `EARLY_EXIT`, `USER_BREAK`, `NAN_DETECTED` gibi runner sinyallerini yönetir; gerekirse tamponları büyütür veya duraklatıp devam ettirir. Böylece derlenmiş kernel hiç bir zaman bellek yeniden ayırmak zorunda kalmaz.
- dönüşte son state/dt, trace’ler ve observer metadatasını `Results` nesnesine kopyalar. Bu yüzden wrapper sıcak döngüyü yalın tutar; derlenmiş runner ise sayısal stepping’e odaklanır.

Bu yol event’leri, değişken kayıt uzunluklarını, stop phase’leri ve büyüme kodlarını takip ettiği için buna **normal runner** diyoruz. En esnek runner budur ve `fastpath` açıkça istenmedikçe `Sim.run` tarafından kullanılır. Hem continuous (ODE) hem de discrete (map) stepper’lar aynı normal runner şablonlarını paylaşır (`RunnerVariant.BASE` veya `RunnerVariant.ANALYSIS`, `discrete` bayrağı ile). Böylece wrapper, zamanı baz alan ufukları ve iterasyon bütçesi bazlı ufukları aynı ABI ile yönetebilir.

## Fast-path runner’lar

`runtime/fastpath/executor.py` özel, sabit-adım (fixed-step) yolunu sürer. Executor her şeyi en başta ayırır (workspace’ler, seçmeli kayıt tamponları, stop bayrakları, variational hook’lar) ve sonra `runner_variants.get_runner` ile `RunnerVariant.FASTPATH` veya `RunnerVariant.FASTPATH_ANALYSIS` ister.

Fast-path runner’lar sadeleştirilmiştir:

- event-log büyütme veya “sticky” tampon yeniden boyutlandırma yoktur — her şey ilk çağrıdan önce seçilen `RecordingPlan`’a göre ( `RecordingPlan.capacity` üzerinden) boyutlandırılır.
- `GROW_*` durumları yoktur; runner aldığı tamponların yeterli olduğunu varsayar. Bu da döngüyü sıkı tutar.
- runner şablonunun içinde event/kesinti döngüsü yoktur; hazırlık için geçici warm-up, son kırpma (trimming) ve metadata üretimi executor’un sorumluluğundadır.

Bu yüzden fast-path runner’lar; tekrarlı toplu çalıştırmalar (`run_batch_fastpath`), throughput benchmark’ları veya sabit adım boyu ve bellek limitlerini garanti edebildiğin senaryolar için idealdir. Executor yine de observer trace’lerini, variational hook’ları ve runtime workspace’lerini hazırlar; ama bunu sayısal olarak “sıcak” döngünün dışında yapar.

## Analysis runner’lar

Observer eklendiğinde, hem normal hem de fast-path runner’lar analysis varyantına geçer:

- `RunnerVariant.ANALYSIS` (wrapper yolu) ve `RunnerVariant.FASTPATH_ANALYSIS` (fast-path yolu) observer hook’larını global olarak (`ANALYSIS_PRE` ve `ANALYSIS_POST`) enjekte eder; böylece wrapper/executor runner ABI üzerinden fonksiyon handle taşımak zorunda kalmaz.
- `runner_variants.compile_analysis_hooks`, `ObserverModule.resolve_hooks` çıktısını çözer, iki hook’u da önceden derler (ya da no-op yedeğini kullanır) ve bunları `analysis_signature_hash` içeren runner cache anahtarına dahil eder.
- Analysis runner’lar ayrıca `analysis_ws`, `analysis_out` ve trace tamponlarını bağlar; `analysis.trace.record_interval()` ve `analysis_kind` gibi metadata’ya uyar. Böylece kullanıcılar ana simulation adımlarının yanında “yan kanal” verisi de toplayabilir.
- Variational observer’lar, runner’ın varsayılan stepper yerine çağıracağı bir `runner_variational_step` callback’i sağlayabilir; bu sayede hook’lar JIT uyumluluğunu bozmadan önerileri (proposals) ayarlayabilir.

Analysis runner’lar, hangi varyant seçildiyse ona göre ya normal runner özelliklerini (kayıt, durdurma, büyüme) miras alır ya da fast-path sadeleştirmelerini uygular.

## Mimari Referans

`runner_variants.py` tüm runner şablonları için tek referans noktasıdır. `Sim.run()` observer yoksa base runner’ı kullanır; `runtime/fastpath/executor.py` ise observer varsa analysis-aware fast-path runner ister. Her executor/normal runner varyantı ayrı ayrı cache’lenir; böylece fast-path batch’leri ve wrapper çağrıları kendi derlenmiş kernel’larını kullanır.

### `runner_variants.py`

- Tüm runner varyantları (`BASE`, `ANALYSIS`, `FASTPATH`, `FASTPATH_ANALYSIS`) için şablonları ve derleme mantığını tanımlar; hem continuous hem discrete modelleri kapsar.
- `get_runner(variant, ...)` fonksiyonunu sunar: cache anahtarı üretir, observer hook’larını global olarak enjekte eder, isteğe bağlı olarak Numba ile JIT derler ve hem LRU (bellek içi) hem de disk üstü runner cache’lerini kullanır.
- Python source üretiminden, analysis hook’larını hazırlamaktan ve aynı runner şablonlarının hem `wrapper.py` hem `executor.py` tarafından kullanılmasını sağlamaktan sorumludur.

### `executor.py` (fast-path)

- Sabit-adım fast-path çalıştırmayı yönetir: tamponlar, workspace’ler, opsiyonel geçici warm-up ve sonuçların toplanması (result marshalling).
- Observer varlığına göre `get_runner(RunnerVariant.FASTPATH, ...)` veya `RunnerVariant.FASTPATH_ANALYSIS` seçer.
- Tekli ve batch çalıştırmayı (opsiyonel paralelleştirme ile) destekler; trajectory mantığını ortak runner şablonlarının içinde bırakır.
- Observer trace/metadata’yı sonlandırır; böylece çağıran taraf normal wrapper yoluyla aynı analysis payload’larını alır.

## runner_variants.py ile runner üretimi

`src/dynlib/compiler/codegen/runner_variants.py` her runner şablonu için tek kaynaktır.

- `RunnerVariant` dört desteklenen türü sayar: `BASE`, `ANALYSIS`, `FASTPATH`, `FASTPATH_ANALYSIS`.
- `_RUNNER_TEMPLATE_MAP`, her varyantı ve continuous/discrete bayrağını doğru şablon metni ve fonksiyon adıyla eşleştirir.
- `get_runner`, `(model_hash, stepper_name, analysis_sig, variant, runner_kind, dtype, cache_token, jit flag, template version)` bileşenlerinden oluşan bir cache anahtarı kurar ve `_variant_cache_continuous` veya `_variant_cache_discrete` içinde arar.
- Runner yoksa: source üretir, isteğe bağlı Numba ile JIT derler, `ANALYSIS_PRE`/`ANALYSIS_POST` hook global’lerini enjekte eder ve çağrılabilir nesneyi hem yerel LRU’ya hem de disk üstü `runner_cache`’e yazar.
- `analysis_signature_hash`, her observer set’ini stabil 16 karakterlik bir hash’e indirger; böylece observer hook’ları dinamik üretilse bile runner’lar cache’lenebilir.

Hem wrapper hem de fast-path executor `get_runner` çağırdığı için, yeni bir runner varyantı (ör. her zaman özel bir log’a yazan ya da ekstra tanıları birleştiren bir varyant) eklemek; yeni bir şablon metni eklemek, `_RUNNER_TEMPLATE_MAP`’e kaydetmek ve ilgili çalıştırma yolundan çağırmak demektir.

## Fast-path executor sorumlulukları

`executor.py` sadece runner çağırmaz:

- `_RunContext`, `_WorkspaceBundle` ve `_call_runner` yapılarını uygular; doğru tamponları ayırır, cursor reset’lerini yönetir ve `run_single_fastpath`, `run_batch_fastpath` ve optimize batch yardımcıları arasında çağrı noktasını birleştirir.
- Observer verilmişse fast-path runner varyantını seçer ve wrapper’a benzer şekilde `analysis=None` veya gerçek observer modülünü `get_runner`’a geçirir; ancak daha kısıtlı bir ortamda.
- Observer varsa yine de trace tamponlarını, metadata’yı (`build_observer_metadata`) ve opsiyonel variational step callback’lerini toplar; böylece runner analysis çıktısı üretebilir.
- Batch yardımcıları opsiyonel olarak thread pool kullanarak aynı runner’ı birden fazla IC/parametre seti üzerinde çalıştırır; derlenmiş runner’ın jit-safe ve GIL-free olmasına güvenir.

Bu ayrım; fast-path executor ile normal wrapper’ın aynı şablonları ve aynı cache altyapısını paylaşıp, iç döngü çevresindeki “defter tutma” (bookkeeping) miktarında ayrışmasını sağlar.
