# Wrapper ve Runner Etkileşimi

Ana simülasyon şeması şu şekilde organize edilmiştir: wrapper ⊃ runner ⊃ stepper. Runner ve stepper, JIT ile derlenebilir bir simülasyon çekirdeği (kernel) oluşturur. Bu çekirdek numba uyumluluğu için tasarlanmıştır, bu nedenle karmaşık Python işlemleri olmadan sıkı bir döngü içinde çalışır. Stepper'ın `emit()` metodu, bu çekirdeğin stepper tarafındaki hesaplayıcısını sağlar ve JIT uyumlu bir fonksiyon döndürür. Her simülasyon adımında runner bu fonksiyonu çağırır. Stepper fonksiyonunun kendi iç döngüsü vardır; başarılı bir sonraki adım değeri elde edilene kadar yeniden deneyebilir. Her adımda runner; stepper sonucunun durumunu, tamponları, kayıtları vb. kontrol eder. Simülasyon bitmeden olağandışı bir olay gerçekleşirse, simülasyonu duraklatabilir ve tüm sorumluluğu bir durum koduyla wrapper'a iade edebilir. Wrapper durum kodunu inceler ve JIT'lenebilir bir çekirdek ile gerçekleştirilemeyecek gerekli eylemi (tamponun yeniden tahsis edilmesi gibi) gerçekleştirir. Gerekli eylemden sonra wrapper, runner'ı (durum terminal değilse) yeniden başlatır.

Aşağıda bu şemanın detayları açıklanmıştır.

## Yığındaki (Stack) Sorumluluklar
- `Sim._execute_run`, `run_with_wrapper`'ı çağırmadan önce tohumları (seeds), kayıt seçimlerini, durdurma aşaması maskelerini ve workspace ipuçlarını hazırlar. Bu sayede wrapper, yürütmeyi sürmek için ihtiyaç duyduğu derlenmiş çağrılabilirleri ve varsayılanları görür.
- `run_with_wrapper` iletken görevi görür: her runner çağrısından önce kayıt/olay tamponlarını, workspace'leri, gözlemci kancalarını ve stepping parametrelerini rezerve eder. Bu, sıcak (hot) runner döngüsünün yalın ve adımlamaya odaklanmış kalmasını sağlarken; kurulum, büyüme ve son işlem adımlarını wrapper'ın yönetmesini sağlar.

## Wrapper Ne Yapar?
1. **Her workspace'i tahsis eder ve tohumlar.** Çalışma zamanı ve stepper workspace'leri bir kez oluşturulur ve devam etme senaryolarını etkinleştirmek için isteğe bağlı olarak `seed.workspace` ile tohumlanabilir.
2. **Kayıt ve olay havuzlarını yönetir.** Wrapper, kayıt dizilerini sadece seçilen durum/yardımcı dizinler için dilimler, olay günlüğü tamponlarını tahsis eder ve runner sonuçları için imleçleri (cursors) tutar.
3. **Yürütme yapılandırmasını ayırır.** `Sim` ayarlarını (dt, uyarlamalı bayraklar, durdurma aşamaları vb.) runner'ın ihtiyaç duyduğu girdilere çevirir.
4. **Analiz ve gözlemcileri bağlar.** Gözlemciler kayıtlıysa, wrapper onların workspace'lerini ve izleme tamponlarını tahsis eder.
5. **Runner döngüsünü sürer.** Bir `while True` döngüsü içinde, derlenmiş runner'ı tekrar tekrar çağırır, sahip olduğu tamponları aktarır ve runner'dan dönen durum kodlarına yanıt verir.

## Runner ABI Sözleşmesi
Runner, kayıtlar veya gözlemciler hakkında hiçbir şey bilmeyen JIT dostu bir çağrılabilir yapıdır; sadece `runner_api.py` içinde tanımlanan donmuş ABI'ye uyar. Her çağrı şunları alır: `(t0, t_end, dt_init, max_steps)` gibi skalerler, model depolaması (`y_curr`, `params`), workspace'ler, kayıt tamponları, olay günlükleri ve düşük seviyeli imleçler. Tek bir adımlama dönemini (epoch) yönetir ve yalnızca iyi tanımlanmış durumlar üzerinden çıkış yapabilir.

Önemli Durumlar:
- `DONE` / `EARLY_EXIT`: Ufka ulaşıldığını veya bir durdurma koşulunun tetiklendiğini wrapper'a bildirir. Wrapper daha sonra son durumu kopyalar ve bir `Results` nesnesi oluşturur.
- `GROW_REC` / `GROW_EVT`: Wrapper'a kayıt veya olay tamponlarını yeniden boyutlandırması gerektiğini bildirir. Wrapper büyütme işlemini yapar ve runner'ın kaldığı yerden devam edebilmesi için imleçleri güncelleyerek runner'a yeniden girer.
- `STEPFAIL`, `NAN_DETECTED`, `USER_BREAK`: Uyarı olarak geri döner; wrapper yine de kısmi çıktıların incelenebilmesi için bir `Results` anlık görüntüsü oluşturur.
- `OK`: Dahili kalır; wrapper, çıkış durumlarından biri görülene kadar runner'ı yeniden çağırmaya devam eder.

## Döngü ve Yeniden Giriş (Re-entry)
Runner büyüme rapor ettiğinde wrapper:
- Mevcut verileri kopyalarken uygun tamponu büyütür, kapasiteyi günceller,
- Başlangıç imleçlerini geri sarar ve
- Runner'ın sürekliliği kaybetmeden bir sonraki parçaya devam edebilmesi için son işlenen zamanı/dt'yi kullanır.

## Veri Akışı Özeti
1. `Sim._execute_run`, ayarları paketler ve `run_with_wrapper`'a iletir.
2. Wrapper tamponları ve analiz üst verilerini hazırlar, ardından derlenmiş runner'ı çağırır.
3. Runner bir durum döndürür; wrapper bunu yorumlar, gerekiyorsa tamponları büyütür ve simülasyon tamamlanana veya iptal edilene kadar döngüye girer.
4. Son olarak wrapper; kayıtlı diziler, olay günlükleri ve son durum ile bir `Results` örneği oluşturur.

Bu ayrım, Python tarafındaki defter tutma işlemlerinden el yazımı wrapper'ın sorumlu olmasını sağlarken, derlenmiş runner'ın sayısal olarak yoğun adımlama döngüsüne odaklanmasına izin verir.