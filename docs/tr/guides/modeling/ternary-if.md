## Ternary (Üçlü) `if` ifadeleri

DSL, kısa, iki yönlü dallanmaları satır içi (inline) tutmanıza olanak tanır çünkü her sağ taraf ifadesi bir Python ifadesi olarak ayrıştırılır. Bu, tanıdık Python ternary (üçlü) formunu yazabileceğiniz anlamına gelir:

```dsl
<doğruysa değer> if <koşul> else <yanlışsa değer>
```

Bu form, koşul sadece yan etkilere veya ek ifadelere ihtiyaç duymadan iki hesaplama arasında seçim yaptığında idealdir. Nihai değeri üretmeden önce birden fazla atamaya, loglamaya veya diğer zorunlu adımlara ihtiyacınız olduğunda tam bir `if`/`else` bloğu kullanın.

### Depodan örnekler

Modlar rehberinun kendisi, eklenen bir yardımcı fonksiyon içinde ternary bir ifade gösterir ve bunu `N == 0` olduğunda bir formülü, `N` pozitif olduğunda başka bir formülü kullanmak için kullanır (bkz. `docs/guides/modeling/mods.md:483-491`):

```toml
h = {args = ["phi","N"], expr="""
phi if N==0 else phi-sum(sign(phi+(2*j-1))+sign(phi-(2*j-1)) for j in range(1,N+1))
"""}
```

Birim testleri de, RHS'nin (sağ tarafın) zamana bağlı olmasını sağlamak için satır içi bir modelde ternary bir dala güvenir; `tests/unit/test_sum_generator_lowering.py:27-110` modeli şununla tanımlar:

```toml
[equations.rhs]
x = "1.0 if t < 0 else sum(i for i in range(N))"
```

ve dalın her iki tarafını da Python ve JIT backend'lerine karşı doğrular.

Bu örnekler, ternary ifadelerin derleyiciye seçmesi için iki ayrı yol sunarken ifadeleri nasıl özlü tuttuğunu gösterir.

### İç içe `if` ifadeleri

Tek bir ifadede birden fazla koşulu işlemek için ternary `if` ifadelerini iç içe yerleştirebilirsiniz. Bu, mantık tamamen fonksiyonel kaldığı sürece tam `if`/`else` bloklarına başvurmadan daha karmaşık dallanmalara izin verir.

#### Sözdizimi

```dsl
<değer1> if <koşul1> else <değer2> if <koşul2> else <değer3> if ... else <varsayılan_değer>
```

İç içe yerleştirmenin okunabilirliği azaltabileceğini unutmayın, bu yüzden idareli kullanın ve gerekirse netlik için parantez kullanmayı düşünün.

#### Örnek

Birden fazla eşiğe dayalı bir değer seçmeniz gerektiğini varsayalım. Bir model denkleminde şöyle yazabilirsiniz:

```toml
result = "0 if x < 0 else 1 if x < 10 else 2"
```

Bu şuna değerlendirilir:
- `x < 0` ise `0`
- `0 <= x < 10` ise `1`
- Aksi takdirde `2`