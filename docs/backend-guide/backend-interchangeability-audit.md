# [CPU](../glossary.md#term-cpu) ↔ [Metal](../glossary.md#term-metal): podrobný audit úplné zaměnitelnosti

Reader navigation: [backend capability](../glossary.md#backend-capability),
[backend ownership](../glossary.md#backend-ownership), [layout](../glossary.md#layout),
[materialization](../glossary.md#materialization),
[numerical semantics](../glossary.md#profile-free-numerical-semantics), and
[general Scan](../glossary.md#general-differentiable-scan).

> **Čtenářský kontext přidaný při redakční revizi:** původní audit a jeho důkazy níže
> zůstávají historickým záznamem. Aplikace volá Engine (případně jej volá NN za aplikaci);
> Compiler se přes Planning ptá provideru na popisy operandů a výsledků, nikoli na jejich
> aktuální čísla. Prepare potom ověřuje realizaci celého oddílu grafu a běh pracuje se skutečnými
> buffery. Kladná odpověď provideru tedy sama nezaručuje přípravu ani správný numerický výsledek.
> Vlastník vstupní paměti ji udržuje živou do uzavření výsledku; explicitní materializace
> vytváří nezávislou hostitelskou kopii. Výraz „leaf“ označuje Tensor bez produkujícího výrazu,
> nikoli nutně alokovaná data. Podrobný výklad hranic volání a kopií je v
> [tabulce vlastnictví](../explanation/ownership.md#which-call-borrows-or-copies);
> rozdíl metadat, přípravy a běhu vysvětluje
> [capability provider](capability-provider.md#who-asks-the-question).
> Aktuální opravy čtěte v [oddílu 15](#15-navazující-opravy-a-aktuální-hranice)
> spolu s [rozsahem produkčních důkazů](production-support.md#status-and-interpretation),
> ne jako zpětnou změnu výsledků původního auditu.

## 1. Závěr a platnost

**[CPU](../glossary.md#term-cpu) a [Metal](../glossary.md#term-metal) nejsou plně zaměnitelné ani pro [FLOAT32](../glossary.md#term-float32).** Shoda názvů operací a úspěšné katalogové reprezentanty neznamenají shodu všech [tvarů](../glossary.md#shape), layoutů, hodnotových domén, gradientových uzávěrů a připravených programů.

Audit míří na repozitář `/Users/phujka/IdeaProjects/Synaptik`, [commit](../glossary.md#term-commit) `65c0eb00745c77167f7df4e2bd3b4aad7a20b31c`. Pracovní strom byl před auditem čistý. Produkční implementace, konfigurace [backendů](../glossary.md#backend) ani zapečetěná distribuce nebyly tímto auditem změněny. Zpráva je snímek schopností, nikoli nový numerický kontrakt.

**Navazující opravy:** oddíly 1–14 zachovávají původní audit commitu `65c0eb0`; nejsou přepsanou
kvalifikací nové implementace. Aktuální změny, nové [běhy](../glossary.md#run--runtime) a zbývající hranice uvádí
[oddíl 15](#15-navazující-opravy-a-aktuální-hranice).

Hlavní zjištění:

- **Typy:** CPU má obecné [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) výpočty a úmyslný zámek [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) aritmetiky. Metal má F32/BF16/F16 numerické vlastníky a oddělené přesné/raw F64 role, nikoli obecnou F64 aritmetiku.
- **F32 domény:** existují malé, skutečně spuštěné protipříklady v obou směrech. Metal odmítne některá prázdna, [unary](../glossary.md#term-unary) nad transponovaným [view](../glossary.md#term-view-meanings) a další [layouty](../glossary.md#layout); CPU odmítne prázdnou normalizační osu softmaxu a platný nulový broadcast [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) attention.
- **Počet operandů:** odstranění stropu 16 neznamenalo neomezenou variadicitu CPU. [Rank](../glossary.md#term-rank)-1 CONCAT prošel se 101 vstupy, se 102 a 257 ne; Metal prošel i s 257. Jde o počet operandů, ne o počet řádků jediného tensoru.
- **Capability versus [prepare](../glossary.md#prepare):** u několika Metal výskytů je [descriptor](../glossary.md#term-descriptor) přijat při [kompilaci](../glossary.md#compiler-engine-and-execution-stage-names), ale [příprava](../glossary.md#prepare) odmítne fyzický stav/provenienci vstupu. Některá odmítnutí jsou úmyslně užší ingress; jiná jsou kandidáty na sjednocení Java/native validace.
- **Numerika:** byly skutečně zachyceny rozdíly CPU RSQRT a [dropout](../glossary.md#dropout--inverted-dropout) vůči předepsaným typovaným mezikrokům. Nezaměňovat s obecným požadavkem bitové shody backendů.
- **[Scan](../glossary.md#general-differentiable-scan) a veřejná [API](../glossary.md#term-application-programming-interface-api):** statický F64 [identity](../glossary.md#term-identity-meanings) Scan a strided F32 Scan prošly CPU, nikoli Metal. [Engine](../glossary.md#term-engine)-level fixed-cell [AD](../glossary.md#autograd) existuje; owned `Model.configureTraining` přesto fixed recurrence odmítá na obou backendech.
- **Symbolika:** CPU umí kvalifikované měnící se B přes `runValues`; Metal je statický. Ani CPU nemá úplnou symbolickou paritu F32/F64.
- **[Persistence](../glossary.md#term-persistence) a [runtime](../glossary.md#term-runtime):** uložený zdroj není přenositelný executable. Load zachovává typy a recept, ne cílovou způsobilost, připravené plány ani [optimizer](../glossary.md#term-optimizer) continuation.

## 2. Metodika a důkazní vrstvy

Zaměnitelnost je průnik několika nezávislých podmínek:

| Vrstva | Co skutečně dokazuje |
|---|---|
| [Model](../glossary.md#term-model) | Platný druh operace, atributy, role, typová/promotion pravidla a odvozené výsledky. |
| Capability | Přípustnost konkrétního výskytu pro [metadata](../glossary.md#term-metadata) dostupná provideru. Neověřuje automaticky obsah bufferů, alokovatelnost či všechny fyzické vazby. |
| [Compiler](../glossary.md#term-compiler)/[AD](../glossary.md#autograd) | Úplný dopředný a případný vybraný derivovaný [graf](../glossary.md#graph) našel přípustné vlastníky. |
| [Prepare](../glossary.md#prepare) | Prošel konkrétní [lowering](../glossary.md#lowering), topologie, [layout](../glossary.md#layout)/provenience, prostředky a [ABI](../glossary.md#term-application-binary-interface-abi). |
| [Run](../glossary.md#term-run)/materialize | Prošla aktuální data, hodnotové kontroly, exekuce a publikace. |
| Numerický [oracle](../glossary.md#term-test-oracle) | Pro konkrétní případ navíc sedí nezávislý očekávaný výsledek. Samotný [hash](../glossary.md#term-hash) výsledku to nedokazuje. |
| Cold/save/load | Zdroj lze rekonstruovat a znovu svázat; neznamená úspěšné spuštění na jiném [backendu](../glossary.md#backend). |

**R** = nově skutečně spuštěno v tomto auditu. **S** = podloženo aktuální implementací, bez samostatného nového [běhu](../glossary.md#run--runtime) dané kombinace. **H** = dřívější uchovaný kvalifikační důkaz. **[[INFERENCE](../glossary.md#term-inference-meanings)]** = odvozený, dosud neprovedený konkrétní scénář. Permanentní testy byly čteny jako zdroj, nikoli vydávány za nově spuštěné.

Nové [běhy](../glossary.md#run--runtime) proběhly na macOS [arm64](../glossary.md#native-packaging-and-platform-vocabulary) / Apple M3 Max / [JDK](../glossary.md#term-java-development-kit-jdk) 26.0.1. [Metal](../glossary.md#term-metal) byl registrován samostatně, bez [CPU](../glossary.md#term-cpu) backendu. Použitá lokální knihovna má [SHA-256](../glossary.md#term-sha-256) `f9f18b2378403b65629de1900b21235b3a44e4e06b709bac81263e9904bf6334`, ABI 7, [schema](../glossary.md#term-schema) 22, semantic [identity](../glossary.md#term-identity-meanings) 43. CPU běhy měly samostatného CPU vlastníka. Nebyl znovu spouštěn root test gate ani benchmark; jejich starší receipts nejsou novými výsledky tohoto auditu.

### Nově provedený katalog

Stejný existující auditovací harness byl znovu zkompilován proti aktuálním třídám a aktuálnímu [Metal](../glossary.md#term-metal) reprezentantovému [fixture](../glossary.md#term-fixture). Nebyl vytvořen druhý katalog operací.

| Výsledek katalogu | [CPU](../glossary.md#term-cpu) | Metal |
|---|---:|---:|
| Záznamy | 5 018 | 5 018 |
| Ordinary druhy / legální signatury | 115 / 137 | 115 / 137 |
| Dokončené dopředné exekuce | 804 | 908 |
| Dokončené vybrané první [AD](../glossary.md#autograd) exekuce | 874 | 1 156 |
| Odmítnutí při výběru/[kompilaci](../glossary.md#compiler-engine-and-execution-stage-names) | 760 | 374 |
| Chybějící legální signaturový fixture | 0 | 0 |
| Selhání po přijetí při [prepare](../glossary.md#prepare)/[run](../glossary.md#term-run)/materialize v katalogu | 0 | 0 |

[General Scan](../glossary.md#general-differentiable-scan) je veden zvlášť. Další záznamy klasifikují neplatné projekce/fixture, výstupní role a nediferencovatelnost; nejsou automaticky backendovými mezerami. Syntetický katalog používá [Model](../glossary.md#term-model) [inference](../glossary.md#term-inference-meanings)/constructible occurrences a skutečný [Engine](../glossary.md#term-engine); není totožný se sadou výhradně veřejných uživatelských constructorů.

Při spárování všech klíčů se liší **1 134 výsledkových klasifikací**: [F64](../glossary.md#floating-point-precision-and-rounding) 374, [BF16](../glossary.md#floating-point-precision-and-rounding) 380, [F16](../glossary.md#floating-point-precision-and-rounding) 380. [F32](../glossary.md#floating-point-precision-and-rounding)/I32/I64/[BOOL](../glossary.md#data-type--datatype) nemají v tomto reprezentativním katalogu rozdílnou klasifikaci. To **není** důkaz shody jejich úplných domén: následující veřejné boundary scénáře rozdíly prokazují.

Samostatně bylo provedeno **39 veřejných boundary scénářů na každý [backend](../glossary.md#backend)**, dále veřejné changing-B a numerické site sondy. Úspěšné konečné boundary případy prošly jednoduchým nezávislým raw-bit oraclem; [softmax](../glossary.md#softmax--log-softmax) s nefinální hodnotou je záměrně evidován odděleně bez obecného numerického tvrzení. Setup chyby diagnostického spotřebitele byly opraveny pouze v něm, nikoli v [backendu](../glossary.md#backend), a nejsou započítány jako mezery produktu.

## 3. Skutečně provedené protipříklady zaměnitelnosti

Všechny řádky této tabulky jsou **R**. „[Compile](../glossary.md#compile)“ znamená odmítnutí ještě před přípravou; „[prepare](../glossary.md#prepare)“ přijetí [kompilace](../glossary.md#compiler-engine-and-execution-stage-names), ale odmítnutí fyzické realizace; „[run](../glossary.md#term-run)“ odmítnutí aktuálních hodnot. Přesné chyby, deskriptory a raw výsledky jsou v `build/backend-interchangeability-audit/public-boundaries-{cpu,metal}.jsonl`.

| Scénář | [CPU](../glossary.md#term-cpu) | [Metal](../glossary.md#term-metal) | Význam |
|---|---|---|---|
| [F64](../glossary.md#floating-point-precision-and-rounding) ADD a smíšené [F32](../glossary.md#floating-point-precision-and-rounding)+F64 ADD | Správný F64 výsledek | Compile odmítne | Obecná F64 aritmetika není společná. |
| [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) ADD, BF16/F16+F32 ADD, smíšený low+F32 [MATMUL](../glossary.md#matrix-multiplication--matmul) | Compile odmítne původní low aritmetický operand | Správný výsledek | Zámek CPU nezmizí promotion do F32. |
| BF16+F16 ADD | Compile odmítne | Správné F32 `[2,4]` | Metal přijímá [Model](../glossary.md#term-model)-promoted smíšené low operandy. |
| Low EXPAND `[1]→[2]`, poté explicitní cast do F32 | Správně na obou typech | Správně na obou typech | Raw/movement není low aritmetika. |
| [Gradient](../glossary.md#term-gradient) stejného low EXPAND, [seed](../glossary.md#term-seed) `[1,1]` | Compile odmítne vyžadovaný SUM | Správný low gradient `2` | Forward podpora není automaticky backward uzávěr. |
| F32 ADD, [rank](../glossary.md#term-rank) 17, pouze dva logické prvky | Správné `[3,5]` | Compile odmítne | Není to limit velikosti dat; je to rank. |
| MaxPool2d, vstup `[1,1,1,1]`, [kernel](../glossary.md#kernel) 257×257, [padding](../glossary.md#padding) 128 | Správný jediný výsledek `7` | Compile odmítne | Limit 66 049 kernel pozic, nikoli velký výstup. |
| F32 `permute(1,0).reshape(6)` | Správné `[1,4,2,5,3,6]` | Stejný správný výsledek | CPU materializující noncontiguous RESHAPE už existuje. |
| F32 `permute(1,0).neg()` | Správné logické pořadí | Compile odmítne NEG | Následný [unary](../glossary.md#term-unary) consumer má na Metalu užší [layout](../glossary.md#layout) doménu. |
| F32 NEG nad `[0]` | Prázdný výsledek | Compile odmítne | Běžné Metal empty [pointwise](../glossary.md#term-pointwise) není obecně uzavřené. |
| F32 SUM nad `[0]` | [Skalár](../glossary.md#axis-rank-scalar-vector-and-matrix) +0 | Compile odmítne | Ordinary redukce a [Scan](../glossary.md#general-differentiable-scan)-local empty podpora nejsou totéž. |
| F32 `[1,0]@[0,1]` | Skalární [matice](../glossary.md#axis-rank-scalar-vector-and-matrix) +0 | Compile odmítne | I nulová kontrakce má různé domény. |
| F32 K=1 MATMUL s `−0 × 1` | Přesné −0 | Přesné −0 | Dřívější [singleton](../glossary.md#term-singleton) oprava je společná a prošla znovu. |
| F32 TOP_K, k=0, neprázdný vstup `[1,2]` | Prázdné values | Compile odmítne | Model-valid nulové k není na Metalu přijato. |
| F32 [softmax](../glossary.md#softmax--log-softmax) `[2,0]`, osa 1 | Compile odmítne | Správné prázdno | Opačný směr asymetrie. |
| Attention Q `[0,2,3]`, K `[1,4,3]`, V `[1,4,5]` | Compile odmítne | Správné prázdno `[0,2,5]` | CPU predicate nesprávně vybírá broadcast [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) pomocí maxima. |
| F32 softmax `[-MAX,+MAX]` | Správné `[0,1]` | Správné `[0,1]` | [Finite](../glossary.md#term-finite) input s [overflow](../glossary.md#term-overflow) záporného shiftu CPU aktuálně přijímá. |
| F32 softmax `[-∞,0]` | Run odmítne: input must be finite | Provede `[+0,1]` | CPU má užší aktuální hodnotovou doménu než Model/Metal. |
| Rank-1 CONCAT 17/65/101 vstupů | Správně | Správně | Strop 16 je odstraněn. |
| Rank-1 CONCAT 102 a 257 vstupů | Compile odmítne | Správně | CPU má stále rank-dependent generační budget. |
| Externí F32 affine EQUAL, [shape](../glossary.md#shape) `[2]`, [stride](../glossary.md#term-stride) 2, offset 1 | Správné [BOOL](../glossary.md#data-type--datatype) `[1,1]` | Prepare odmítne vstupní stav | Capability/kompilace samy nestačí. |
| Externí [transpose](../glossary.md#term-transpose) F32 MATMUL, shape `[3,2]`, strides `[1,3]` | Správné `[5,7,9]` | Prepare odmítne | Samotný transpose [descriptor](../glossary.md#term-descriptor) nestačí bez lokální [provenance](../glossary.md#provenance). |
| Offset/strided F32 MATMUL, strides `[4,1]`, offset 1 | Správné `[6,15]` | Compile odmítne | CPU má širší read-layout doménu. |
| Read shape `[2,2,2]`, strides `[0,1,1]`, ADD skaláru | Správné opakované logické čtení | Prepare odmítne feed geometry | Zero stride nezaručuje nekolidující ostatní osy. |
| Externí F64 FLOOR broadcast [view](../glossary.md#term-view-meanings), shape `[2]`, stride 0 | Správné `[1,1]` | Prepare: native INVALID_ARGUMENT | Java/exact/native vstupní stav není jednotně přijímán. |
| Statický F64 [identity](../glossary.md#term-identity-meanings) GeneralScan bez aritmetiky | Přesné emise `[3,4]` | Compile odmítne celý Scan | Metal Scan vylučuje F64 i pro raw passthrough. |
| F32 GeneralScan se strided/offset sequence | Správný final [carry](../glossary.md#term-carry) `6` | Prepare odmítne MATERIALIZED_LAYOUT | Binder/physical ingress je další rozměr rozdílu. |

Úspěchy neznamenají libovolně velké nebo libovolně aliasované varianty stejného řádku. Sources: [C1]<span data-glossary-navigation="true"> ([C1 glossary](../glossary.md#term-c1))</span>, [C3], [C4], [C7], [C8], [M1]–[M11], [S1]–[S4].

## 4. Numerické rozdíly: co je dovolená variace a co vyžaduje opravu/review

### 4.1 Obecně nelze slíbit bitovou shodu

Stejný [Model](../glossary.md#term-model) program neslibuje bitově identický obecný floating výsledek mezi [CPU](../glossary.md#term-cpu), Metalem, native routes a strategiemi. [Finite](../glossary.md#term-finite) accuracy se kvalifikuje po rodinách/typových cestách. Neexistuje jeden veřejný universal [tolerance](../glossary.md#term-tolerance) parametr. To ale neuděluje volnost měnit indexy, masky, tie-breaking, divisor, členství contributorů, uložené hodnoty, typované mezikroky nebo předepsané speciální hodnoty. [T3]

- Aritmetické [NaN](../glossary.md#term-nan) obvykle porovnáváme třídou, ne payloadem/signem; raw přesuny musí zachovat uložené bity a CAST má vlastní přesná konverzní pravidla.
- [Signed zero](../glossary.md#term-signed-zero) je [operation](../glossary.md#operation)-specific; z přípustné volby znaménka výsledku některé agregace nevzniká volnost přidat +0 do K=1 [MATMUL](../glossary.md#matrix-multiplication--matmul).
- [DAZ](../glossary.md#term-daz)/[FTZ](../glossary.md#term-ftz) je dovoleno jen na jmenovaných [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) aritmetických sites. Ne pro [F64](../glossary.md#floating-point-precision-and-rounding), přesné rozhodování, raw přesuny, CAST, masky či stav.
- Low numerická implementace na Metalu používá F32 pracovní hodnoty a zúží až příslušný deklarovaný low výstup. Nejde o F32 proxy [tensor](../glossary.md#tensor) nahrazující původní [graph](../glossary.md#graph) [dtype](../glossary.md#data-type--datatype).

### 4.2 [CPU](../glossary.md#term-cpu) [F32](../glossary.md#floating-point-precision-and-rounding) RSQRT: zachycený odlišný mezikrok

[Model](../glossary.md#term-model) RSQRT popisuje square-root site a následné typed-one division. CPU F32 emitter místo F32 rootu použije double sqrt i double division a až potom zúží. [Metal](../glossary.md#term-metal) má float root a float division. [T1] [C9] [M13]

| RSQRT(7.0f) | Raw F32 bity |
|---|---|
| Typovaný reference `1.0f / (float)StrictMath.sqrt(7)` | `0x3ec18490` |
| Skutečný CPU | `0x3ec1848f` |
| Skutečný Metal | `0x3ec18490` |

**R:** rozdíl je 1 [ULP](../glossary.md#term-unit-in-the-last-place-ulp); CPU odpovídá widened reference. Samotná odlišnost o 1 ULP ještě obecně není chybou backendové parity. Zde je navíc doložen odlišný předepsaný working-site postup; proto jde o konkrétní numerický [conformance](../glossary.md#term-conformance) nález k opravě/review, ne o důvod zavést blanket bitwise-equality testy nebo [epsilon](../glossary.md#epsilon) do predicates. Ostatních šest prověřených RSQRT vstupů v této sondě tento rozdíl neukázalo.

### 4.3 [CPU](../glossary.md#term-cpu) [dropout](../glossary.md#dropout--inverted-dropout): shodná maska/stav, odlišná kept aritmetika

[Model](../glossary.md#term-model) předepisuje typed complement, typed reciprocal, potom násobení vstupu. CPU emitter používá přímé widened dělení vstupu complementem; [Metal](../glossary.md#term-metal) odpovídá float complement/reciprocal/multiply. [T2] [C10] [M14]

**R:** 256 [F32](../glossary.md#floating-point-precision-and-rounding) prvků na případ, stejné původní key/counter; pozorované masky i next-state byly mezi [backendy](../glossary.md#backend) shodné a dropped hodnoty měly přesné +0. Rozdíly se týkaly zachovaných hodnot, nikoli výběru [RNG](../glossary.md#graph-rng-state--graphrngstate) větve.

| p | Kept prvků | CPU: rozdílné proti typed-site reference | Metal: rozdílné proti reference |
|---|---:|---:|---:|
| 0.1 | 237 | 164 | 0 |
| 0.3 | 181 | 91 | 0 |
| 0.7 | 81 | 10 | 0 |

Např. p=0.1, [lane](../glossary.md#term-lane) 0: input `0xbf89d89e`, CPU `0xbf992993`, typed reference `0xbf992994`. Zdroj CPU [F64](../glossary.md#floating-point-precision-and-rounding) větve také používá přímé dělení; nový F64 numerický counterexample zde proveden nebyl. Nález se neopravoval a nemá být maskován změkčením numerického kontraktu.

RNG přitom obecně **nemá přenosný bitstream promise mezi backendy/verzemi**. Shoda masek v těchto třech případech není trvalý cross-[backend](../glossary.md#backend) RNG kontrakt. [T4]

## 5. Typy, promotion a gradientové uzávěry

### 5.1 Základní asymetrie

[CPU](../glossary.md#term-cpu) low lock se vztahuje na **původní numerické operandy**, tedy i low+[F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) výpočet s vyšším výsledným typem, prázdný výstup, p=0 [dropout](../glossary.md#dropout--inverted-dropout) či [exponent](../glossary.md#term-exponent) 0/1. Explicitní low→F32 CAST→výpočet je jiný validní [graf](../glossary.md#graph); není implicitním emulačním fallbackem. [C5]

CPU zachovává low storage, všech 49 CAST párů, classification, raw movement/unfold, homogenní low přesné [unary](../glossary.md#term-unary) `ABS/NEG/SIGN/FLOOR/CEIL/RELU`, clamp/[extrema](../glossary.md#saturation-pole-and-extremum)/porovnání/WHERE, MIN/MAX/arg redukce, ordering a vybrané winner/replacement role. Jejich [VJP](../glossary.md#term-vector-jacobian-product-vjp) ale může přidat zakázané SUM, fold či jiné low výpočty. MaxPool2d může přijmout low forward s grad [metadata](../glossary.md#term-metadata), MaxPool3d low grad=true výslovně odmítá. [C1]<span data-glossary-navigation="true"> ([C1 glossary](../glossary.md#term-c1))</span> [C2]<span data-glossary-navigation="true"> ([C2 glossary](../glossary.md#term-c2))</span> [C3]

CPU smíšené F32/F64 a I32/I64 [pointwise](../glossary.md#term-pointwise) páry podporuje; u low extrema/comparison/WHERE vyžaduje homogenní low rodinu. [Metal](../glossary.md#term-metal) má širší floating promotion v těchto přesných rozhodovacích rolích, včetně [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding)→F32. Oba odmítají [Model](../glossary.md#term-model)-nelegální floating/integer míchání; existence CAST není implicitní promotion těchto kategorií. [C1] [M1] [M8]

Metal F64 není „vypnut úplně“: existují raw movement/CAST/classification/ordering/comparison, vybrané přesné unary/extrema/WHERE, maximum [pooling](../glossary.md#term-pooling) a neoverlapující folds. To neopravňuje F64 ADD, SUM, matmul, [convolution](../glossary.md#term-channels-first-convolution-layers), [normalization](../glossary.md#term-normalization) ani obecné neural arithmetic. F64 [Scan](../glossary.md#general-differentiable-scan) je užší ještě víc: jeho globální whitelist odmítá i čistý [identity](../glossary.md#term-identity-meanings) body. [M2] [M8] [S4]

### 5.2 Kompletní katalogová [matice](../glossary.md#term-review-matrix) po operacích

Následující tabulka zahrnuje všech **115 ordinary druhů a samostatný GeneralScan**. Je převzata ze stejného existujícího formátu katalogu; všech 5 018 klasifikací na [backend](../glossary.md#backend) bylo právě znovu spárováno a ověřeno proti čerstvému [běhu](../glossary.md#run--runtime). Tabulka říká **které typové forward/[AD](../glossary.md#autograd) reprezentanty se skutečně podařilo spustit**, ne všechny jejich [shape](../glossary.md#shape)/[layout](../glossary.md#layout)/value kombinace.

- **32/64** = [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding); **F** = obě; **L** = [BF16](../glossary.md#floating-point-precision-and-rounding)+[F16](../glossary.md#floating-point-precision-and-rounding); **I** = I32+I64; **B** = [BOOL](../glossary.md#data-type--datatype); **N** = všech šest numerických typů; **A** = N+B.
- Typ značí skutečný hlavní hodnotový datový role; pevné masky/indexy/stav se neprojektují jako floating data. U ONE_HOT jde o indexový typ (výstup je BOOL); INITIAL_STATE je I64.
- **AD** = vybraný první [gradient](../glossary.md#term-gradient) opravdu [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/[run](../glossary.md#term-run)/materialize; **[ND](../glossary.md#term-nd)** = [fixture](../glossary.md#term-fixture) nemá diferencovatelnou roli; **—** = žádná úspěšná [Model](../glossary.md#term-model)-valid projekce.
- **\*** = jen část prověřených signatur/variant/výstupních či vstupních rolí. Ani buňka bez hvězdičky neslibuje všechny možné [tvary](../glossary.md#shape), [layouty](../glossary.md#layout) nebo jiné role.
- F64 v přesném/source-only [Metal](../glossary.md#term-metal) [gradientu](../glossary.md#gradient-and-learning-rate) neznamená obecnou F64 numerickou aritmetiku.

| Operace Modelu | [CPU](../glossary.md#term-cpu) forward | CPU AD | Metal forward | Metal AD |
|---|---|---|---|---|
| `AggregateReductionKind.ALL` | B | ND | B | ND |
| `AggregateReductionKind.ANY` | B | ND | B | ND |
| `AggregateReductionKind.ARG_MAX` | N | ND | N | ND |
| `AggregateReductionKind.ARG_MIN` | N | ND | N | ND |
| `AggregateReductionKind.L1_NORM` | F | F | L+32 | L+32 |
| `AggregateReductionKind.L2_NORM` | F | F | L+32 | L+32 |
| `AggregateReductionKind.LOG_SUM_EXP` | F | F | L+32 | L+32 |
| `AggregateReductionKind.MAX` | N | F | N | L+32 |
| `AggregateReductionKind.MEAN` | F | F | L+32 | L+32 |
| `AggregateReductionKind.MIN` | N | F | N | L+32 |
| `AggregateReductionKind.PROD` | F+I | F | L+I+32 | L+32 |
| `AggregateReductionKind.STANDARD_DEVIATION` | F | F | L+32 | L+32 |
| `AggregateReductionKind.SUM` | F+I | F | L+I+32 | L+32 |
| `AggregateReductionKind.VARIANCE` | F | F | L+32 | L+32 |
| `AxisGatherKind.GATHER` | A | F | A | L+32 |
| `AxisGatherKind.GATHER_ELEMENTS` | A | F | A | L+32 |
| `AxisScatterKind.SCATTER_ADD` | F+I | F | L+I+32 | L+32 |
| `AxisScatterKind.SCATTER_ELEMENTS` | F+I+B; L* | F; L* | L+I+32+B; 64* | L+32; 64* |
| `AxisTransformKind.EXPAND_DIMS` | A | F+L | A | F+L |
| `AxisTransformKind.PERMUTE` | A | F+L | A | F+L |
| `AxisTransformKind.SQUEEZE` | A | F+L | A | F+L |
| `BatchNormKind.BATCH_NORM_INFERENCE` | F | F | L+32 | L+32 |
| `BatchNormKind.BATCH_NORM_TRAINING` | F | F | L+32 | L+32 |
| `BinaryArithmeticKind.ADD` | F+I | F | L+I+32 | L+32 |
| `BinaryArithmeticKind.DIV` | F | F | L+32 | L+32 |
| `BinaryArithmeticKind.MAX` | N | F | N | L+32 |
| `BinaryArithmeticKind.MIN` | N | F | N | L+32 |
| `BinaryArithmeticKind.MUL` | F+I | F | L+I+32 | L+32 |
| `BinaryArithmeticKind.POW` | F | F | L+32 | L+32 |
| `BinaryArithmeticKind.SUB` | F+I | F | L+I+32 | L+32 |
| `BinaryComparisonKind.EQUAL` | N | ND | N | ND |
| `BinaryComparisonKind.GREATER_OR_EQUAL` | N | ND | N | ND |
| `BinaryComparisonKind.GREATER_THAN` | N | ND | N | ND |
| `BinaryComparisonKind.LESS_OR_EQUAL` | N | ND | N | ND |
| `BinaryComparisonKind.LESS_THAN` | N | ND | N | ND |
| `BinaryComparisonKind.NOT_EQUAL` | N | ND | N | ND |
| `BooleanLogicalKind.AND` | B | ND | B | ND |
| `BooleanLogicalKind.NOT` | B | ND | B | ND |
| `BooleanLogicalKind.OR` | B | ND | B | ND |
| `CastKind.CAST` | A | F+L | A | F+L |
| `ContiguousKind.CONTIGUOUS` | A | F+L | A | F+L |
| `Conv2dKind.CONV2D` | F | F | L+32 | L+32 |
| `Conv3dKind.CONV3D` | F | F | L+32 | L+32 |
| `CumulativeScanKind.CUM_PROD` | F+I | F | L+I+32 | L+32 |
| `CumulativeScanKind.CUM_SUM` | F+I | F | L+I+32 | L+32 |
| `DropoutKind.DROPOUT` | F | F | L+32 | L+32 |
| `FloatingClassificationKind.IS_FINITE` | F+L | ND | F+L | ND |
| `FloatingClassificationKind.IS_INF` | F+L | ND | F+L | ND |
| `FloatingClassificationKind.IS_NAN` | F+L | ND | F+L | ND |
| `GatherNdKind.GATHER_ND` | A | F | A | L+32 |
| `GraphRngKind.INITIAL_STATE` | I64 | ND | I64 | ND |
| `LayerNormKind.LAYER_NORM` | F | F | L+32 | L+32 |
| `LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | F | F | L+32 | L+32 |
| `LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | F | F | L+32 | L+32 |
| `LossKind.MEAN_SQUARED_ERROR` | F | F | L+32 | L+32 |
| `MatmulKind.MATMUL` | F+I | F | L+I+32 | L+32 |
| `OneHotKind.ONE_HOT` | I | ND | I | ND |
| `OrderingKind.ARGSORT` | A | ND | A | ND |
| `OrderingKind.SORT` | A | F+L | A | F+L |
| `PadKind.PAD` | A | F+L | A | F+L |
| `Pool2dKind.AVERAGE_POOL2D` | F | F | L+32 | L+32 |
| `Pool2dKind.MAX_POOL2D` | F+L | F | F+L | L+32 |
| `Pool3dKind.AVERAGE_POOL3D` | F | F | L+32 | L+32 |
| `Pool3dKind.MAX_POOL3D` | F+L | F | F+L | L+32 |
| `RecurrentScanKind.GRU_RESET_AFTER` | F | F | L+32 | L+32 |
| `RecurrentScanKind.LSTM` | F | F | L+32 | L+32 |
| `RecurrentScanKind.RNN_TANH` | F | F | L+32 | L+32 |
| `RmsNormKind.RMS_NORM` | F | F | L+32 | L+32 |
| `ScalarElementwiseKind.ADD` | F+I | F | L+I+32 | L+32 |
| `ScalarElementwiseKind.CLAMP` | F+L | F | F+L | L+32 |
| `ScalarElementwiseKind.DIV` | F | F | L+32 | L+32 |
| `ScalarElementwiseKind.MAX` | N | F | N | L+32 |
| `ScalarElementwiseKind.MIN` | N | F | N | L+32 |
| `ScalarElementwiseKind.MUL` | F+I | F | L+I+32 | L+32 |
| `ScalarElementwiseKind.POW` | F | F | L+32 | L+32 |
| `ScalarElementwiseKind.SUB` | F+I | F | L+I+32 | L+32 |
| `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` | F | F | L+32 | L+32 |
| `ScanKind.SCAN` | F+I | F | L+I+32 | L+32 |
| `ScatterNdKind.SCATTER_ND` | F+I+B; L* | F; L* | L+I+32+B; 64* | L+32; 64* |
| `SelectKind.SELECT` | A | F+L | A | F+L |
| `ShapeTransformKind.EXPAND` | A | F | A | L+32 |
| `ShapeTransformKind.RESHAPE` | A | F+L | A | F+L |
| `SliceKind.SLICE` | A | F+L | A | F+L |
| `SliceKind.SLICE_UPDATE` | A | F+L | A | F+L |
| `SoftmaxKind.LOG_SOFTMAX` | F | F | L+32 | L+32 |
| `SoftmaxKind.SOFTMAX` | F | F | L+32 | L+32 |
| `TensorCompositionKind.CONCAT` | A | F+L | A | F+L |
| `TensorCompositionKind.STACK` | A | F+L | A | F+L |
| `TileKind.TILE` | A | F | A | L+32 |
| `TopKKind.TOP_K` | A | F+L | A | F+L |
| `UnaryElementwiseKind.ABS` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.CEIL` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.ERF` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.EXP` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.EXPM1` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.FLOOR` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.GELU` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.GELU_TANH_APPROXIMATION` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.LOG` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.LOG1P` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.NEG` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.RECIPROCAL` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.RELU` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.RSQRT` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.SIGMOID` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.SIGN` | F+L | F+L | F+L | F+L |
| `UnaryElementwiseKind.SILU` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.SQRT` | F | F | L+32 | L+32 |
| `UnaryElementwiseKind.TANH` | F | F | L+32 | L+32 |
| `WhereSelectionKind.WHERE` | F+L | F | F+L | L+32 |
| `WindowTransformKind.FOLD2D` | F | F | F+L | F+L |
| `WindowTransformKind.FOLD3D` | F | F | F+L | F+L |
| `WindowTransformKind.FOLD_AXIS` | F+I | F | N | F+L |
| `WindowTransformKind.UNFOLD2D` | F+L | F | F+L | L+32 |
| `WindowTransformKind.UNFOLD3D` | F+L | F | F+L | L+32 |
| `WindowTransformKind.UNFOLD_AXIS` | A | F | A | L+32 |

## 6. [Shape](../glossary.md#shape)/[layout](../glossary.md#layout)/value omezení podle rodin

Tato tabulka doplňuje [dtype](../glossary.md#data-type--datatype) matici: zde jsou důvody, proč stejná operace ve stejném [F32](../glossary.md#floating-point-precision-and-rounding) ještě nemusí znamenat přenositelnost. **S** jsou aktuální zdrojové podmínky; konkrétní **R** výsledky jsou v sekci 3.

| Rodina | [CPU](../glossary.md#term-cpu) | [Metal](../glossary.md#term-metal) | Zdroj / důkaz |
|---|---|---|---|
| [Unary](../glossary.md#term-unary) arithmetic a [activations](../glossary.md#term-activation) | Statické resolved nonnegative-[stride](../glossary.md#term-stride)/offset reads; podporuje i nekanonické input [views](../glossary.md#term-view-meanings). | N=F32/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) unary obecně vyžaduje [canonical](../glossary.md#term-canonical-form) input/output. Přesná [F64](../glossary.md#floating-point-precision-and-rounding) unary má vlastní affine read cestu. | [C1]<span data-glossary-navigation="true"> ([C1 glossary](../glossary.md#term-c1))</span> [M1] [M8]; R [transpose](../glossary.md#term-transpose)→NEG. |
| [Tensor](../glossary.md#tensor) [binary](../glossary.md#term-binary-descriptor-and-metadata) arithmetic | F32/F64 nebo I32/I64 promotion podle druhu, right-aligned broadcast; noninjective reads mohou být legální. | N promotion a přesné integral cesty; affine read musí splnit fyzický storage proof; canonical materializovaný output. Zero-stride shortcut v provideru není dostatečný důkaz ostatních os. | [C1] [M1] [M4]; R colliding read. |
| [Scalar](../glossary.md#term-scalar) arithmetic / clamp | Scalar [metadata](../glossary.md#term-metadata) přesně input dtype; floating clamp, bez integer clamp. | Stejný exact scalar convention, včetně F16 scalar metadata pro F16 scalar op. Nelze přenášet [norm](../glossary.md#term-norm) [epsilon](../glossary.md#epsilon) pravidlo na clamp. | [C1] [M1]. |
| Comparison / WHERE | CPU mixed-low není obecně přijat; floating a integer kategorie se nemíchají. | Přesné floating mixed comparisons a floating WHERE, ale external affine comparison se může zastavit až v [prepare](../glossary.md#prepare). | [C1] [M8] [M9] [M10]; R EQUAL ingress. |
| CAST / classification | Všech 49 cast párů; classification čte čtyři floating storage typy. | Totéž v přípustné fyzické doméně; žádné oprávnění k implicitní F64 arithmetic. | [C1] [M1] [M3]. |
| RESHAPE / [rank](../glossary.md#term-rank) edits / PERMUTE | Přesné view mapování, nebo materializující nekanonický RESHAPE; zachová původní logické pořadí. | View/[materialization](../glossary.md#materialization) doména také existuje; subsequent consumer může vyžadovat canonical input. | [C4] [C12] [M4]; R transpose→RESHAPE. |
| EXPAND / [CONTIGUOUS](../glossary.md#term-contiguous) | Low/raw forward možný, ale redukující [VJP](../glossary.md#term-vector-jacobian-product-vjp) může narazit na lock. | EXPAND má represented-empty výjimku; běžný CONTIGUOUS nemá automaticky všechny empty varianty. | [C4] [M3] [M4]; R low EXPAND a jeho VJP. |
| SELECT / signed [SLICE](../glossary.md#slice) | Checked [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix); negative-step nebo empty [materializace](../glossary.md#materialization) nevytváří negative physical strides. | Podobné odlišení logical signed slice od physical strides, ale empty slices mají vlastní stricter predicate (např. explicitní start=0). | [C4] [M3]. |
| SLICE_UPDATE / [PAD](../glossary.md#padding) / TILE | Širší statické prázdné a resolved-read domény podle rodiny; update funkcionální, nikoli in-place. | Canonical výsledky; řada retained read predicates vyžaduje kladné extenty, tedy ne obecné empty update/pad/tile. | [C4] [M3]. |
| CONCAT / STACK | Bez stropu 16, ale semantic occurrences a output rank vstupují do generačního budgetu. | Dynamické operand refs, přímé disjunktní kopie do finálního výstupu; žádný původní 16-input cap. STACK navíc rank vstupu <16. | [C7] [M3]; R 101/102/257. |
| UNFOLD_AXIS / UNFOLD2D / UNFOLD3D | [Axis](../glossary.md#term-axis) všech 7 storage typů; [spatial](../glossary.md#term-spatial-axes) unfold čtyři floating storage typy. Exact geometry/[padding](../glossary.md#padding), canonical/[injective](../glossary.md#term-injective) materializace. | Podobná typová raw doména, ale rank a affine/positive geometry limity Metalu. UNFOLD_AXIS zvyšuje rank, vstup tedy nejvýše rank 15. | [C4] [M3]. |
| FOLD_AXIS / FOLD2D / FOLD3D | F32/F64 addition a integral axis fold; low addition zamčená i při neoverlapujícím okně. | N overlap podporován; F64 a integral FOLD_AXIS jen nonoverlap, F64 spatial fold jen nonoverlap každé osy. | [C3] [C5] [M3]. |
| [GATHER](../glossary.md#gather) / [ELEMENTS](../glossary.md#term-element) / [ND](../glossary.md#term-nd) / ONE_HOT | Všech 7 data typů, I32/I64 indexy, exact bounds; ONE_HOT má [BOOL](../glossary.md#data-type--datatype) výstup. | Totéž pod rank/positive-layout/uint podmínkami. GATHER_ND [tuple depth](../glossary.md#term-tuple-depth) a [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix přesně kontrolované. | [C3] [M3]. |
| SCATTER modes | NONE všechny storage typy, MIN/MAX šest numeric, ADD/MUL F32/F64/I32/I64; uniqueness pro replacement ověřena před writes. | N všechny modes; integral [modular arithmetic](../glossary.md#term-modular-arithmetic); F64 NONE a vybrané MIN/MAX, ne ADD/MUL. Arithmetic scatter má zvláštní [uint32](../glossary.md#index-mappings-and-cost-dominance) byte-span cap. | [C3] [M3]. |
| SORT / ARGSORT / TOP_K | Všech 7 storage typů, canonical decisions a I64 indexy; k=0 dovoleno. | Canonical inputs/outputs, uint32 [extent](../glossary.md#term-extent)/count; TOP_K vyžaduje k>=1. | [C3]; Metal provider 445–546; R k=0. |
| [MATMUL](../glossary.md#matrix-multiplication--matmul) | Vektory, [matice](../glossary.md#axis-rank-scalar-vector-and-matrix), broadcast batches; resolved offsets/strides, K=0; nejde o backendový rank-12 native cap. | Rank<=16, positive extents, zero-offset dense nebo exact transpose-of-dense. Transpose external [descriptor](../glossary.md#term-descriptor) není autentizovaná lokální transpose [provenance](../glossary.md#provenance). | [C3] [M2] [M10]; R K0, offset, external transpose. |
| [CONV2D](../glossary.md#convolution--conv2d) / [CONV3D](../glossary.md#conv3d--ncdhw) | Promoted F32/F64 role, grouped/depthwise, strided/offset reads v rodinné doméně, injective output. | N promotion; image/[weight](../glossary.md#term-weight) a [bias](../glossary.md#term-bias) canonical nebo explicitní [singleton](../glossary.md#term-singleton)-height view výjimka; canonical output a uint32 geometry. | [C2]<span data-glossary-navigation="true"> ([C2 glossary](../glossary.md#term-c2))</span> [M2]. |
| MAX / AVERAGE_POOL2D/3D | Exact floor/literal-ceil/padding/[dilation](../glossary.md#term-dilation), širší read layouts; low max forward samostatně kvalifikován. | **Všechny čtyři** [pooling](../glossary.md#term-pooling) kinds mají [kernel](../glossary.md#kernel) volume<=65 536, nejen MAX. F64 max s grad navíc nonoverlap; N overlap podporován. | [C3] [M2]; R 66 049→1 output. |
| SUM / [MEAN](../glossary.md#term-mean) / PROD / masked / [SUM-to-shape](../glossary.md#sum-to-shape) | Full/single/multiaxis, specifické empty [identities](../glossary.md#term-identity-meanings); floating exact-state workspace může omezit [přípravu](../glossary.md#prepare). | N, vybrané exact integral cesty; ordinary reduction physical proof obecně vyžaduje neprázdnou geometrii a uint32 physical bytes. [Scan](../glossary.md#general-differentiable-scan)-local empty proof je zvláštní. | [C2] [M5]; R empty SUM. |
| MIN/MAX/ALL/ANY/ARG_MIN/MAX | Typované [identity](../glossary.md#term-identity-meanings) a exact predicates; arg vyžaduje positive selected extent. | Storage-exact domény existují, ale canonical/uint a empty podmínky jsou užší; ne-N redukční grad role nejsou blanket povoleny. | [C2] [M5]; Metal provider 499–546,1973–2012. |
| LSE / [variance](../glossary.md#term-variance) / std / [L1](../glossary.md#term-l1) / [L2](../glossary.md#term-l2) | F32/F64; variance/std count>correction i při jinak empty output. | N; affine reads/canonical output a physical-byte proof; stejný count/correction [guard](../glossary.md#term-guard). | [C2] [M5]. |
| CUM_SUM / CUM_PROD | F32/F64/I32/I64; oba směry a inclusive/exclusive; ordinary zero extent možný. | N/I32/I64, canonical positive-rank input/output, běžně positive geometry; žádná F64 cumulative arithmetic. | [C2] [M1]. |
| [SOFTMAX](../glossary.md#softmax--log-softmax) / LOG_SOFTMAX | F32/F64, positive selected extent, **[runtime](../glossary.md#term-runtime) [finite](../glossary.md#term-finite) input**. | N, zvláštní empty domény a širší [Model](../glossary.md#term-model) special-value semantics. | [C2] [C8] [M6] [M12]; R obousměrné rozdíly. |
| LayerNorm / RMSNorm | F32/F64 promotion, exact trailing normalized suffix a affine shape; resource declarations samostatně. | N promotion, finite positive epsilon; nonempty row [dispatch](../glossary.md#term-dispatch) uint32. F16 výsledky mají Model-[normalization](../glossary.md#term-normalization) F32 scalar metadata. | [C2] [M6]. |
| BatchNorm [inference](../glossary.md#term-inference)/training | Pět input rolí; training pět output rolí, C==0 nebo contributor count>=2; samostatné grad flags. | Stejné role/sibling equations v N doméně, canonical outputs a [channel](../glossary.md#term-channel) dispatch. Saved auxiliaries nejsou volné [AD](../glossary.md#autograd) roots. | [C2] [M6]. |
| [MSE](../glossary.md#mean-squared-error--mse) | Exact stejné [tvary](../glossary.md#shape) [prediction](../glossary.md#term-prediction)/[target](../glossary.md#term-target), ne target broadcast; empty SUM=+0, MEAN=[NaN](../glossary.md#term-nan). | N promotion, oba operands canonical, nikoli libovolná affine MSE čtení; empty doména není plošně zakázaná. | [C2] [M6]. |
| Dense / index cross entropy | F32/F64, class-axis geometry; ignore porovnání před bounds; bez plošného finite-logits validatoru softmaxu. | N, vlastní zero-class/[sample](../glossary.md#term-sample-machine-learning-and-hyperparameters)/ignore domain. Empty/fully-ignored mean není zakázán pouze kvůli nulovému included count. | [C2] [M6]. |
| Attention | F32/F64, Q/K/V a [mask](../glossary.md#term-boolean-mask) equations; aktuální zero+singleton batch broadcast bug. | N, zvlášť odvozené score/output geometrie a workspace; empty primary může obejít nepotřebné score scratch, ne validaci. | [C3] [M7]; R zero batch. |
| [RNG](../glossary.md#graph-rng-state--graphrngstate) / [dropout](../glossary.md#dropout--inverted-dropout) | Explicit state, finite probability, F32/F64; CPU kept aritmetika je nález ze sekce 4. | N, explicit mask/next state, canonical geometry a uint32 bounds. | [C3] [T2] [M14]. |

Conv1d/Pool1d a [`linear`](../glossary.md#term-linear-meanings) nejsou další ordinary kindy: jsou to kompozice existujících operací. Jejich přenositelnost zahrnuje všechny vnitřní rank edits/views, kontrakce/pooling a vybraný backward; nestačí přítomnost Conv2d nebo MATMUL v katalogu.

## 7. Symbolické rozměry, GeneralScan, recurrence a [AD](../glossary.md#autograd)

### 7.1 Ordinary changing-B: nová veřejná exekuce

**R:** skutečný veřejný `Engine.compile`→jedno [`prepare`](../glossary.md#prepare)→opakované `runValues` při B=2,0,3:

| Symbolický [graf](../glossary.md#graph) `[B,2]` | [CPU](../glossary.md#term-cpu) | [Metal](../glossary.md#term-metal) |
|---|---|---|
| [F32](../glossary.md#floating-point-precision-and-rounding) ADD | Všechny tři B, správné hodnoty i [runtime](../glossary.md#term-runtime) [shape](../glossary.md#shape) | [Compile](../glossary.md#compile) odmítne symboliku |
| F32 [MATMUL](../glossary.md#matrix-multiplication--matmul) s pevnou `[2,2]` vahou | Všechny tři B, stejná [příprava](../glossary.md#prepare) | Compile odmítne |
| F32 [TANH](../glossary.md#term-tanh) | Všechny tři B, nezávislý numerický [oracle](../glossary.md#term-test-oracle) | Compile odmítne |
| [F64](../glossary.md#floating-point-precision-and-rounding) ADD | Všechny tři B, správné hodnoty | Compile odmítne |
| F64 MATMUL a TANH | Compile odmítne, přestože statické varianty CPU umí | Compile odmítne |

B se váže z nových plně statických caller-input descriptorů, ne přes nový [argument](../glossary.md#term-argument) `prepare`. Používá se `runValues`, ne exact-ID [`run`](../glossary.md#term-run) se symbolic [prototype](../glossary.md#term-prototype) [descriptor](../glossary.md#term-descriptor). Public [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) plan vyžaduje CPU-only [ownership](../glossary.md#term-ownership-meanings), jeden společný [symbol](../glossary.md#term-symbol) odvozený z caller vstupu, zachované typy/ranky/grad flags, přesné fixed input descriptors a zero-offset [canonical](../glossary.md#term-canonical-form) [strides](../glossary.md#term-stride) symbolic inputů. Shape/stride/count/span musí projít int/byte [admission](../glossary.md#term-admission) i za prázdnou osou. [C6] [E3] [E4]

CPU ordinary symbolic [matice](../glossary.md#term-review-matrix) je záměrně omezená, převážně [rank](../glossary.md#term-rank)<=3 se zvláštními pravidly. F32 má MATMUL, vybrané [binary](../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../glossary.md#term-scalar) arithmetic, NEG/TANH/[SIGMOID](../glossary.md#term-sigmoid), SUM/[MEAN](../glossary.md#term-mean), nutné [layouty](../glossary.md#layout) a cast. F64 typed větev je menší (ADD/MUL/DIV, potřebné redukce/layouty/cast); statická podpora F64 nepřechází automaticky na symbolickou. F64 NEG/RELU/mixed MATMUL uvnitř širší CPU [Scan](../glossary.md#general-differentiable-scan) composition nejsou důkazem stejné standalone ordinary admission. [C6] [S1] [Q2]

### 7.2 GeneralScan — odlišný body i binder průnik

| Rozměr | [CPU](../glossary.md#term-cpu) | [Metal](../glossary.md#term-metal) | Důkaz |
|---|---|---|---|
| [Shape](../glossary.md#shape) | Static a kvalifikované one-named-B closure; ne libovolné expressions/více symbolů | Všechny local/boundary shapes static | S [S1] [S4], public symbolic zdroje [Q2]. |
| Typy | Per-value types bez univerzální [homogeneous](../glossary.md#term-homogeneous) floating podmínky; jednotliví ordinary ownery a low lock stále platí | Whitelist [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding)/[BOOL](../glossary.md#data-type--datatype)/I32/I64; **žádný [F64](../glossary.md#floating-point-precision-and-rounding) local/boundary value** | R F64 raw [identity](../glossary.md#term-identity-meanings); S [S1] [S3] [S4]. |
| Operace body | Nejen starý ADD/SUB/MUL/DIV/NEG/[TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid)/[MATMUL](../glossary.md#matrix-multiplication--matmul)/SUM slice; při neúspěchu legacy analysis ordinary composition fallback | Každý actual body [node](../glossary.md#node) projde ordinary semantics, typed ownerem a native [ABI](../glossary.md#term-application-binary-interface-abi) | S [S2] [S3]. |
| [Carry](../glossary.md#term-carry)-free / zero-ROW length | Generic composition je umí v kvalifikovaných rolích | Source/binder je obecně nevylučuje | S; neznamená cold/save podporu. |
| Direct read [layout](../glossary.md#layout) | Generic statická composition uchovává actual offset/[strides](../glossary.md#term-stride) a mapovaný ingress | Final [Scan](../glossary.md#general-differentiable-scan) input fyzicky [canonical](../glossary.md#term-canonical-form) represented storage; materialized-layout feed může [prepare](../glossary.md#prepare) odmítnout | R strided F32 Scan; S [S1] [S4]. |
| Směr a lengths | Oba směry; [runtime](../glossary.md#term-runtime) [INT64](../glossary.md#data-type--datatype) lengths v `[0,T]` před Scan-local writes | Totéž; native time loop a once-prepared children | S; rozsah starších numerických průchodů H. |
| Prázdno/T0/inaktivita | Structural body kontrola neodpadá; T0 zachová carry, inactive rows nečtou neplatné operandy | Stejný požadavek, ne blanket empty ordinary unlock | S; low lock nezmizí T0. |
| Velikost body | Legacy 64-formal/128-node/256-value limity nejsou automaticky limity composition fallbacku; generované unity mají vlastní budgety | 4 096 values, 512 units, 1 048 576 encoded words plus physical/workspace bounds | S [S2] [S5]. |

Důležitá oprava starších popisů: CPU Scan není nyní univerzálně homogeneous F32/F64, dense-zero-offset ani carry-required. Některé širší named-F64/mixed/empty body numerical testy jsou ale [backend](../glossary.md#backend)-private. Jejich přesnou veřejnou [Engine](../glossary.md#term-engine) dosažitelnost nelze vydávat za již provedený veřejný průchod; external public symbolic length-Scan F32/F64 má vlastní integrační zdroj [Q2].

### 7.3 Co omezuje společný [Compiler](../glossary.md#term-compiler), nikoli jeden [backend](../glossary.md#backend)

- General/fixed recurrence má vybraný první [AD](../glossary.md#autograd); vyšší řád je společně odmítán. Vyšší-order request s původním GeneralScan může být odmítnut i kvůli jinak vedlejší původní větvi. [S7]
- U původního length-aware GeneralScan AD musí všechny původní carries a sequences splňovat ROW(0), včetně nepoužitých/nongradient declarations. [Model](../glossary.md#term-model)-valid forward s ROW(1), SHARED sequence nebo terminal SHARED [carry](../glossary.md#term-carry) proto není automaticky diferencovatelný. Generated reverse [Scan](../glossary.md#general-differentiable-scan) má jiné povolené history/[seed](../glossary.md#term-seed)/terminal role. [S6]
- Délky a discrete [mask](../glossary.md#term-boolean-mask)/index/state nemají [cotangent](../glossary.md#cotangent). Saved [residual](../glossary.md#scan-residual-port-and-history-demand) se nesmí stát samostatným cotangent rootem; [primal](../glossary.md#term-primal)-only použití např. ve WHERE condition je jiná věc.
- BatchNorm saved [mean](../glossary.md#term-mean)/inverse-std, attention [weights](../glossary.md#term-weight) a ostatní [canonical](../glossary.md#term-canonical-form) siblings mají vlastní role a derivability, ne libovolný Cartesian součin output×input.
- [RNG](../glossary.md#graph-rng-state--graphrngstate)/[dropout](../glossary.md#dropout--inverted-dropout)/effectful operace, hidden leaves, [host](../glossary.md#term-host)-backed body interiors a vnořené recurrence nepatří automaticky do [PURE](../glossary.md#term-pure) Scan jen proto, že běží samostatně.

### 7.4 Fixed [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm): [Engine](../glossary.md#term-engine) [AD](../glossary.md#autograd) versus [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions)

**Engine-level první AD není na [CPU](../glossary.md#term-cpu) chybějící funkce.** Čerstvý katalog ho skutečně provedl. Current public test source dále provádí selected [weight](../glossary.md#term-weight) [gradients](../glossary.md#term-gradient) všech tří cells, obou směrů, [bias](../glossary.md#term-bias)/no-bias a [layout](../glossary.md#layout) variants v CPU [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding); to není totéž jako nové all-role [finite-difference](../glossary.md#term-finite-difference-gradient-check) ověření v tomto auditu. [Q3]

CPU forward je F32/F64 s kvalifikovanými resolved reads; [Metal](../glossary.md#term-metal) F32/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) s [canonical](../glossary.md#term-canonical-form) physical [descriptors](../glossary.md#term-descriptor) všech rolí včetně lengths. Společný [Compiler](../glossary.md#term-compiler) staví cell transition a ordinary [Scan](../glossary.md#general-differentiable-scan) [VJP](../glossary.md#term-vector-jacobian-product-vjp), nikoli oddělený handwritten [backend](../glossary.md#backend) [adjoint](../glossary.md#term-adjoint). [S8] [S9] [S10]

**S:** `Model.configureTraining/train` má přesto vlastní [guard](../glossary.md#term-guard) proti jakékoliv fixed recurrence, i ve smíšeném fixed/general modelu. `ModelBuilder.rnnScan` zvolí GeneralScan pro F32/F64, ale pro BF16/F16 zachová fixed; `gruScan/lstmScan` zachovají fixed. Metal fixed-cell AD tedy **neznamená**, že všechny low owned recurrent buildery lze trénovat přes tento owned Model surface. [Objective](../glossary.md#term-objective) navíc musí být [scalar](../glossary.md#term-scalar) grad-eligible F32/F64. [N1] [N2]

Explicitní `RnnSequence.forwardGeneralScan` může vytvořit low ordinary-cell recipe. Ten má jiné [rounding](../glossary.md#term-rounding) boundaries než fixed low F32-working recipe: není to kompatibilní náhradní implementace stejného zdroje. Tvarově stejné váhy/délky proto nemusejí dát shodné low hodnoty. [N3]

## 8. [Persistence](../glossary.md#term-persistence), přechod na jiný [backend](../glossary.md#backend) a pokračování tréninku

**S:** [Graph](../glossary.md#graph) [checkpoint](../glossary.md#term-checkpoint) ukládá zdrojový recept, [descriptors](../glossary.md#term-descriptor), operace/atributy, [scope](../glossary.md#term-scope)/ports/[aliases](../glossary.md#aliasing-and-arena) a raw named state. Zachová [F64](../glossary.md#floating-point-precision-and-rounding)/[F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding)/I32/I64/[BOOL](../glossary.md#data-type--datatype) typy, nikoli automatickou konverzi do typů nového [backendu](../glossary.md#backend). [Rebind](../glossary.md#capture-graph-ports-and-reconstruction) vyžaduje full descriptor equality. [P2] [P3]

- Formáty v1/v2 nesou starší flat domény, v3 [CNN](../glossary.md#term-convolutional-neural-network-cnn)/fixed recurrence, v4 all-valid GeneralScan, v5 lengths, v6 [Pool3d](../glossary.md#pool3d--ncdhw-pooling) (včetně contained). Backend switch není změna wire verze ani změna fixed/general [identity](../glossary.md#term-identity-meanings).
- [Cold reconstruction](../glossary.md#term-cold-reconstruction) je užší než compute: static [shapes](../glossary.md#shape), uzavřený constructor/[opcode](../glossary.md#term-opcode) inventory, nonempty [carry](../glossary.md#term-carry) a pro length [Scan](../glossary.md#general-differentiable-scan) pozitivní ROW-carry count. Carry-free či zero-ROW compute podpora tedy nezaručuje save/load.
- Length cold role zůstávají floating, ale současné executable guardy již nejsou univerzálně [homogeneous](../glossary.md#term-homogeneous)-F32/F64-only; per-role mixed a low-storage/F32-working recipes existují. Povolené row/[layout](../glossary.md#layout) transformace a [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) [contraction](../glossary.md#contraction-dimension) jsou dál omezené. [P1]
- `Model.loadGraph(..., targetEngine)` nekontroluje při samotném load dostupnost/[prepare](../glossary.md#prepare) každé operace na zvoleném backendu. První compute znovu [compile](../glossary.md#compile)/prepare. F64 soubor může jít načíst a potom selhat na [Metal](../glossary.md#term-metal); low numerický soubor analogicky na [CPU](../glossary.md#term-cpu).
- Externí [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) a lengths je třeba znovu dodat, pokud nebyly výslovně named state. Není obnoveno [objective](../glossary.md#term-objective), [optimizer](../glossary.md#term-optimizer) slots/step count, [AD](../glossary.md#autograd) history, připravený plán, [GPU](../glossary.md#term-gpu) kontext nebo compiled [artifact](../glossary.md#term-artifact).
- Fresh configured training po load není optimizer continuation a bitově identická trajektorie tréninku na jiném backendu není slib.

Zpráva neprovedla nový cross-backend save/load [matrix](../glossary.md#term-matrix-meanings); tyto odlišné hranice jsou source-backed. Dřívější izolované CPU/Metal save/load/nové-trénování receipts jsou H, nikoli nový důkaz libovolné migrace.

## 9. Resource, transport a [runtime](../glossary.md#term-runtime) rozdíly

### 9.1 Neexistuje jeden univerzální maximum [tensor](../glossary.md#tensor) size

| Limit / vztah | [CPU](../glossary.md#term-cpu) | [Metal](../glossary.md#term-metal) / společná mez |
|---|---|---|
| [Rank](../glossary.md#term-rank) | Ordinary static provider nemá univerzální rank-16 strop; rodinné/generační limity stále existují | Pervasive rank<=16; některé rank-zvyšující operace potřebují input<=15. R rank17 ADD. |
| Velikost/span | Checked long geometry, [heap](../glossary.md#term-heap) arrays [JVM](../glossary.md#term-java-virtual-machine-jvm)-int limited, segment carriers mohou mít long addressing; není to příslib alokovatelnosti | Family-specific [uint32](../glossary.md#index-mappings-and-cost-dominance) geometry/count/span/byte limity. **Není** jeden plošný „4GB [backend](../glossary.md#backend) limit“. |
| [Pooling](../glossary.md#term-pooling) | Checked geometry/product bez uvedeného Metal [kernel](../glossary.md#kernel) capu | Všechny čtyři pooling kinds kernel product<=65 536. |
| Scatter arithmetic | Exact resources a span/[alias](../glossary.md#aliasing-and-arena) checks | Vybrané logical/physical byte spans<=4 294 967 295, včetně offsetů/holes. |
| [Variadic](../glossary.md#term-variadic) CPU movement | `occurrences+7<=254`; nad 16 navíc `(occurrences+1)*(outputRank+4)<=512` | Metal nezdědil tento Java entry budget; má vlastní encoded/ref/resource limity. [C7] |
| [Injectivity](../glossary.md#storage-mapping-contiguity-and-injectivity) proof | Span/residue proof + exact fallback nejvýše 1 000 000 prvků; větší neprokázaný [layout](../glossary.md#layout) může být odmítnut i pokud matematicky [injective](../glossary.md#term-injective) | Nonbroadcast osy storage proof nesmějí kolidovat; materializing outputs obvykle přesně [canonical](../glossary.md#term-canonical-form). [C11] [M4] |
| Workspace | Exact reductions/PROD limbs/scatter/order/affine address tables mohou růst s logickým rozsahem, ne jen výstupem | Attention [F32](../glossary.md#floating-point-precision-and-rounding) score work, [Scan](../glossary.md#general-differentiable-scan) locals/scratch/control a další declared workspaces; low publikované [weights](../glossary.md#term-weight) nenahrazují F32 working probabilities. |
| [Host](../glossary.md#term-host) materialize | Oba [backendy](../glossary.md#backend) vracejí canonical [row-major](../glossary.md#term-row-major) **[big-endian](../glossary.md#text-encoding-and-file-boundaries)** bytes; jeden payload musí vejít do `byte[]` | Tedy nejvýše `Integer.MAX_VALUE` payload bytes. `maximumTotalBytes` není limit peak procesu, staging ani všech [runtime](../glossary.md#term-runtime) [buffers](../glossary.md#term-buffer). [E5] |

Příklady unproven obří injectivity či multi-GiB allocations nebyly alokovány. `[INFERENCE]` z postačujícího proofu není skutečně vyzkoušená hranice zařízení. Native `newBufferWithLength` může selhat i pro [descriptor](../glossary.md#term-descriptor), který projde integer geometrickou validací.

### 9.2 [Host](../glossary.md#term-host) [borrowing](../glossary.md#term-borrowing) a mixed-[owner](../glossary.md#term-owner) transport

[CPU](../glossary.md#term-cpu) intrinsic borrow drží skutečný non-owning host segment s kontrolou původního carrieru, velikosti, živosti a accessibility. [Metal](../glossary.md#term-metal) borrow znamená upload kopie do owned Metal bufferu; zachování reference na [host storage](../glossary.md#host-storage) není koherentní sdílená [residency](../glossary.md#residency). Mutace původního host bufferu sama neaktualizuje již vytvořenou Metal reprezentaci. Nové [Engine](../glossary.md#term-engine) bindingy/čerstvé inputy jsou samostatná věc. [E6]

Cross-owner CPU↔Metal transfer je užší než oba lokální [backends](../glossary.md#backend): static [rank](../glossary.md#term-rank)<=16, positive extents/[strides](../glossary.md#term-stride) a nonoverlap proof, checked [referenced span](../glossary.md#referenced-element-span), CPU native staging místo [heap](../glossary.md#term-heap) carrieru. Přenáší celý referenced span včetně prefix/holes bez konverze. Empty, unresolved nebo zero-stride broadcast boundary nepodporuje. [Graph](../glossary.md#graph) tedy může mít každý [node](../glossary.md#node) lokálně přijatelný a přesto selhat na rozhraní mezi vlastníky. [E2]

[Run](../glossary.md#term-run) resources a publikace mají vlastní [lifetime](../glossary.md#term-lifetime)/[leases](../glossary.md#term-lease). Caller storage musí být live/accesssible bez konfliktní mutace až do [synchronous](../glossary.md#term-synchronous) completion. Shareable prepared plan není slib nezávislého ordering/fairness/throughputu nebo souběžného [GPU](../glossary.md#term-gpu) overlapu; Metal kontext má jednu [command queue](../glossary.md#term-command-queue). CPU default má [concurrency](../glossary.md#term-concurrency) budget 1; obecná existence [vector](../glossary.md#term-vector)/parallel strategií není jejich implicitní zapnutí v každém programu.

## 10. [Výběr vlastníka](../glossary.md#backend-ownership), native cesty, platformy a [cache](../glossary.md#term-cache)

**S:** `Engine.standard()` registruje pouze [CPU](../glossary.md#term-cpu). `Engine.builder()` začíná bez [backendu](../glossary.md#backend). Při více registracích bez jiné class preference vyhraje první hard-eligible provider pro výskyt; s preferencí se hledá první eligible provider dané třídy. [Metal](../glossary.md#term-metal)-first+CPU tedy může naplánovat smíšený [graf](../glossary.md#graph) bez zvláštního fallback přepínače. To je **plánovaná composition**, nikoli [runtime](../glossary.md#term-runtime) retry. Metal-only nemá odkud CPU výpočet získat a [run](../glossary.md#term-run)/[prepare](../glossary.md#prepare) selhání nezahájí fallback. [E1] [E8]

| Oblast | CPU | Metal |
|---|---|---|
| Základní deploy | [JDK](../glossary.md#term-java-development-kit-jdk) 26; [scalar](../glossary.md#term-scalar) nepotřebuje incubator [Vector](../glossary.md#term-vector) [module](../glossary.md#term-module); native probing vyžaduje správné [FFM](../glossary.md#term-ffm) permission | JDK 26 + dodaná [arm64](../glossary.md#native-packaging-and-platform-vocabulary) macOS native knihovna; checked-in build deployment minimum macOS 26.0, Xcode a systémové frameworky. [E7] |
| Native získání | [OpenBLAS](../glossary.md#term-openblas) optional bounded discovery/[qualification](../glossary.md#term-qualification); neúspěch nechá portable CPU. Explicitní [oneDNN](../glossary.md#term-onednn) load je required a chybu neskrývá | Explicitní absolute path; aktuální [ABI](../glossary.md#term-application-binary-interface-abi)/[schema](../glossary.md#term-schema)/semantic [identity](../glossary.md#term-identity-meanings); system-default Metal [device](../glossary.md#term-device), nikoli libovolně přenositelný připravený kontext |
| Co je další [backend](../glossary.md#backend) | OpenBLAS/oneDNN **nejsou** další [backendy](../glossary.md#backend): jsou CPU-internal routes | Custom [MSL](../glossary.md#term-metal-shading-language-msl)/[MPSGraph](../glossary.md#term-mpsgraph) jsou také routes téhož Metal vlastníka |
| OpenBLAS | [Homogeneous](../glossary.md#term-homogeneous) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) nonbatched 2D [MATMUL](../glossary.md#matrix-multiplication--matmul) K>1, int ABI a vlastní kvalifikace; nikoli omezení všech CPU matmulů | Nemá roli Metal fallbacku |
| oneDNN | Explicitní 3.14.0, [LP64](../glossary.md#term-lp64) [little-endian](../glossary.md#text-encoding-and-file-boundaries), SEQ; qualified homogeneous F32 bare MATMUL K>1, normalized dense/permuted [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) do native [rank](../glossary.md#term-rank) 12 | Nezvyšuje Metal capability ani jeho rank/[layout](../glossary.md#layout) doménu |
| [API](../glossary.md#term-application-programming-interface-api) preference | AUTO/PRIMITIVES/[GRAPH](../glossary.md#graph) omezuje kandidáty, nevnucuje native execution; whole-plan dimensionless heuristika, ne naměřená latence | Existence MPSGraph sama nedokazuje libovolný [occurrence](../glossary.md#term-occurrence) support |
| Caching | Generated ClassFile [weak interning](../glossary.md#weak-reference-and-weak-interning), případně oddělená trusted-root cache class bytes; ne [JIT](../glossary.md#term-just-in-time-compilation-jit) strojový kód ani přenositelné prepared native handle | Bounded session-local tuning/cache, ABI/schema/source identity; ne [portable executable](../glossary.md#native-packaging-and-platform-vocabulary) uložený checkpointem |
| Native distribuce | oneDNN není bundlováno; OpenBLAS library/ABI musí projít kvalifikací | Lokálně source-built/ad-hoc signed package, nikoli publisher-authenticated/notarized veřejný prebuilt |

Výkonová zaměnitelnost, stejná peak memory, stejný thread count ani stejné startup/prepare náklady z toho neplynou. V tomto auditu nebyla znovu měřena rychlost. Dřívější report-only benchmark zůstal oddělený od capability rozhodnutí.

## 11. Priority nálezů a návrh dalších kroků

### P1 — správnost / jasná hranice přijetí

1. **[CPU](../glossary.md#term-cpu) RSQRT a [dropout](../glossary.md#dropout--inverted-dropout):** zachovat předepsaná pracovní místa ve společných emitters, nikoli přidávat value special-cases nebo povolit širší [epsilon](../glossary.md#epsilon). Přidat cílený failing-before/passing-after [oracle](../glossary.md#term-test-oracle) při vlastní opravě; tento audit permanentní testy ani [runtime](../glossary.md#term-runtime) nemění.
2. **CPU attention zero+[singleton](../glossary.md#term-singleton) [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters):** opravit broadcast predicate (`Math.max` nesmí přepsat ne-one nulu na 1); nález má zdroj i veřejný R counterexample.
3. **[Metal](../glossary.md#term-metal) [descriptor](../glossary.md#term-descriptor)/[prepare](../glossary.md#prepare)/native rozhraní:** sjednotit podporu/[provenance](../glossary.md#provenance) nebo explicitně přesně vymezit external affine comparisons, external [transpose](../glossary.md#term-transpose) [MATMUL](../glossary.md#matrix-multiplication--matmul) a external broadcast exact [owner](../glossary.md#term-owner). Oddělit úmyslný dense/provenance ingress od nesouladu validátorů; neskrývat native INVALID_ARGUMENT fallbackem.

### P2 — největší praktické mezery portability

4. Doplnit průnik běžných empty domén (NEG/SUM/K0 [MATMUL](../glossary.md#matrix-multiplication--matmul)/TOP_K k0/[softmax](../glossary.md#softmax--log-softmax)) po rodinách, ne globálním obcházením preflightu. [Scan](../glossary.md#general-differentiable-scan)-local empty podpora se nesmí prohlásit za ordinary podporu bez správné closure.
5. Rozšířit [Metal](../glossary.md#term-metal) [unary](../glossary.md#term-unary)/kontrakční physical read domény nebo sdílenou plánovanou [materializaci](../glossary.md#materialization). Dnešní [canonical](../glossary.md#term-canonical-form) vstupy samy nestačí, protože i legitimní mezikrok PERMUTE vytvoří [view](../glossary.md#term-view-meanings).
6. Vyřešit [CPU](../glossary.md#term-cpu) [variadic](../glossary.md#term-variadic) movement budget bez nového N-wide [JVM](../glossary.md#term-java-virtual-machine-jvm) entry limitu. Neuvádět „bez limitu“ místo současného závislého budgetu; nejde o [tensor](../glossary.md#tensor)-row limit.
7. Ujasnit [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) fixed-recurrence training surface vůči už existujícímu [Engine](../glossary.md#term-engine) [AD](../glossary.md#autograd); explicitně oddělit fixed a general low [rounding](../glossary.md#term-rounding) recipes.
8. Symbolický Metal/general-Scan rozsah a širší cold/[persistence](../glossary.md#term-persistence) doménu plánovat samostatně; nevyvozovat je z ordinary [F32](../glossary.md#floating-point-precision-and-rounding) katalogu.

### Záměrné rozdíly, které audit nenavrhuje potají odemknout

- [CPU](../glossary.md#term-cpu) [BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) aritmetický zámek je výslovná uživatelská produktová volba, ne opomenutí k tiché opravě.
- [Metal](../glossary.md#term-metal) obecná [F64](../glossary.md#floating-point-precision-and-rounding) aritmetika není nynější cíl; exact/raw F64 podpora nemá být nahrazena nepřiznanou [F32](../glossary.md#floating-point-precision-and-rounding) aproximací.
- [Rank](../glossary.md#term-rank)/body/resource bounds a platform packaging je třeba zveřejnit a případně promyšleně rozšířit, nikoli odstranit checked [overflow](../glossary.md#term-overflow) či [alias](../glossary.md#aliasing-and-arena) validaci.
- Žádný nový [runtime](../glossary.md#term-runtime) fallback, implicitní low emulace, druhý [AD](../glossary.md#autograd) [engine](../glossary.md#term-engine) ani alternativní numerical policy.

## 12. Dokumentační rozpory a zbylé neověřené domény

Potvrzené stale popisy nejsou totéž co chybějící implementace:

- `production-support.md` ještě uvádí CPU0079 noncontiguous RESHAPE jako omezení; R [transpose](../glossary.md#term-transpose)→materializing reshape prošel na [CPU](../glossary.md#term-cpu) i Metalu.
- Starší [Scan](../glossary.md#general-differentiable-scan) a CPU provider komentáře popisují užší [homogeneous](../glossary.md#term-homogeneous)/dense/[carry](../glossary.md#term-carry)-required/symbolic slice než aktuální generic composition. Je nutné sledovat fallback, nikoli jen první legacy predicate. [S1] [S2]
- Popisy cold homogeneous [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding)-only neodpovídají nynější per-role low/F32-working floating doméně. To ale neodstraňuje static/closed-constructor/ROW/carry limity. [P1]
- „Fixed recurrent je forward-only“ neplatí pro [Engine](../glossary.md#term-engine)/[Compiler](../glossary.md#term-compiler) první [AD](../glossary.md#autograd); pro [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) train [guard](../glossary.md#term-guard) je však omezení skutečné. [S8] [N2]
- Některé [Metal](../glossary.md#term-metal) komentáře stále říkají F32-only overlapping folds/pool cotangents; implementace používá shared N vlastníka. Jiné starší komentáře uvádějí schema21/codec40 místo aktuálního schema22/codec43.
- `Engine.standard()` prose „without discovery“ nezachycuje současné optional [OpenBLAS](../glossary.md#term-openblas) discovery/[qualification](../glossary.md#term-qualification); stálý [backend](../glossary.md#backend) [owner](../glossary.md#term-owner) je přesto CPU-only.

**Neprovedeno nově:** jiné [GPU](../glossary.md#term-gpu)/OS/[JDK](../glossary.md#term-java-development-kit-jdk), x86 audit celé této nové boundary [matice](../glossary.md#term-review-matrix), obří allocations, úplný cross-backend [checkpoint](../glossary.md#term-checkpoint) migration [matrix](../glossary.md#term-matrix-meanings), každá mixed-type/[shape](../glossary.md#shape)/[layout](../glossary.md#layout)/AD kombinace, vyšší-order closure všech ordinary operací, všechny [tolerance](../glossary.md#term-tolerance)/transcendentální value sweeps a souběžné mixed CPU↔Metal transfery. Source-predicted external affine SCATTER_ADD state gap zůstává **[[INFERENCE](../glossary.md#term-inference-meanings)]**; není vydáván za vykonaný failure.

Předchozí CPU/Metal rozsáhlé numerické/trénovací receipts zůstávají H pro své přesné fixtures. Úspěšný katalog zde pouze potvrzuje, že přijaté reprezentanty provedly celý pipeline; **nelze z něj odvodit univerzální numerickou správnost ani úplnou zaměnitelnost**. [Q1]

## 13. Co lze dnes rozumně považovat za přenositelný základ

Konzervativní společný základ je **konkrétně kvalifikovaný** statický [F32](../glossary.md#floating-point-precision-and-rounding) program, [rank](../glossary.md#term-rank)<=16, s podporovanými materialized/intermediate [layouts](../glossary.md#layout), rodinnou empty doménou, platnými indexy/maskami, [finite](../glossary.md#term-finite) standalone [softmax](../glossary.md#softmax--log-softmax) inputy, pool/window/resource budgety a podporovaným vybraným backward uzávěrem. Je nutné prověřit také vnitřní [views](../glossary.md#term-view-meanings) a [AD](../glossary.md#autograd)-created operations, ne jen vstupní [tensors](../glossary.md#tensor).

Ani tento popis není nová blanket certifikace všech takových programů: RSQRT/[dropout](../glossary.md#dropout--inverted-dropout) nálezy a další samostatné přípustnosti musí být vyřešeny či explicitně kvalifikovány. Doporučená portability kontrola vždy zkusí **oba isolated [backends](../glossary.md#backend)** přes [compile](../glossary.md#compile)→[prepare](../glossary.md#prepare)→[run](../glossary.md#term-run)→materialize a přesně deklaruje [numerical oracle](../glossary.md#adr-ci-and-qualification); samotné `supports()` ani dostupnost druhého [backendu](../glossary.md#backend) není náhradou.

## 14. Podklady a reprodukce

Lokální podklady tohoto auditu: `build/backend-interchangeability-audit/`:

- `catalog-{cpu,metal}.{jsonl,tsv,summary.json}` — nových 5 018 řádků na [backend](../glossary.md#backend).
- `catalog-differences.json`, `catalog-summary.json`, `audit-summary.json` — spárované klasifikace a měřené souhrny.
- `public-boundaries-{cpu,metal}.jsonl` — konečných 39 veřejných scénářů na backend s fází odmítnutí a raw oraclem.
- `numeric-sites-{cpu,metal}.jsonl` — RSQRT a [dropout](../glossary.md#dropout--inverted-dropout) typed-site údaje, skutečné masky/next-state.
- `symbolic-{cpu,metal}.jsonl` — veřejné changing-B průchody a odmítnutí.
- `commands.json`, `audit-environment.json`, `consumer-sources/` — přesné příkazy, prostředí a zdroje diagnostických spotřebitelů.
- `source-reviews/` — čtyři oddělené read-only zdrojové analýzy a upřesnění public symbolic cesty; nejsou samostatným [runtime](../glossary.md#term-runtime) důkazem.

Reprodukce používá existující current main/[fixture](../glossary.md#term-fixture) classes ze zdrojové kvalifikace commitu, [JDK](../glossary.md#term-java-development-kit-jdk) 26, samostatně registrovaného [CPU](../glossary.md#term-cpu) nebo Metalu a výslovnou native path. Diagnostické sources nejsou přidány do runtime implementace ani permanentní testovací sady. Počáteční diagnostický byte-order/[F16](../glossary.md#floating-point-precision-and-rounding)-fixture setup byl opraven podle existujících public [API](../glossary.md#term-application-programming-interface-api): [host](../glossary.md#term-host) [publication](../glossary.md#term-publication) je [big-endian](../glossary.md#text-encoding-and-file-boundaries) a raw F16 nese vlastní storage type. Prvotní setup chyby nejsou nálezy backendových numerických chyb.

## 15. Navazující opravy a aktuální hranice

Tato část nahrazuje doporučení oprav z oddílu 11 pro současný pracovní zdroj; původní měření
a 57 připnutých odkazů v oddílech 1–14 zůstávají historickým auditním snímkem `65c0eb0`.
[Model](../glossary.md#term-model) numerika se nemění a nevzniká druhý executor, [AD](../glossary.md#autograd) [engine](../glossary.md#term-engine) ani [runtime](../glossary.md#term-runtime) fallback.

### 15.1 Uzavřené nálezy

| Oblast | Oprava a ověřený výsledek |
|---|---|
| [CPU](../glossary.md#term-cpu) RSQRT | [F32](../glossary.md#floating-point-precision-and-rounding) [square root](../glossary.md#norms-elementary-functions-and-probability-distributions) se zaokrouhlí před F32 reciprocal. Sdílené emitters zachovávají toto pořadí ve [scalar](../glossary.md#term-scalar)/[vector](../glossary.md#term-vector)/remainder i složených tělech; veřejná typovaná sonda již nevykazuje původní rozdíl. |
| CPU [dropout](../glossary.md#dropout--inverted-dropout) | Typovaný complement → reciprocal → multiply nahrazuje přímé dělení vstupu. F32 i [F64](../glossary.md#floating-point-precision-and-rounding) sondy odpovídají nezávislému typovanému oraclu; masky a [RNG](../glossary.md#graph-rng-state--graphrngstate) next-state zůstaly stejné. |
| CPU attention | Broadcast nuly a singletonu zachová nulu místo `Math.max(0,1)=1`; platný nulový [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) se provede jako prázdný výstup. |
| CPU [softmax](../glossary.md#softmax--log-softmax) | Prázdná vybraná osa a [Model](../glossary.md#term-model) [NaN](../glossary.md#term-nan)/±[infinity](../glossary.md#term-infinity) doména již nejsou odmítnuty starým [finite](../glossary.md#term-finite)-only validátorem. Veřejné sondy ověřily SOFTMAX i LOG_SOFTMAX včetně velkých `[128,2048]` výstupů. |
| [Metal](../glossary.md#term-metal) fyzické reads | [Unary](../glossary.md#term-unary), comparisons a numerický [MATMUL](../glossary.md#matrix-multiplication--matmul) používají původní affine/offset/strided/broadcast geometrii. PERMUTE není indexováno podruhé; raw F64 FLOOR zachová přesné hodnoty. |
| Metal empty closure | NEG/unary, SUM, K0 či prázdný MATMUL a TOP_K k0 mají správné prázdné výsledky/[identity](../glossary.md#term-identity-meanings). Prázdný MATMUL→RELU přežije také [fused](../glossary.md#specialization-and-fusion)-anchor konstrukci bez nativního [dispatch](../glossary.md#term-dispatch). |
| CPU [variadic](../glossary.md#term-variadic) | Nad 16 výskytů se váže připravená operand table, nikoli N-wide [JVM](../glossary.md#term-java-virtual-machine-jvm) argumenty. Veřejné CONCAT případy se 102/257 vstupy již projdou; regrese pokrývají také STACK, 1025 vstupů, raw typy, opakované identity, [gradienty](../glossary.md#gradient-and-learning-rate) a controlled [Scan](../glossary.md#general-differentiable-scan). |
| Scan ingress a raw F64 | Numerický strided ingress se mapuje stávajícím [CONTIGUOUS](../glossary.md#term-contiguous) vlastníkem do deklarovaného workspace. [Canonical](../glossary.md#term-canonical-form) non-[gradient](../glossary.md#term-gradient) source-only F64 Scan zachová raw hodnoty; affine F64 Scan odmítá už capability, nikoli pozdější native [příprava](../glossary.md#prepare). |
| Cold [topology](../glossary.md#term-topology) | [Carry](../glossary.md#term-carry)-free a zero-ROW/scalar-SHARED zdroje používají existující v4/v5/v6. Veřejná sonda provedla 96 copy/load/[rebind](../glossary.md#capture-graph-ports-and-reconstruction)/[Engine](../glossary.md#term-engine) běhů pro F32/F64, T0/B0, oba směry a pojmenované aliasy s přesnými výsledky. |
| Owned [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) training | Odstraněn blanket fixed-recurrence [guard](../glossary.md#term-guard). [GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) používají stávající [TrainingSession](../glossary.md#training-session--trainingsession)/[Compiler](../glossary.md#term-compiler) [AD](../glossary.md#autograd), včetně mixed zdrojů a nového trénování po load; deset veřejných CPU/Metal [dtype](../glossary.md#data-type--datatype)/family průchodů ověřilo train/save/load/fresh-train a raw parametry. |

Následná nezávislá kontrola zachytila čtyři integrační mezery Metalu: Java affine [EXP](../glossary.md#term-expx)/RECIPROCAL,
raw EXP/[SIGMOID](../glossary.md#term-sigmoid) [preflight](../glossary.md#term-preflight), nulový fused MATMUL epilog a příliš širokou F64 Scan odpověď. Každá
byla skutečně reprodukována před opravou. Po opravě všech 18 dtype/[view](../glossary.md#term-view-meanings) variant unárních operací
a prázdný MATMUL→RELU provedly přesný veřejný [oracle](../glossary.md#term-test-oracle); nekanonický F64 Scan odmítl již [compile](../glossary.md#compile).
Opakovaná nezávislá zdrojová kontrola nenašla zbývající blocker.

### 15.2 Kompatibilita a hranice

- CPU používá selektivní schema 75 pro wide movement, schema 76 pro F32 RSQRT pracovní místa,
  dropout numeric identity v2 a softmax algorithm 3. Nedotčené malé movement ABI zůstává původní.<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [schema glossary](../glossary.md#term-schema); [F32 glossary](../glossary.md#floating-point-precision-and-rounding); [dropout glossary](../glossary.md#dropout--inverted-dropout); [identity glossary](../glossary.md#term-identity-meanings); [softmax glossary](../glossary.md#softmax--log-softmax); [ABI glossary](../glossary.md#term-application-binary-interface-abi))</span>
- Metal má semantic identity 44, ABI7 a schema22/header124 beze změny. Ověřený shader corpus má
  23 komponent, 190 688 UTF-8 bajtů a SHA-256<span data-glossary-navigation="true"> ([Metal glossary](../glossary.md#term-metal); [shader glossary](../glossary.md#term-shader); [UTF-8 glossary](../glossary.md#term-utf-8); [SHA-256 glossary](../glossary.md#term-sha-256))</span>
  `f6a560fd381664d01e0ec7f65f2148646d9d9837015133c5571ebadfb2c776fc`.
- Cold nulové počty nemění wire version ani inventory. Délky zůstávají externími batch daty,
  nejsou snapshotem. Batch geometry se odvozuje z jejich descriptoru, nikoli prvního carry/formal.<span data-glossary-navigation="true"> ([batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters); [carry glossary](../glossary.md#term-carry))</span>
- Fixed a general low recepty se neslučují; původní rounding boundaries a storage typy zůstávají.
  Low training objective vyžaduje explicitní F32/F64 cast.<span data-glossary-navigation="true"> ([rounding glossary](../glossary.md#term-rounding); [objective glossary](../glossary.md#term-objective); [F64 glossary](../glossary.md#floating-point-precision-and-rounding))</span>
- **Záměrné hranice trvají:** CPU BF16/F16 aritmetika je uzamčená; Metal nemá obecnou F64
  aritmetiku. Dynamic-B Metal, širší CPU symbolická F32/F64 parita, arbitrary cold PURE inventory,
  vyšší-order recurrent AD a širší resource/rank/pooling domény jsou samostatné návrhové fronty,
  nikoli schopnosti odvozené z tohoto cutoveru.<span data-glossary-navigation="true"> ([BF16 glossary](../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../glossary.md#floating-point-precision-and-rounding); [PURE glossary](../glossary.md#term-pure); [AD glossary](../glossary.md#autograd); [rank glossary](../glossary.md#term-rank); [pooling glossary](../glossary.md#term-pooling))</span>
- Původní veřejné changing-B sondy prokázaly 12 CPU běhů na stejných přípravách; dvě CPU a šest
  Metal kombinací zůstalo odmítnuto. Tato oprava nepřidává skryté rebinding ani jiné wire/API pro
  dynamický Metal. Příští návrh musí samostatně vyřešit symbolický kontrakt, per-run geometrii,
  zdrojovou/artefaktovou identitu a skutečný compile→prepare→run průchod.<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api); [run glossary](../glossary.md#term-run); [compile glossary](../glossary.md#compile); [prepare glossary](../glossary.md#prepare))</span>

### 15.3 Nové podklady

Pracovní kvalifikace je odděleně v `build/backend-interchangeability-audit/repairs/`:
veřejné boundary/numeric/owned-training výstupy, reviewed [Metal](../glossary.md#term-metal) reproducer, cold-[topology](../glossary.md#term-topology) sonda,
root gate a příslušné příkazy/zdroje. Prvotní chyby diagnostických/testovacích fixtures
(nekanonický flat-import [target](../glossary.md#term-target) a named [PARAMETER](../glossary.md#term-parameter) bez requiresGrad) byly opraveny ve spotřebiteli,
nikoli oslabením produkčních validátorů.

Finální neomezená root brána `test build` a [Javadoc](../glossary.md#term-javadoc) [CPU](../glossary.md#term-cpu), Metalu, [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Checkpointu a Compileru prošla
s aktuálním ověřeným nativním balíkem. Poslední příkaz vykázal 159 tasků: 4 executed a 155
up-to-date; jde o přijetí úspěšných uchovaných reportů, nikoli tvrzení, že se každý test spustil
znovu v posledním příkazu. Logy prvotního selhání i konečného úspěchu jsou uchovány samostatně.
Diagnostické zdroje jsou archivovány v sousedním `consumer-sources/`, nikoli v produkčním [runtime](../glossary.md#term-runtime).

Tyto změny ani nové source/test/native důkazy nepřeznačují starší zapečetěný
`build/release/0.1.0` za znovu vydaný balík. Neprokazují univerzální zaměnitelnost, nový výkonový
SLA, x86 [běh](../glossary.md#run--runtime) této nové [matice](../glossary.md#term-review-matrix) ani libovolnou cross-[backend](../glossary.md#backend) [optimizer](../glossary.md#term-optimizer) continuation.

Následující odkazy jsou připnuté na auditovaný [commit](../glossary.md#term-commit), aby se nemíchala dnešní zjištění s budoucí implementací.

[C1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java#L343-L497
[C2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java#L498-L907
[C3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java#L909-L1320
[C4]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java#L1338-L1664
[C5]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java#L92-L290
[C6]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java#L525-L727
[C7]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIr.java#L268-L284
[C8]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuSoftmaxInputValidator.java#L91-L174
[C9]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java#L234-L260
[C10]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomEmitter.java#L358-L390
[C11]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuOutputInjectivity.java#L14-L103
[C12]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLowering.java#L113-L157
[M1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java#L1340-L1604
[M2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java#L1606-L1972
[M3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java#L548-L1338
[M4]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java#L2107-L2370
[M5]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalReductionSupport.java#L27-L225
[M6]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNormalizationSupport.java#L169-L343
[M7]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalAttentionSupport.java#L162-L227
[M8]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalExactSupport.java#L194-L279
[M9]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphProgram.java#L364-L417
[M10]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java#L220-L333
[M11]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/macos-arm64/src/synaptik_exact_host.h#L47-L92
[M12]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/macos-arm64/src/synaptik_normalization_loss_kernels.h#L90-L126
[M13]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/macos-arm64/src/synaptik_float32_math_kernels.h#L109-L117
[M14]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/macos-arm64/src/synaptik_task0065_rng_dropout_kernels.h#L7-L23
[S1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuGeneralScanCompositionLowering.java#L414-L505
[S2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuGeneralScanLowering.java#L412-L471
[S3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalScanLowering.java#L95-L209
[S4]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalScanLowering.java#L293-L353
[S5]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalScanEncoding.java#L14-L63
[S6]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ScanGradientRules.java#L43-L112
[S7]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java#L240-L249
[S8]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentGradientRules.java#L25-L99
[S9]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalRecurrentSupport.java#L118-L187
[S10]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRecurrentScanLowering.java#L27-L95
[P1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/SourceGraphReconstruction.java#L89-L275
[P2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java#L419-L476
[P3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphSource.java#L192-L215
[N1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelBuilder.java#L714-L800
[N2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java#L208-L390
[N3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence.java#L245-L320
[E1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelection.java#L29-L40
[E2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuMetalPreparedBufferTransfer.java#L78-L132
[E3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java#L1505-L1567
[E4]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java#L224-L264
[E5]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/engine/src/main/java/io/github/pho001/synaptik/engine/HostTensorValue.java#L12-L18
[E6]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendRuntime.java#L145-L187
[E7]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/macos-arm64/build.sh#L4-L66
[E8]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java#L65-L159
[T1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java#L142-L150
[T2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutKind.java#L15-L26
[T3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/docs/architecture/contracts/foundational-modules.md#L223-L378
[T4]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/GraphRngState.java#L15-L29
[Q1]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/modules/backend/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalOperationCompletenessAuditTest.java#L108-L168
[Q2]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuSymbolicScanIntegrationTest.java#L15-L166
[Q3]: https://github.com/pho001/Synaptik/blob/65c0eb00745c77167f7df4e2bd3b4aad7a20b31c/testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuRecurrentScanIntegrationTest.java#L47-L96
