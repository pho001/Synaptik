# CPU ↔ Metal: podrobný audit úplné zaměnitelnosti

Reader navigation: [backend capability](../glossary.md#backend-capability),
[backend ownership](../glossary.md#backend-ownership), [layout](../glossary.md#layout),
[materialization](../glossary.md#materialization),
[numerical semantics](../glossary.md#profile-free-numerical-semantics), and
[general Scan](../glossary.md#general-differentiable-scan).

## 1. Závěr a platnost

**CPU a Metal nejsou plně zaměnitelné ani pro FLOAT32.** Shoda názvů operací a úspěšné katalogové reprezentanty neznamenají shodu všech tvarů, layoutů, hodnotových domén, gradientových uzávěrů a připravených programů.

Audit míří na repozitář `/Users/phujka/IdeaProjects/Synaptik`, commit `65c0eb00745c77167f7df4e2bd3b4aad7a20b31c`. Pracovní strom byl před auditem čistý. Produkční implementace, konfigurace backendů ani zapečetěná distribuce nebyly tímto auditem změněny. Zpráva je snímek schopností, nikoli nový numerický kontrakt.

**Navazující opravy:** oddíly 1–14 zachovávají původní audit commitu `65c0eb0`; nejsou přepsanou
kvalifikací nové implementace. Aktuální změny, nové běhy a zbývající hranice uvádí
[oddíl 15](#15-navazující-opravy-a-aktuální-hranice).

Hlavní zjištění:

- **Typy:** CPU má obecné F32/F64 výpočty a úmyslný zámek BF16/F16 aritmetiky. Metal má F32/BF16/F16 numerické vlastníky a oddělené přesné/raw F64 role, nikoli obecnou F64 aritmetiku.
- **F32 domény:** existují malé, skutečně spuštěné protipříklady v obou směrech. Metal odmítne některá prázdna, unary nad transponovaným view a další layouty; CPU odmítne prázdnou normalizační osu softmaxu a platný nulový broadcast batch attention.
- **Počet operandů:** odstranění stropu 16 neznamenalo neomezenou variadicitu CPU. Rank-1 CONCAT prošel se 101 vstupy, se 102 a 257 ne; Metal prošel i s 257. Jde o počet operandů, ne o počet řádků jediného tensoru.
- **Capability versus prepare:** u několika Metal výskytů je descriptor přijat při kompilaci, ale příprava odmítne fyzický stav/provenienci vstupu. Některá odmítnutí jsou úmyslně užší ingress; jiná jsou kandidáty na sjednocení Java/native validace.
- **Numerika:** byly skutečně zachyceny rozdíly CPU RSQRT a dropout vůči předepsaným typovaným mezikrokům. Nezaměňovat s obecným požadavkem bitové shody backendů.
- **Scan a veřejná API:** statický F64 identity Scan a strided F32 Scan prošly CPU, nikoli Metal. Engine-level fixed-cell AD existuje; owned `Model.configureTraining` přesto fixed recurrence odmítá na obou backendech.
- **Symbolika:** CPU umí kvalifikované měnící se B přes `runValues`; Metal je statický. Ani CPU nemá úplnou symbolickou paritu F32/F64.
- **Persistence a runtime:** uložený zdroj není přenositelný executable. Load zachovává typy a recept, ne cílovou způsobilost, připravené plány ani optimizer continuation.

## 2. Metodika a důkazní vrstvy

Zaměnitelnost je průnik několika nezávislých podmínek:

| Vrstva | Co skutečně dokazuje |
|---|---|
| Model | Platný druh operace, atributy, role, typová/promotion pravidla a odvozené výsledky. |
| Capability | Přípustnost konkrétního výskytu pro metadata dostupná provideru. Neověřuje automaticky obsah bufferů, alokovatelnost či všechny fyzické vazby. |
| Compiler/AD | Úplný dopředný a případný vybraný derivovaný graf našel přípustné vlastníky. |
| Prepare | Prošel konkrétní lowering, topologie, layout/provenience, prostředky a ABI. |
| Run/materialize | Prošla aktuální data, hodnotové kontroly, exekuce a publikace. |
| Numerický oracle | Pro konkrétní případ navíc sedí nezávislý očekávaný výsledek. Samotný hash výsledku to nedokazuje. |
| Cold/save/load | Zdroj lze rekonstruovat a znovu svázat; neznamená úspěšné spuštění na jiném backendu. |

**R** = nově skutečně spuštěno v tomto auditu. **S** = podloženo aktuální implementací, bez samostatného nového běhu dané kombinace. **H** = dřívější uchovaný kvalifikační důkaz. **[INFERENCE]** = odvozený, dosud neprovedený konkrétní scénář. Permanentní testy byly čteny jako zdroj, nikoli vydávány za nově spuštěné.

Nové běhy proběhly na macOS arm64 / Apple M3 Max / JDK 26.0.1. Metal byl registrován samostatně, bez CPU backendu. Použitá lokální knihovna má SHA-256 `f9f18b2378403b65629de1900b21235b3a44e4e06b709bac81263e9904bf6334`, ABI 7, schema 22, semantic identity 43. CPU běhy měly samostatného CPU vlastníka. Nebyl znovu spouštěn root test gate ani benchmark; jejich starší receipts nejsou novými výsledky tohoto auditu.

### Nově provedený katalog

Stejný existující auditovací harness byl znovu zkompilován proti aktuálním třídám a aktuálnímu Metal reprezentantovému fixture. Nebyl vytvořen druhý katalog operací.

| Výsledek katalogu | CPU | Metal |
|---|---:|---:|
| Záznamy | 5 018 | 5 018 |
| Ordinary druhy / legální signatury | 115 / 137 | 115 / 137 |
| Dokončené dopředné exekuce | 804 | 908 |
| Dokončené vybrané první AD exekuce | 874 | 1 156 |
| Odmítnutí při výběru/kompilaci | 760 | 374 |
| Chybějící legální signaturový fixture | 0 | 0 |
| Selhání po přijetí při prepare/run/materialize v katalogu | 0 | 0 |

General Scan je veden zvlášť. Další záznamy klasifikují neplatné projekce/fixture, výstupní role a nediferencovatelnost; nejsou automaticky backendovými mezerami. Syntetický katalog používá Model inference/constructible occurrences a skutečný Engine; není totožný se sadou výhradně veřejných uživatelských constructorů.

Při spárování všech klíčů se liší **1 134 výsledkových klasifikací**: F64 374, BF16 380, F16 380. F32/I32/I64/BOOL nemají v tomto reprezentativním katalogu rozdílnou klasifikaci. To **není** důkaz shody jejich úplných domén: následující veřejné boundary scénáře rozdíly prokazují.

Samostatně bylo provedeno **39 veřejných boundary scénářů na každý backend**, dále veřejné changing-B a numerické site sondy. Úspěšné konečné boundary případy prošly jednoduchým nezávislým raw-bit oraclem; softmax s nefinální hodnotou je záměrně evidován odděleně bez obecného numerického tvrzení. Setup chyby diagnostického spotřebitele byly opraveny pouze v něm, nikoli v backendu, a nejsou započítány jako mezery produktu.

## 3. Skutečně provedené protipříklady zaměnitelnosti

Všechny řádky této tabulky jsou **R**. „Compile“ znamená odmítnutí ještě před přípravou; „prepare“ přijetí kompilace, ale odmítnutí fyzické realizace; „run“ odmítnutí aktuálních hodnot. Přesné chyby, deskriptory a raw výsledky jsou v `build/backend-interchangeability-audit/public-boundaries-{cpu,metal}.jsonl`.

| Scénář | CPU | Metal | Význam |
|---|---|---|---|
| F64 ADD a smíšené F32+F64 ADD | Správný F64 výsledek | Compile odmítne | Obecná F64 aritmetika není společná. |
| BF16/F16 ADD, BF16/F16+F32 ADD, smíšený low+F32 MATMUL | Compile odmítne původní low aritmetický operand | Správný výsledek | Zámek CPU nezmizí promotion do F32. |
| BF16+F16 ADD | Compile odmítne | Správné F32 `[2,4]` | Metal přijímá Model-promoted smíšené low operandy. |
| Low EXPAND `[1]→[2]`, poté explicitní cast do F32 | Správně na obou typech | Správně na obou typech | Raw/movement není low aritmetika. |
| Gradient stejného low EXPAND, seed `[1,1]` | Compile odmítne vyžadovaný SUM | Správný low gradient `2` | Forward podpora není automaticky backward uzávěr. |
| F32 ADD, rank 17, pouze dva logické prvky | Správné `[3,5]` | Compile odmítne | Není to limit velikosti dat; je to rank. |
| MaxPool2d, vstup `[1,1,1,1]`, kernel 257×257, padding 128 | Správný jediný výsledek `7` | Compile odmítne | Limit 66 049 kernel pozic, nikoli velký výstup. |
| F32 `permute(1,0).reshape(6)` | Správné `[1,4,2,5,3,6]` | Stejný správný výsledek | CPU materializující noncontiguous RESHAPE už existuje. |
| F32 `permute(1,0).neg()` | Správné logické pořadí | Compile odmítne NEG | Následný unary consumer má na Metalu užší layout doménu. |
| F32 NEG nad `[0]` | Prázdný výsledek | Compile odmítne | Běžné Metal empty pointwise není obecně uzavřené. |
| F32 SUM nad `[0]` | Skalár +0 | Compile odmítne | Ordinary redukce a Scan-local empty podpora nejsou totéž. |
| F32 `[1,0]@[0,1]` | Skalární matice +0 | Compile odmítne | I nulová kontrakce má různé domény. |
| F32 K=1 MATMUL s `−0 × 1` | Přesné −0 | Přesné −0 | Dřívější singleton oprava je společná a prošla znovu. |
| F32 TOP_K, k=0, neprázdný vstup `[1,2]` | Prázdné values | Compile odmítne | Model-valid nulové k není na Metalu přijato. |
| F32 softmax `[2,0]`, osa 1 | Compile odmítne | Správné prázdno | Opačný směr asymetrie. |
| Attention Q `[0,2,3]`, K `[1,4,3]`, V `[1,4,5]` | Compile odmítne | Správné prázdno `[0,2,5]` | CPU predicate nesprávně vybírá broadcast batch pomocí maxima. |
| F32 softmax `[-MAX,+MAX]` | Správné `[0,1]` | Správné `[0,1]` | Finite input s overflow záporného shiftu CPU aktuálně přijímá. |
| F32 softmax `[-∞,0]` | Run odmítne: input must be finite | Provede `[+0,1]` | CPU má užší aktuální hodnotovou doménu než Model/Metal. |
| Rank-1 CONCAT 17/65/101 vstupů | Správně | Správně | Strop 16 je odstraněn. |
| Rank-1 CONCAT 102 a 257 vstupů | Compile odmítne | Správně | CPU má stále rank-dependent generační budget. |
| Externí F32 affine EQUAL, shape `[2]`, stride 2, offset 1 | Správné BOOL `[1,1]` | Prepare odmítne vstupní stav | Capability/kompilace samy nestačí. |
| Externí transpose F32 MATMUL, shape `[3,2]`, strides `[1,3]` | Správné `[5,7,9]` | Prepare odmítne | Samotný transpose descriptor nestačí bez lokální provenance. |
| Offset/strided F32 MATMUL, strides `[4,1]`, offset 1 | Správné `[6,15]` | Compile odmítne | CPU má širší read-layout doménu. |
| Read shape `[2,2,2]`, strides `[0,1,1]`, ADD skaláru | Správné opakované logické čtení | Prepare odmítne feed geometry | Zero stride nezaručuje nekolidující ostatní osy. |
| Externí F64 FLOOR broadcast view, shape `[2]`, stride 0 | Správné `[1,1]` | Prepare: native INVALID_ARGUMENT | Java/exact/native vstupní stav není jednotně přijímán. |
| Statický F64 identity GeneralScan bez aritmetiky | Přesné emise `[3,4]` | Compile odmítne celý Scan | Metal Scan vylučuje F64 i pro raw passthrough. |
| F32 GeneralScan se strided/offset sequence | Správný final carry `6` | Prepare odmítne MATERIALIZED_LAYOUT | Binder/physical ingress je další rozměr rozdílu. |

Úspěchy neznamenají libovolně velké nebo libovolně aliasované varianty stejného řádku. Sources: [C1], [C3], [C4], [C7], [C8], [M1]–[M11], [S1]–[S4].

## 4. Numerické rozdíly: co je dovolená variace a co vyžaduje opravu/review

### 4.1 Obecně nelze slíbit bitovou shodu

Stejný Model program neslibuje bitově identický obecný floating výsledek mezi CPU, Metalem, native routes a strategiemi. Finite accuracy se kvalifikuje po rodinách/typových cestách. Neexistuje jeden veřejný universal tolerance parametr. To ale neuděluje volnost měnit indexy, masky, tie-breaking, divisor, členství contributorů, uložené hodnoty, typované mezikroky nebo předepsané speciální hodnoty. [T3]

- Aritmetické NaN obvykle porovnáváme třídou, ne payloadem/signem; raw přesuny musí zachovat uložené bity a CAST má vlastní přesná konverzní pravidla.
- Signed zero je operation-specific; z přípustné volby znaménka výsledku některé agregace nevzniká volnost přidat +0 do K=1 MATMUL.
- DAZ/FTZ je dovoleno jen na jmenovaných F32/BF16/F16 aritmetických sites. Ne pro F64, přesné rozhodování, raw přesuny, CAST, masky či stav.
- Low numerická implementace na Metalu používá F32 pracovní hodnoty a zúží až příslušný deklarovaný low výstup. Nejde o F32 proxy tensor nahrazující původní graph dtype.

### 4.2 CPU F32 RSQRT: zachycený odlišný mezikrok

Model RSQRT popisuje square-root site a následné typed-one division. CPU F32 emitter místo F32 rootu použije double sqrt i double division a až potom zúží. Metal má float root a float division. [T1] [C9] [M13]

| RSQRT(7.0f) | Raw F32 bity |
|---|---|
| Typovaný reference `1.0f / (float)StrictMath.sqrt(7)` | `0x3ec18490` |
| Skutečný CPU | `0x3ec1848f` |
| Skutečný Metal | `0x3ec18490` |

**R:** rozdíl je 1 ULP; CPU odpovídá widened reference. Samotná odlišnost o 1 ULP ještě obecně není chybou backendové parity. Zde je navíc doložen odlišný předepsaný working-site postup; proto jde o konkrétní numerický conformance nález k opravě/review, ne o důvod zavést blanket bitwise-equality testy nebo epsilon do predicates. Ostatních šest prověřených RSQRT vstupů v této sondě tento rozdíl neukázalo.

### 4.3 CPU dropout: shodná maska/stav, odlišná kept aritmetika

Model předepisuje typed complement, typed reciprocal, potom násobení vstupu. CPU emitter používá přímé widened dělení vstupu complementem; Metal odpovídá float complement/reciprocal/multiply. [T2] [C10] [M14]

**R:** 256 F32 prvků na případ, stejné původní key/counter; pozorované masky i next-state byly mezi backendy shodné a dropped hodnoty měly přesné +0. Rozdíly se týkaly zachovaných hodnot, nikoli výběru RNG větve.

| p | Kept prvků | CPU: rozdílné proti typed-site reference | Metal: rozdílné proti reference |
|---|---:|---:|---:|
| 0.1 | 237 | 164 | 0 |
| 0.3 | 181 | 91 | 0 |
| 0.7 | 81 | 10 | 0 |

Např. p=0.1, lane 0: input `0xbf89d89e`, CPU `0xbf992993`, typed reference `0xbf992994`. Zdroj CPU F64 větve také používá přímé dělení; nový F64 numerický counterexample zde proveden nebyl. Nález se neopravoval a nemá být maskován změkčením numerického kontraktu.

RNG přitom obecně **nemá přenosný bitstream promise mezi backendy/verzemi**. Shoda masek v těchto třech případech není trvalý cross-backend RNG kontrakt. [T4]

## 5. Typy, promotion a gradientové uzávěry

### 5.1 Základní asymetrie

CPU low lock se vztahuje na **původní numerické operandy**, tedy i low+F32/F64 výpočet s vyšším výsledným typem, prázdný výstup, p=0 dropout či exponent 0/1. Explicitní low→F32 CAST→výpočet je jiný validní graf; není implicitním emulačním fallbackem. [C5]

CPU zachovává low storage, všech 49 CAST párů, classification, raw movement/unfold, homogenní low přesné unary `ABS/NEG/SIGN/FLOOR/CEIL/RELU`, clamp/extrema/porovnání/WHERE, MIN/MAX/arg redukce, ordering a vybrané winner/replacement role. Jejich VJP ale může přidat zakázané SUM, fold či jiné low výpočty. MaxPool2d může přijmout low forward s grad metadata, MaxPool3d low grad=true výslovně odmítá. [C1] [C2] [C3]

CPU smíšené F32/F64 a I32/I64 pointwise páry podporuje; u low extrema/comparison/WHERE vyžaduje homogenní low rodinu. Metal má širší floating promotion v těchto přesných rozhodovacích rolích, včetně BF16/F16→F32. Oba odmítají Model-nelegální floating/integer míchání; existence CAST není implicitní promotion těchto kategorií. [C1] [M1] [M8]

Metal F64 není „vypnut úplně“: existují raw movement/CAST/classification/ordering/comparison, vybrané přesné unary/extrema/WHERE, maximum pooling a neoverlapující folds. To neopravňuje F64 ADD, SUM, matmul, convolution, normalization ani obecné neural arithmetic. F64 Scan je užší ještě víc: jeho globální whitelist odmítá i čistý identity body. [M2] [M8] [S4]

### 5.2 Kompletní katalogová matice po operacích

Následující tabulka zahrnuje všech **115 ordinary druhů a samostatný GeneralScan**. Je převzata ze stejného existujícího formátu katalogu; všech 5 018 klasifikací na backend bylo právě znovu spárováno a ověřeno proti čerstvému běhu. Tabulka říká **které typové forward/AD reprezentanty se skutečně podařilo spustit**, ne všechny jejich shape/layout/value kombinace.

- **32/64** = F32/F64; **F** = obě; **L** = BF16+F16; **I** = I32+I64; **B** = BOOL; **N** = všech šest numerických typů; **A** = N+B.
- Typ značí skutečný hlavní hodnotový datový role; pevné masky/indexy/stav se neprojektují jako floating data. U ONE_HOT jde o indexový typ (výstup je BOOL); INITIAL_STATE je I64.
- **AD** = vybraný první gradient opravdu compile/prepare/run/materialize; **ND** = fixture nemá diferencovatelnou roli; **—** = žádná úspěšná Model-valid projekce.
- **\*** = jen část prověřených signatur/variant/výstupních či vstupních rolí. Ani buňka bez hvězdičky neslibuje všechny možné tvary, layouty nebo jiné role.
- F64 v přesném/source-only Metal gradientu neznamená obecnou F64 numerickou aritmetiku.

| Operace Modelu | CPU forward | CPU AD | Metal forward | Metal AD |
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

## 6. Shape/layout/value omezení podle rodin

Tato tabulka doplňuje dtype matici: zde jsou důvody, proč stejná operace ve stejném F32 ještě nemusí znamenat přenositelnost. **S** jsou aktuální zdrojové podmínky; konkrétní **R** výsledky jsou v sekci 3.

| Rodina | CPU | Metal | Zdroj / důkaz |
|---|---|---|---|
| Unary arithmetic a activations | Statické resolved nonnegative-stride/offset reads; podporuje i nekanonické input views. | N=F32/BF16/F16 unary obecně vyžaduje canonical input/output. Přesná F64 unary má vlastní affine read cestu. | [C1] [M1] [M8]; R transpose→NEG. |
| Tensor binary arithmetic | F32/F64 nebo I32/I64 promotion podle druhu, right-aligned broadcast; noninjective reads mohou být legální. | N promotion a přesné integral cesty; affine read musí splnit fyzický storage proof; canonical materializovaný output. Zero-stride shortcut v provideru není dostatečný důkaz ostatních os. | [C1] [M1] [M4]; R colliding read. |
| Scalar arithmetic / clamp | Scalar metadata přesně input dtype; floating clamp, bez integer clamp. | Stejný exact scalar convention, včetně F16 scalar metadata pro F16 scalar op. Nelze přenášet norm epsilon pravidlo na clamp. | [C1] [M1]. |
| Comparison / WHERE | CPU mixed-low není obecně přijat; floating a integer kategorie se nemíchají. | Přesné floating mixed comparisons a floating WHERE, ale external affine comparison se může zastavit až v prepare. | [C1] [M8] [M9] [M10]; R EQUAL ingress. |
| CAST / classification | Všech 49 cast párů; classification čte čtyři floating storage typy. | Totéž v přípustné fyzické doméně; žádné oprávnění k implicitní F64 arithmetic. | [C1] [M1] [M3]. |
| RESHAPE / rank edits / PERMUTE | Přesné view mapování, nebo materializující nekanonický RESHAPE; zachová původní logické pořadí. | View/materialization doména také existuje; subsequent consumer může vyžadovat canonical input. | [C4] [C12] [M4]; R transpose→RESHAPE. |
| EXPAND / CONTIGUOUS | Low/raw forward možný, ale redukující VJP může narazit na lock. | EXPAND má represented-empty výjimku; běžný CONTIGUOUS nemá automaticky všechny empty varianty. | [C4] [M3] [M4]; R low EXPAND a jeho VJP. |
| SELECT / signed SLICE | Checked coordinates; negative-step nebo empty materializace nevytváří negative physical strides. | Podobné odlišení logical signed slice od physical strides, ale empty slices mají vlastní stricter predicate (např. explicitní start=0). | [C4] [M3]. |
| SLICE_UPDATE / PAD / TILE | Širší statické prázdné a resolved-read domény podle rodiny; update funkcionální, nikoli in-place. | Canonical výsledky; řada retained read predicates vyžaduje kladné extenty, tedy ne obecné empty update/pad/tile. | [C4] [M3]. |
| CONCAT / STACK | Bez stropu 16, ale semantic occurrences a output rank vstupují do generačního budgetu. | Dynamické operand refs, přímé disjunktní kopie do finálního výstupu; žádný původní 16-input cap. STACK navíc rank vstupu <16. | [C7] [M3]; R 101/102/257. |
| UNFOLD_AXIS / UNFOLD2D / UNFOLD3D | Axis všech 7 storage typů; spatial unfold čtyři floating storage typy. Exact geometry/padding, canonical/injective materializace. | Podobná typová raw doména, ale rank a affine/positive geometry limity Metalu. UNFOLD_AXIS zvyšuje rank, vstup tedy nejvýše rank 15. | [C4] [M3]. |
| FOLD_AXIS / FOLD2D / FOLD3D | F32/F64 addition a integral axis fold; low addition zamčená i při neoverlapujícím okně. | N overlap podporován; F64 a integral FOLD_AXIS jen nonoverlap, F64 spatial fold jen nonoverlap každé osy. | [C3] [C5] [M3]. |
| GATHER / ELEMENTS / ND / ONE_HOT | Všech 7 data typů, I32/I64 indexy, exact bounds; ONE_HOT má BOOL výstup. | Totéž pod rank/positive-layout/uint podmínkami. GATHER_ND tuple depth a batch prefix přesně kontrolované. | [C3] [M3]. |
| SCATTER modes | NONE všechny storage typy, MIN/MAX šest numeric, ADD/MUL F32/F64/I32/I64; uniqueness pro replacement ověřena před writes. | N všechny modes; integral modular arithmetic; F64 NONE a vybrané MIN/MAX, ne ADD/MUL. Arithmetic scatter má zvláštní uint32 byte-span cap. | [C3] [M3]. |
| SORT / ARGSORT / TOP_K | Všech 7 storage typů, canonical decisions a I64 indexy; k=0 dovoleno. | Canonical inputs/outputs, uint32 extent/count; TOP_K vyžaduje k>=1. | [C3]; Metal provider 445–546; R k=0. |
| MATMUL | Vektory, matice, broadcast batches; resolved offsets/strides, K=0; nejde o backendový rank-12 native cap. | Rank<=16, positive extents, zero-offset dense nebo exact transpose-of-dense. Transpose external descriptor není autentizovaná lokální transpose provenance. | [C3] [M2] [M10]; R K0, offset, external transpose. |
| CONV2D / CONV3D | Promoted F32/F64 role, grouped/depthwise, strided/offset reads v rodinné doméně, injective output. | N promotion; image/weight a bias canonical nebo explicitní singleton-height view výjimka; canonical output a uint32 geometry. | [C2] [M2]. |
| MAX / AVERAGE_POOL2D/3D | Exact floor/literal-ceil/padding/dilation, širší read layouts; low max forward samostatně kvalifikován. | **Všechny čtyři** pooling kinds mají kernel volume<=65 536, nejen MAX. F64 max s grad navíc nonoverlap; N overlap podporován. | [C3] [M2]; R 66 049→1 output. |
| SUM / MEAN / PROD / masked / SUM-to-shape | Full/single/multiaxis, specifické empty identities; floating exact-state workspace může omezit přípravu. | N, vybrané exact integral cesty; ordinary reduction physical proof obecně vyžaduje neprázdnou geometrii a uint32 physical bytes. Scan-local empty proof je zvláštní. | [C2] [M5]; R empty SUM. |
| MIN/MAX/ALL/ANY/ARG_MIN/MAX | Typované identity a exact predicates; arg vyžaduje positive selected extent. | Storage-exact domény existují, ale canonical/uint a empty podmínky jsou užší; ne-N redukční grad role nejsou blanket povoleny. | [C2] [M5]; Metal provider 499–546,1973–2012. |
| LSE / variance / std / L1 / L2 | F32/F64; variance/std count>correction i při jinak empty output. | N; affine reads/canonical output a physical-byte proof; stejný count/correction guard. | [C2] [M5]. |
| CUM_SUM / CUM_PROD | F32/F64/I32/I64; oba směry a inclusive/exclusive; ordinary zero extent možný. | N/I32/I64, canonical positive-rank input/output, běžně positive geometry; žádná F64 cumulative arithmetic. | [C2] [M1]. |
| SOFTMAX / LOG_SOFTMAX | F32/F64, positive selected extent, **runtime finite input**. | N, zvláštní empty domény a širší Model special-value semantics. | [C2] [C8] [M6] [M12]; R obousměrné rozdíly. |
| LayerNorm / RMSNorm | F32/F64 promotion, exact trailing normalized suffix a affine shape; resource declarations samostatně. | N promotion, finite positive epsilon; nonempty row dispatch uint32. F16 výsledky mají Model-normalization F32 scalar metadata. | [C2] [M6]. |
| BatchNorm inference/training | Pět input rolí; training pět output rolí, C==0 nebo contributor count>=2; samostatné grad flags. | Stejné role/sibling equations v N doméně, canonical outputs a channel dispatch. Saved auxiliaries nejsou volné AD roots. | [C2] [M6]. |
| MSE | Exact stejné tvary prediction/target, ne target broadcast; empty SUM=+0, MEAN=NaN. | N promotion, oba operands canonical, nikoli libovolná affine MSE čtení; empty doména není plošně zakázaná. | [C2] [M6]. |
| Dense / index cross entropy | F32/F64, class-axis geometry; ignore porovnání před bounds; bez plošného finite-logits validatoru softmaxu. | N, vlastní zero-class/sample/ignore domain. Empty/fully-ignored mean není zakázán pouze kvůli nulovému included count. | [C2] [M6]. |
| Attention | F32/F64, Q/K/V a mask equations; aktuální zero+singleton batch broadcast bug. | N, zvlášť odvozené score/output geometrie a workspace; empty primary může obejít nepotřebné score scratch, ne validaci. | [C3] [M7]; R zero batch. |
| RNG / dropout | Explicit state, finite probability, F32/F64; CPU kept aritmetika je nález ze sekce 4. | N, explicit mask/next state, canonical geometry a uint32 bounds. | [C3] [T2] [M14]. |

Conv1d/Pool1d a `linear` nejsou další ordinary kindy: jsou to kompozice existujících operací. Jejich přenositelnost zahrnuje všechny vnitřní rank edits/views, kontrakce/pooling a vybraný backward; nestačí přítomnost Conv2d nebo MATMUL v katalogu.

## 7. Symbolické rozměry, GeneralScan, recurrence a AD

### 7.1 Ordinary changing-B: nová veřejná exekuce

**R:** skutečný veřejný `Engine.compile`→jedno `prepare`→opakované `runValues` při B=2,0,3:

| Symbolický graf `[B,2]` | CPU | Metal |
|---|---|---|
| F32 ADD | Všechny tři B, správné hodnoty i runtime shape | Compile odmítne symboliku |
| F32 MATMUL s pevnou `[2,2]` vahou | Všechny tři B, stejná příprava | Compile odmítne |
| F32 TANH | Všechny tři B, nezávislý numerický oracle | Compile odmítne |
| F64 ADD | Všechny tři B, správné hodnoty | Compile odmítne |
| F64 MATMUL a TANH | Compile odmítne, přestože statické varianty CPU umí | Compile odmítne |

B se váže z nových plně statických caller-input descriptorů, ne přes nový argument `prepare`. Používá se `runValues`, ne exact-ID `run` se symbolic prototype descriptor. Public batch plan vyžaduje CPU-only ownership, jeden společný symbol odvozený z caller vstupu, zachované typy/ranky/grad flags, přesné fixed input descriptors a zero-offset canonical strides symbolic inputů. Shape/stride/count/span musí projít int/byte admission i za prázdnou osou. [C6] [E3] [E4]

CPU ordinary symbolic matice je záměrně omezená, převážně rank<=3 se zvláštními pravidly. F32 má MATMUL, vybrané binary/scalar arithmetic, NEG/TANH/SIGMOID, SUM/MEAN, nutné layouty a cast. F64 typed větev je menší (ADD/MUL/DIV, potřebné redukce/layouty/cast); statická podpora F64 nepřechází automaticky na symbolickou. F64 NEG/RELU/mixed MATMUL uvnitř širší CPU Scan composition nejsou důkazem stejné standalone ordinary admission. [C6] [S1] [Q2]

### 7.2 GeneralScan — odlišný body i binder průnik

| Rozměr | CPU | Metal | Důkaz |
|---|---|---|---|
| Shape | Static a kvalifikované one-named-B closure; ne libovolné expressions/více symbolů | Všechny local/boundary shapes static | S [S1] [S4], public symbolic zdroje [Q2]. |
| Typy | Per-value types bez univerzální homogeneous floating podmínky; jednotliví ordinary ownery a low lock stále platí | Whitelist F32/BF16/F16/BOOL/I32/I64; **žádný F64 local/boundary value** | R F64 raw identity; S [S1] [S3] [S4]. |
| Operace body | Nejen starý ADD/SUB/MUL/DIV/NEG/TANH/SIGMOID/MATMUL/SUM slice; při neúspěchu legacy analysis ordinary composition fallback | Každý actual body node projde ordinary semantics, typed ownerem a native ABI | S [S2] [S3]. |
| Carry-free / zero-ROW length | Generic composition je umí v kvalifikovaných rolích | Source/binder je obecně nevylučuje | S; neznamená cold/save podporu. |
| Direct read layout | Generic statická composition uchovává actual offset/strides a mapovaný ingress | Final Scan input fyzicky canonical represented storage; materialized-layout feed může prepare odmítnout | R strided F32 Scan; S [S1] [S4]. |
| Směr a lengths | Oba směry; runtime INT64 lengths v `[0,T]` před Scan-local writes | Totéž; native time loop a once-prepared children | S; rozsah starších numerických průchodů H. |
| Prázdno/T0/inaktivita | Structural body kontrola neodpadá; T0 zachová carry, inactive rows nečtou neplatné operandy | Stejný požadavek, ne blanket empty ordinary unlock | S; low lock nezmizí T0. |
| Velikost body | Legacy 64-formal/128-node/256-value limity nejsou automaticky limity composition fallbacku; generované unity mají vlastní budgety | 4 096 values, 512 units, 1 048 576 encoded words plus physical/workspace bounds | S [S2] [S5]. |

Důležitá oprava starších popisů: CPU Scan není nyní univerzálně homogeneous F32/F64, dense-zero-offset ani carry-required. Některé širší named-F64/mixed/empty body numerical testy jsou ale backend-private. Jejich přesnou veřejnou Engine dosažitelnost nelze vydávat za již provedený veřejný průchod; external public symbolic length-Scan F32/F64 má vlastní integrační zdroj [Q2].

### 7.3 Co omezuje společný Compiler, nikoli jeden backend

- General/fixed recurrence má vybraný první AD; vyšší řád je společně odmítán. Vyšší-order request s původním GeneralScan může být odmítnut i kvůli jinak vedlejší původní větvi. [S7]
- U původního length-aware GeneralScan AD musí všechny původní carries a sequences splňovat ROW(0), včetně nepoužitých/nongradient declarations. Model-valid forward s ROW(1), SHARED sequence nebo terminal SHARED carry proto není automaticky diferencovatelný. Generated reverse Scan má jiné povolené history/seed/terminal role. [S6]
- Délky a discrete mask/index/state nemají cotangent. Saved residual se nesmí stát samostatným cotangent rootem; primal-only použití např. ve WHERE condition je jiná věc.
- BatchNorm saved mean/inverse-std, attention weights a ostatní canonical siblings mají vlastní role a derivability, ne libovolný Cartesian součin output×input.
- RNG/dropout/effectful operace, hidden leaves, host-backed body interiors a vnořené recurrence nepatří automaticky do PURE Scan jen proto, že běží samostatně.

### 7.4 Fixed RNN/GRU/LSTM: Engine AD versus owned Model

**Engine-level první AD není na CPU chybějící funkce.** Čerstvý katalog ho skutečně provedl. Current public test source dále provádí selected weight gradients všech tří cells, obou směrů, bias/no-bias a layout variants v CPU F32/F64; to není totéž jako nové all-role finite-difference ověření v tomto auditu. [Q3]

CPU forward je F32/F64 s kvalifikovanými resolved reads; Metal F32/BF16/F16 s canonical physical descriptors všech rolí včetně lengths. Společný Compiler staví cell transition a ordinary Scan VJP, nikoli oddělený handwritten backend adjoint. [S8] [S9] [S10]

**S:** `Model.configureTraining/train` má přesto vlastní guard proti jakékoliv fixed recurrence, i ve smíšeném fixed/general modelu. `ModelBuilder.rnnScan` zvolí GeneralScan pro F32/F64, ale pro BF16/F16 zachová fixed; `gruScan/lstmScan` zachovají fixed. Metal fixed-cell AD tedy **neznamená**, že všechny low owned recurrent buildery lze trénovat přes tento owned Model surface. Objective navíc musí být scalar grad-eligible F32/F64. [N1] [N2]

Explicitní `RnnSequence.forwardGeneralScan` může vytvořit low ordinary-cell recipe. Ten má jiné rounding boundaries než fixed low F32-working recipe: není to kompatibilní náhradní implementace stejného zdroje. Tvarově stejné váhy/délky proto nemusejí dát shodné low hodnoty. [N3]

## 8. Persistence, přechod na jiný backend a pokračování tréninku

**S:** Graph checkpoint ukládá zdrojový recept, descriptors, operace/atributy, scope/ports/aliases a raw named state. Zachová F64/F32/BF16/F16/I32/I64/BOOL typy, nikoli automatickou konverzi do typů nového backendu. Rebind vyžaduje full descriptor equality. [P2] [P3]

- Formáty v1/v2 nesou starší flat domény, v3 CNN/fixed recurrence, v4 all-valid GeneralScan, v5 lengths, v6 Pool3d (včetně contained). Backend switch není změna wire verze ani změna fixed/general identity.
- Cold reconstruction je užší než compute: static shapes, uzavřený constructor/opcode inventory, nonempty carry a pro length Scan pozitivní ROW-carry count. Carry-free či zero-ROW compute podpora tedy nezaručuje save/load.
- Length cold role zůstávají floating, ale současné executable guardy již nejsou univerzálně homogeneous-F32/F64-only; per-role mixed a low-storage/F32-working recipes existují. Povolené row/layout transformace a rank-two contraction jsou dál omezené. [P1]
- `Model.loadGraph(..., targetEngine)` nekontroluje při samotném load dostupnost/prepare každé operace na zvoleném backendu. První compute znovu compile/prepare. F64 soubor může jít načíst a potom selhat na Metal; low numerický soubor analogicky na CPU.
- Externí batch a lengths je třeba znovu dodat, pokud nebyly výslovně named state. Není obnoveno objective, optimizer slots/step count, AD history, připravený plán, GPU kontext nebo compiled artifact.
- Fresh configured training po load není optimizer continuation a bitově identická trajektorie tréninku na jiném backendu není slib.

Zpráva neprovedla nový cross-backend save/load matrix; tyto odlišné hranice jsou source-backed. Dřívější izolované CPU/Metal save/load/nové-trénování receipts jsou H, nikoli nový důkaz libovolné migrace.

## 9. Resource, transport a runtime rozdíly

### 9.1 Neexistuje jeden univerzální maximum tensor size

| Limit / vztah | CPU | Metal / společná mez |
|---|---|---|
| Rank | Ordinary static provider nemá univerzální rank-16 strop; rodinné/generační limity stále existují | Pervasive rank<=16; některé rank-zvyšující operace potřebují input<=15. R rank17 ADD. |
| Velikost/span | Checked long geometry, heap arrays JVM-int limited, segment carriers mohou mít long addressing; není to příslib alokovatelnosti | Family-specific uint32 geometry/count/span/byte limity. **Není** jeden plošný „4GB backend limit“. |
| Pooling | Checked geometry/product bez uvedeného Metal kernel capu | Všechny čtyři pooling kinds kernel product<=65 536. |
| Scatter arithmetic | Exact resources a span/alias checks | Vybrané logical/physical byte spans<=4 294 967 295, včetně offsetů/holes. |
| Variadic CPU movement | `occurrences+7<=254`; nad 16 navíc `(occurrences+1)*(outputRank+4)<=512` | Metal nezdědil tento Java entry budget; má vlastní encoded/ref/resource limity. [C7] |
| Injectivity proof | Span/residue proof + exact fallback nejvýše 1 000 000 prvků; větší neprokázaný layout může být odmítnut i pokud matematicky injective | Nonbroadcast osy storage proof nesmějí kolidovat; materializing outputs obvykle přesně canonical. [C11] [M4] |
| Workspace | Exact reductions/PROD limbs/scatter/order/affine address tables mohou růst s logickým rozsahem, ne jen výstupem | Attention F32 score work, Scan locals/scratch/control a další declared workspaces; low publikované weights nenahrazují F32 working probabilities. |
| Host materialize | Oba backendy vracejí canonical row-major **big-endian** bytes; jeden payload musí vejít do `byte[]` | Tedy nejvýše `Integer.MAX_VALUE` payload bytes. `maximumTotalBytes` není limit peak procesu, staging ani všech runtime buffers. [E5] |

Příklady unproven obří injectivity či multi-GiB allocations nebyly alokovány. `[INFERENCE]` z postačujícího proofu není skutečně vyzkoušená hranice zařízení. Native `newBufferWithLength` může selhat i pro descriptor, který projde integer geometrickou validací.

### 9.2 Host borrowing a mixed-owner transport

CPU intrinsic borrow drží skutečný non-owning host segment s kontrolou původního carrieru, velikosti, živosti a accessibility. Metal borrow znamená upload kopie do owned Metal bufferu; zachování reference na host storage není koherentní sdílená residency. Mutace původního host bufferu sama neaktualizuje již vytvořenou Metal reprezentaci. Nové Engine bindingy/čerstvé inputy jsou samostatná věc. [E6]

Cross-owner CPU↔Metal transfer je užší než oba lokální backends: static rank<=16, positive extents/strides a nonoverlap proof, checked referenced span, CPU native staging místo heap carrieru. Přenáší celý referenced span včetně prefix/holes bez konverze. Empty, unresolved nebo zero-stride broadcast boundary nepodporuje. Graph tedy může mít každý node lokálně přijatelný a přesto selhat na rozhraní mezi vlastníky. [E2]

Run resources a publikace mají vlastní lifetime/leases. Caller storage musí být live/accesssible bez konfliktní mutace až do synchronous completion. Shareable prepared plan není slib nezávislého ordering/fairness/throughputu nebo souběžného GPU overlapu; Metal kontext má jednu command queue. CPU default má concurrency budget 1; obecná existence vector/parallel strategií není jejich implicitní zapnutí v každém programu.

## 10. Výběr vlastníka, native cesty, platformy a cache

**S:** `Engine.standard()` registruje pouze CPU. `Engine.builder()` začíná bez backendu. Při více registracích bez jiné class preference vyhraje první hard-eligible provider pro výskyt; s preferencí se hledá první eligible provider dané třídy. Metal-first+CPU tedy může naplánovat smíšený graf bez zvláštního fallback přepínače. To je **plánovaná composition**, nikoli runtime retry. Metal-only nemá odkud CPU výpočet získat a run/prepare selhání nezahájí fallback. [E1] [E8]

| Oblast | CPU | Metal |
|---|---|---|
| Základní deploy | JDK 26; scalar nepotřebuje incubator Vector module; native probing vyžaduje správné FFM permission | JDK 26 + dodaná arm64 macOS native knihovna; checked-in build deployment minimum macOS 26.0, Xcode a systémové frameworky. [E7] |
| Native získání | OpenBLAS optional bounded discovery/qualification; neúspěch nechá portable CPU. Explicitní oneDNN load je required a chybu neskrývá | Explicitní absolute path; aktuální ABI/schema/semantic identity; system-default Metal device, nikoli libovolně přenositelný připravený kontext |
| Co je další backend | OpenBLAS/oneDNN **nejsou** další backendy: jsou CPU-internal routes | Custom MSL/MPSGraph jsou také routes téhož Metal vlastníka |
| OpenBLAS | Homogeneous F32/F64 nonbatched 2D MATMUL K>1, int ABI a vlastní kvalifikace; nikoli omezení všech CPU matmulů | Nemá roli Metal fallbacku |
| oneDNN | Explicitní 3.14.0, LP64 little-endian, SEQ; qualified homogeneous F32 bare MATMUL K>1, normalized dense/permuted batches do native rank 12 | Nezvyšuje Metal capability ani jeho rank/layout doménu |
| API preference | AUTO/PRIMITIVES/GRAPH omezuje kandidáty, nevnucuje native execution; whole-plan dimensionless heuristika, ne naměřená latence | Existence MPSGraph sama nedokazuje libovolný occurrence support |
| Caching | Generated ClassFile weak interning, případně oddělená trusted-root cache class bytes; ne JIT strojový kód ani přenositelné prepared native handle | Bounded session-local tuning/cache, ABI/schema/source identity; ne portable executable uložený checkpointem |
| Native distribuce | oneDNN není bundlováno; OpenBLAS library/ABI musí projít kvalifikací | Lokálně source-built/ad-hoc signed package, nikoli publisher-authenticated/notarized veřejný prebuilt |

Výkonová zaměnitelnost, stejná peak memory, stejný thread count ani stejné startup/prepare náklady z toho neplynou. V tomto auditu nebyla znovu měřena rychlost. Dřívější report-only benchmark zůstal oddělený od capability rozhodnutí.

## 11. Priority nálezů a návrh dalších kroků

### P1 — správnost / jasná hranice přijetí

1. **CPU RSQRT a dropout:** zachovat předepsaná pracovní místa ve společných emitters, nikoli přidávat value special-cases nebo povolit širší epsilon. Přidat cílený failing-before/passing-after oracle při vlastní opravě; tento audit permanentní testy ani runtime nemění.
2. **CPU attention zero+singleton batch:** opravit broadcast predicate (`Math.max` nesmí přepsat ne-one nulu na 1); nález má zdroj i veřejný R counterexample.
3. **Metal descriptor/prepare/native rozhraní:** sjednotit podporu/provenance nebo explicitně přesně vymezit external affine comparisons, external transpose MATMUL a external broadcast exact owner. Oddělit úmyslný dense/provenance ingress od nesouladu validátorů; neskrývat native INVALID_ARGUMENT fallbackem.

### P2 — největší praktické mezery portability

4. Doplnit průnik běžných empty domén (NEG/SUM/K0 MATMUL/TOP_K k0/softmax) po rodinách, ne globálním obcházením preflightu. Scan-local empty podpora se nesmí prohlásit za ordinary podporu bez správné closure.
5. Rozšířit Metal unary/kontrakční physical read domény nebo sdílenou plánovanou materializaci. Dnešní canonical vstupy samy nestačí, protože i legitimní mezikrok PERMUTE vytvoří view.
6. Vyřešit CPU variadic movement budget bez nového N-wide JVM entry limitu. Neuvádět „bez limitu“ místo současného závislého budgetu; nejde o tensor-row limit.
7. Ujasnit owned Model fixed-recurrence training surface vůči už existujícímu Engine AD; explicitně oddělit fixed a general low rounding recipes.
8. Symbolický Metal/general-Scan rozsah a širší cold/persistence doménu plánovat samostatně; nevyvozovat je z ordinary F32 katalogu.

### Záměrné rozdíly, které audit nenavrhuje potají odemknout

- CPU BF16/F16 aritmetický zámek je výslovná uživatelská produktová volba, ne opomenutí k tiché opravě.
- Metal obecná F64 aritmetika není nynější cíl; exact/raw F64 podpora nemá být nahrazena nepřiznanou F32 aproximací.
- Rank/body/resource bounds a platform packaging je třeba zveřejnit a případně promyšleně rozšířit, nikoli odstranit checked overflow či alias validaci.
- Žádný nový runtime fallback, implicitní low emulace, druhý AD engine ani alternativní numerical policy.

## 12. Dokumentační rozpory a zbylé neověřené domény

Potvrzené stale popisy nejsou totéž co chybějící implementace:

- `production-support.md` ještě uvádí CPU0079 noncontiguous RESHAPE jako omezení; R transpose→materializing reshape prošel na CPU i Metalu.
- Starší Scan a CPU provider komentáře popisují užší homogeneous/dense/carry-required/symbolic slice než aktuální generic composition. Je nutné sledovat fallback, nikoli jen první legacy predicate. [S1] [S2]
- Popisy cold homogeneous F32/F64-only neodpovídají nynější per-role low/F32-working floating doméně. To ale neodstraňuje static/closed-constructor/ROW/carry limity. [P1]
- „Fixed recurrent je forward-only“ neplatí pro Engine/Compiler první AD; pro owned Model train guard je však omezení skutečné. [S8] [N2]
- Některé Metal komentáře stále říkají F32-only overlapping folds/pool cotangents; implementace používá shared N vlastníka. Jiné starší komentáře uvádějí schema21/codec40 místo aktuálního schema22/codec43.
- `Engine.standard()` prose „without discovery“ nezachycuje současné optional OpenBLAS discovery/qualification; stálý backend owner je přesto CPU-only.

**Neprovedeno nově:** jiné GPU/OS/JDK, x86 audit celé této nové boundary matice, obří allocations, úplný cross-backend checkpoint migration matrix, každá mixed-type/shape/layout/AD kombinace, vyšší-order closure všech ordinary operací, všechny tolerance/transcendentální value sweeps a souběžné mixed CPU↔Metal transfery. Source-predicted external affine SCATTER_ADD state gap zůstává **[INFERENCE]**; není vydáván za vykonaný failure.

Předchozí CPU/Metal rozsáhlé numerické/trénovací receipts zůstávají H pro své přesné fixtures. Úspěšný katalog zde pouze potvrzuje, že přijaté reprezentanty provedly celý pipeline; **nelze z něj odvodit univerzální numerickou správnost ani úplnou zaměnitelnost**. [Q1]

## 13. Co lze dnes rozumně považovat za přenositelný základ

Konzervativní společný základ je **konkrétně kvalifikovaný** statický F32 program, rank<=16, s podporovanými materialized/intermediate layouts, rodinnou empty doménou, platnými indexy/maskami, finite standalone softmax inputy, pool/window/resource budgety a podporovaným vybraným backward uzávěrem. Je nutné prověřit také vnitřní views a AD-created operations, ne jen vstupní tensors.

Ani tento popis není nová blanket certifikace všech takových programů: RSQRT/dropout nálezy a další samostatné přípustnosti musí být vyřešeny či explicitně kvalifikovány. Doporučená portability kontrola vždy zkusí **oba isolated backends** přes compile→prepare→run→materialize a přesně deklaruje numerical oracle; samotné `supports()` ani dostupnost druhého backendu není náhradou.

## 14. Podklady a reprodukce

Lokální podklady tohoto auditu: `build/backend-interchangeability-audit/`:

- `catalog-{cpu,metal}.{jsonl,tsv,summary.json}` — nových 5 018 řádků na backend.
- `catalog-differences.json`, `catalog-summary.json`, `audit-summary.json` — spárované klasifikace a měřené souhrny.
- `public-boundaries-{cpu,metal}.jsonl` — konečných 39 veřejných scénářů na backend s fází odmítnutí a raw oraclem.
- `numeric-sites-{cpu,metal}.jsonl` — RSQRT a dropout typed-site údaje, skutečné masky/next-state.
- `symbolic-{cpu,metal}.jsonl` — veřejné changing-B průchody a odmítnutí.
- `commands.json`, `audit-environment.json`, `consumer-sources/` — přesné příkazy, prostředí a zdroje diagnostických spotřebitelů.
- `source-reviews/` — čtyři oddělené read-only zdrojové analýzy a upřesnění public symbolic cesty; nejsou samostatným runtime důkazem.

Reprodukce používá existující current main/fixture classes ze zdrojové kvalifikace commitu, JDK 26, samostatně registrovaného CPU nebo Metalu a výslovnou native path. Diagnostické sources nejsou přidány do runtime implementace ani permanentní testovací sady. Počáteční diagnostický byte-order/F16-fixture setup byl opraven podle existujících public API: host publication je big-endian a raw F16 nese vlastní storage type. Prvotní setup chyby nejsou nálezy backendových numerických chyb.

## 15. Navazující opravy a aktuální hranice

Tato část nahrazuje doporučení oprav z oddílu 11 pro současný pracovní zdroj; původní měření
a 57 připnutých odkazů v oddílech 1–14 zůstávají historickým auditním snímkem `65c0eb0`.
Model numerika se nemění a nevzniká druhý executor, AD engine ani runtime fallback.

### 15.1 Uzavřené nálezy

| Oblast | Oprava a ověřený výsledek |
|---|---|
| CPU RSQRT | F32 square root se zaokrouhlí před F32 reciprocal. Sdílené emitters zachovávají toto pořadí ve scalar/vector/remainder i složených tělech; veřejná typovaná sonda již nevykazuje původní rozdíl. |
| CPU dropout | Typovaný complement → reciprocal → multiply nahrazuje přímé dělení vstupu. F32 i F64 sondy odpovídají nezávislému typovanému oraclu; masky a RNG next-state zůstaly stejné. |
| CPU attention | Broadcast nuly a singletonu zachová nulu místo `Math.max(0,1)=1`; platný nulový batch se provede jako prázdný výstup. |
| CPU softmax | Prázdná vybraná osa a Model NaN/±infinity doména již nejsou odmítnuty starým finite-only validátorem. Veřejné sondy ověřily SOFTMAX i LOG_SOFTMAX včetně velkých `[128,2048]` výstupů. |
| Metal fyzické reads | Unary, comparisons a numerický MATMUL používají původní affine/offset/strided/broadcast geometrii. PERMUTE není indexováno podruhé; raw F64 FLOOR zachová přesné hodnoty. |
| Metal empty closure | NEG/unary, SUM, K0 či prázdný MATMUL a TOP_K k0 mají správné prázdné výsledky/identity. Prázdný MATMUL→RELU přežije také fused-anchor konstrukci bez nativního dispatch. |
| CPU variadic | Nad 16 výskytů se váže připravená operand table, nikoli N-wide JVM argumenty. Veřejné CONCAT případy se 102/257 vstupy již projdou; regrese pokrývají také STACK, 1025 vstupů, raw typy, opakované identity, gradienty a controlled Scan. |
| Scan ingress a raw F64 | Numerický strided ingress se mapuje stávajícím CONTIGUOUS vlastníkem do deklarovaného workspace. Canonical non-gradient source-only F64 Scan zachová raw hodnoty; affine F64 Scan odmítá už capability, nikoli pozdější native příprava. |
| Cold topology | Carry-free a zero-ROW/scalar-SHARED zdroje používají existující v4/v5/v6. Veřejná sonda provedla 96 copy/load/rebind/Engine běhů pro F32/F64, T0/B0, oba směry a pojmenované aliasy s přesnými výsledky. |
| Owned NN training | Odstraněn blanket fixed-recurrence guard. GRU/LSTM používají stávající TrainingSession/Compiler AD, včetně mixed zdrojů a nového trénování po load; deset veřejných CPU/Metal dtype/family průchodů ověřilo train/save/load/fresh-train a raw parametry. |

Následná nezávislá kontrola zachytila čtyři integrační mezery Metalu: Java affine EXP/RECIPROCAL,
raw EXP/SIGMOID preflight, nulový fused MATMUL epilog a příliš širokou F64 Scan odpověď. Každá
byla skutečně reprodukována před opravou. Po opravě všech 18 dtype/view variant unárních operací
a prázdný MATMUL→RELU provedly přesný veřejný oracle; nekanonický F64 Scan odmítl již compile.
Opakovaná nezávislá zdrojová kontrola nenašla zbývající blocker.

### 15.2 Kompatibilita a hranice

- CPU používá selektivní schema 75 pro wide movement, schema 76 pro F32 RSQRT pracovní místa,
  dropout numeric identity v2 a softmax algorithm 3. Nedotčené malé movement ABI zůstává původní.
- Metal má semantic identity 44, ABI7 a schema22/header124 beze změny. Ověřený shader corpus má
  23 komponent, 190 688 UTF-8 bajtů a SHA-256
  `f6a560fd381664d01e0ec7f65f2148646d9d9837015133c5571ebadfb2c776fc`.
- Cold nulové počty nemění wire version ani inventory. Délky zůstávají externími batch daty,
  nejsou snapshotem. Batch geometry se odvozuje z jejich descriptoru, nikoli prvního carry/formal.
- Fixed a general low recepty se neslučují; původní rounding boundaries a storage typy zůstávají.
  Low training objective vyžaduje explicitní F32/F64 cast.
- **Záměrné hranice trvají:** CPU BF16/F16 aritmetika je uzamčená; Metal nemá obecnou F64
  aritmetiku. Dynamic-B Metal, širší CPU symbolická F32/F64 parita, arbitrary cold PURE inventory,
  vyšší-order recurrent AD a širší resource/rank/pooling domény jsou samostatné návrhové fronty,
  nikoli schopnosti odvozené z tohoto cutoveru.
- Původní veřejné changing-B sondy prokázaly 12 CPU běhů na stejných přípravách; dvě CPU a šest
  Metal kombinací zůstalo odmítnuto. Tato oprava nepřidává skryté rebinding ani jiné wire/API pro
  dynamický Metal. Příští návrh musí samostatně vyřešit symbolický kontrakt, per-run geometrii,
  zdrojovou/artefaktovou identitu a skutečný compile→prepare→run průchod.

### 15.3 Nové podklady

Pracovní kvalifikace je odděleně v `build/backend-interchangeability-audit/repairs/`:
veřejné boundary/numeric/owned-training výstupy, reviewed Metal reproducer, cold-topology sonda,
root gate a příslušné příkazy/zdroje. Prvotní chyby diagnostických/testovacích fixtures
(nekanonický flat-import target a named PARAMETER bez requiresGrad) byly opraveny ve spotřebiteli,
nikoli oslabením produkčních validátorů.

Finální neomezená root brána `test build` a Javadoc CPU, Metalu, NN, Checkpointu a Compileru prošla
s aktuálním ověřeným nativním balíkem. Poslední příkaz vykázal 159 tasků: 4 executed a 155
up-to-date; jde o přijetí úspěšných uchovaných reportů, nikoli tvrzení, že se každý test spustil
znovu v posledním příkazu. Logy prvotního selhání i konečného úspěchu jsou uchovány samostatně.
Diagnostické zdroje jsou archivovány v sousedním `consumer-sources/`, nikoli v produkčním runtime.

Tyto změny ani nové source/test/native důkazy nepřeznačují starší zapečetěný
`build/release/0.1.0` za znovu vydaný balík. Neprokazují univerzální zaměnitelnost, nový výkonový
SLA, x86 běh této nové matice ani libovolnou cross-backend optimizer continuation.

Následující odkazy jsou připnuté na auditovaný commit, aby se nemíchala dnešní zjištění s budoucí implementací.

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
