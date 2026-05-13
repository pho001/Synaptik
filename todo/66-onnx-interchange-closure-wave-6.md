# ONNX Interchange Closure Wave 6

## Stav Rozpracovani

Status: `IMPLEMENTED`

Wave 2-5 posunuly ONNX vrstvu z uzkeho import/export prototypu na prakticky staticky dense inference interchange:

- import umi vetsinu dnes evidovaneho opset-18 dense subsetu;
- export umi zakladni tensor, NN, layout, reduction, indexing/scatter a activation rows;
- `OnnxCoverageMatrix` oddeluje import/export/CPU/Metal/CUDA stav;
- compat fixture harness umi `IMPORTED`, `EXECUTED` a `REJECTED_WITH_REASON`;
- shape-param hardening uz brani tomu, aby runtime `INT32` tensor omylem fungoval jako compile-time shape/axes/repeats konstanta.

Aktualni coverage stav po wave 5:

```text
Import: supported=78, partial=0, unsupported=1
Export: supported=65, partial=1, unsupported=13
CPU: supported=78, partial=0, unsupported=1
Metal: supported=50, partial=9, unsupported=20
CUDA: supported=34, partial=9, unsupported=36
Round-trip evidence: round_trip_tested=17, explicitly_classified=49, import_only_tested=12, rejection_tested=1
```

Wave 6 ma zavrit dalsi nejdulezitejsi mezery v ONNX interchange kontraktu, ale bez toho, aby se z ONNX vrstvy stal obecny dynamic-shape runtime nebo aby ONNX import automaticky tvrdil GPU podporu.

## Cil

Dotahnout ONNX podporu do stavu, kde je jasne:

- ktere importovane slozene ONNX opy se umi znovu exportovat jako canonical ONNX op;
- kde je staticka shape podpora zamerne hranice a kde chybi jen hardening test;
- ktere male modely dokazujeme jako prakticke kompatibilitni scenare;
- ktere coverage rows maji realny round-trip dukaz a ktere jsou jen explicitne klasifikovane;
- kde ONNX coverage konci a kde zacina samostatna GPU lowering/backend parity prace.

## Non-Goals

- Nezavadet runtime dynamic shapes.
- Neimplementovat `NonZero` jako runtime op.
- Nezavadet obecny ONNX executor pro `If`, `Loop`, `Scan`, sequence, map nebo optional hodnoty.
- Neclaimovat Metal/CUDA podporu jen proto, ze ONNX import/export funguje.
- Nezavadet ONNX-only semanticke operace, pokud jde ciste o rozpoznatelnou kompozici existujicich Synaptik primitiv.
- Nezavadet compatibility mezivrstvy pro stare ONNX API; cilem je cisty cilovy stav.

## Problem

ONNX vrstva je dnes funkcne siroka, ale zbyvaji ctyri typy nejistoty:

1. Nektere ONNX opy jsou import-only kompozice, ale export je neumi znovu slozit do canonical ONNX opu.
2. Staticke shape-helper cesty maji dobry zaklad, ale potrebujeme sirsi rejection coverage, aby se zadny runtime tensor nedostal do compile-time parametru.
3. Compat harness ma male modely, ale porad nepokryva dost model-like kombinaci redukci, normalizaci, indexingu a explicitnich rejection pripadu.
4. Coverage evidence je smichana: mnoho rows je `explicitly_classified`, ale jeste nema skutecny round-trip test.

To je problem hlavne proto, ze ONNX ma byt duveryhodna hranice interoperability. Pokud import funguje, export funguje a CPU execution funguje, musi byt presne zrejme, zda jde o plnou round-trip podporu, import-only lowering, staticky shape-only helper, nebo zamerne unsupported dynamic-shape pripad.

## Cilovy Model

ONNX kontrakt ma mit tri oddelene roviny:

1. **Interchange semantics**
   - mapovani mezi ONNX grafem a Synaptik tensor DAG;
   - canonical export recognizery pro slozene patterny;
   - jasne unsupported diagnostiky.

2. **Execution semantics**
   - CPU je correctness oracle;
   - importovany graf musi jit spustit, pokud status rika `EXECUTED`;
   - runtime `INT64` a dynamic output shape zustavaji mimo scope.

3. **Backend lowering semantics**
   - Metal/CUDA stav zustava oddeleny coverage sloupec;
   - GPU lowerability se neodvozuje z ONNX supportu;
   - GPU parity je samostatny audit/prace, ne vedlejsi efekt ONNX importu.

## Scope 1: Export Closure Pro Import-Only Composite Rows

### `ReduceL1`

Import dnes loweruje:

```text
ReduceL1(x, axes) = ReduceSum(abs(x), axes)
```

Wave 6 ma pridat export recognizer:

```text
abs(x).sum(axis, keepDims) -> ReduceL1
```

Minimalni scope:

- rozpoznat single-axis `abs -> sum`;
- zachovat `keepdims`;
- nepretvarovat pattern, pokud je `abs(x)` sdileny dalsim consumerem;
- nepretvarovat, pokud je mezivystup graph output.

### `ReduceL2`

Import dnes loweruje:

```text
ReduceL2(x, axes) = sqrt(ReduceSum(x * x, axes))
```

Wave 6 ma pridat export recognizer:

```text
sqrt(sum(x.mul(x), axis, keepDims)) -> ReduceL2
```

Minimalni scope:

- rozpoznat `MUL(x, x)` nebo ekvivalentni self-multiply;
- rozpoznat finalni `SQRT`;
- zachovat `keepdims`;
- multi-axis podporu resit jen pokud jde bezpecne z retezce opu odvodit z importovaneho lowering patternu.

### `ReduceLogSum`

Import dnes loweruje:

```text
ReduceLogSum(x, axes) = log(ReduceSum(x, axes))
```

Wave 6 ma pridat export recognizer:

```text
log(sum(x, axis, keepDims)) -> ReduceLogSum
```

### `ReduceLogSumExp`

Import dnes loweruje:

```text
ReduceLogSumExp(x, axes) = log(ReduceSum(exp(x), axes))
```

Wave 6 ma pridat export recognizer:

```text
log(sum(exp(x), axis, keepDims)) -> ReduceLogSumExp
```

Poznamka:

- tato forma neni numericky stabilizovana max-shiftem;
- recognizer smi canonicalizovat jen presny import/export pattern, ne obecnou matematicky ekvivalentni algebru.

### `GlobalAveragePool`

Import dnes loweruje na repeated spatial `mean`.

Wave 6 ma pridat export recognizer pro canonical rank>=3 global spatial mean:

```text
x.mean(2, true).mean(3, true) -> GlobalAveragePool
```

Minimalni scope:

- rank 4 NCHW jako prvni podporovany pripad;
- spatial axes `[2, 3]`;
- `keepdims=true`;
- nerozpoznavat, pokud intermediate mean ma dalsi consumery.

## Scope 2: Staticka Shape A Broadcast Hranice

Wave 5 zavedla jednotnou statickou konstantni branu pro `Tile`, reductions a `Squeeze`/`Unsqueeze`. Wave 6 ma doplnit dalsi hardening kolem vsech compile-time parametru.

### Dalsi rejection testy

Pridat testy, ze runtime vstupy jsou odmitnute pro:

- `Slice` `starts`, `ends`, `axes`, `steps`;
- `Pad` `pads`;
- `ConstantOfShape` `shape`;
- `CumSum` `axis`;
- `Split` `split`;
- `Clip` scalar `min`/`max`, pokud nejsou initializer nebo `Constant`;
- `Pow` exponent, pokud neni scalar initializer nebo `Constant`.

### Static helper success tests

Pridat pozitivni testy pro shape-only graph plumbing:

```text
Shape -> Gather -> Unsqueeze -> Concat -> Reshape
Shape -> Slice -> Concat -> Expand
Shape -> Size -> ConstantOfShape
```

Cil:

- dokazat, ze compile-time shape DSL funguje;
- soucasne dokazat, ze runtime data tensor neni povolen jako shape DSL hodnota.

### Broadcast diagnostics

Doplnit testy pro nekompatibilni broadcast:

- binary `Add/Mul`;
- ternary `Where`;
- scalar constant broadcast;
- bool mask broadcast.

Importer ma na hranici vracet `OnnxUnsupportedException`, ne nahodne `IllegalArgumentException` z vnitrni tensor vrstvy.

## Scope 3: Mini Model Suite Expansion

Compat fixture suite se ma rozsirit z jednotlivych op kombinaci na male model-like grafy.

### Navrzene nove fixture modely

1. `multi_axis_reduction_classifier.onnx`
   - `MatMul/Gemm -> Relu -> ReduceLogSumExp/ReduceMean`;
   - cil: composite reduction + NN flow.

2. `normalization_residual_broadcast.onnx`
   - `LayerNormalization -> Add residual -> Where mask`;
   - cil: broadcast, normalization, bool mask.

3. `index_select_scatter_tiny.onnx`
   - `Gather/GatherElements -> ScatterElements/ScatterND`;
   - cil: index dtype, shape, duplicate/reduction policy.

4. `conv_pool_global_average_tiny.onnx`
   - `Conv -> Relu/HardSigmoid -> GlobalAveragePool`;
   - cil: GlobalAveragePool import/export recognition.

5. `shape_rejection_dynamic_slice.onnx`
   - runtime `Slice` params;
   - status: `REJECTED_WITH_REASON`.

6. `shape_rejection_runtime_pow_exponent.onnx`
   - runtime exponent;
   - status: `REJECTED_WITH_REASON`.

### Fixture pravidla

- Checked-in `.onnx` soubor musi byt byte-for-byte generovan z builderu.
- Rejection fixture musi testovat konkretni reason substring.
- Execution fixture musi nastavovat vstupy a kontrolovat dtype, shape a hodnoty.
- Model nesmi vyzadovat dynamic shape runtime.

## Scope 4: Coverage Evidence Upgrade

`OnnxCoverageMatrix` ma zustat source of truth, ale wave 6 ma snizit pocet rows, ktere jsou jen `explicitly_classified`.

### Prioritni rows pro realny round-trip test

Pridat nebo rozsireni testu pro:

- `Sub`, `Mul`, `Div`;
- `Min`, `Max`, `Pow`;
- compare/logical family: `Equal`, `Greater`, `Less`, `And`, `Or`, `Not`;
- layout family: `Transpose`, `Reshape`, `Flatten`, `Expand`, `Squeeze`, `Unsqueeze`, `Slice`, `Concat`, `Tile`, `Pad`;
- reductions: `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd`;
- `ArgMax`, `CumSum`.

### Matrix policy

Pro kazdy radek plati:

- `SUPPORTED` export musi mit bud round-trip test, nebo explicitni komentar proc je pouze klasifikovany;
- import-only rows musi mit bud execution fixture, nebo rejection/shape-helper test;
- GPU status se nesmi rucne zlepsit bez backend lowering evidence.

## Scope 5: GPU Parity Audit Bez Implementace GPU Podpory

Wave 6 nema implementovat Metal/CUDA lowerery. Ma ale z ONNX pohledu presne vypsat, kde je ONNX support sirsi nez GPU support.

### Vystup auditu

Vytvorit kratky report nebo rozsireni coverage docs, ktere rozdeli GPU mezery:

1. **ONNX supported, GPU supported**
   - bez akce.

2. **ONNX supported, GPU unsupported but CPU supported**
   - legitimni CPU execution;
   - neoznacovat jako ONNX chybu.

3. **ONNX import-only static helper**
   - `Shape`, `Size`, `ConstantOfShape`, `Range`;
   - nema byt GPU op, pokud se vyhodnoti pri importu.

4. **GPU blocker podle dtype/layout/route**
   - napr. `INT32` vystup, bool mask, strided compute, unsupported CUDA route.

5. **True semantic blocker**
   - dynamic output shape, runtime `INT64`, unsupported control-flow.

### Pravidlo

ONNX coverage report muze rict "Metal unsupported", ale nesmi tim implikovat, ze ONNX import/export je nekompletni. Backend parity se resi v backend milestonech.

## Scope 6: Documentation And Developer Workflow

Aktualizovat ONNX dokumentaci tak, aby vysvetlovala rozdil mezi:

- ONNX op support;
- Synaptik semantic DAG support;
- CPU executability;
- accelerator lowerability;
- static shape helper;
- runtime dynamic shape.

### Dokumentacni priklady

Pridat priklady:

```text
Static shape supported:
Shape(x) -> Gather(dim) -> Concat -> Reshape(x)
```

```text
Rejected dynamic shape:
Reshape(x, runtime_shape_input)
```

```text
Import supported, GPU not guaranteed:
GatherND(data, runtime_int32_indices)
```

```text
Composite canonical export:
log(sum(exp(x), axis)) -> ReduceLogSumExp
```

### Workflow

Dopsat nebo zpresnit:

- jak regenerovat compat fixtures;
- jak regenerovat `docs/onnx-coverage.md`;
- jake testy spustit po pridani ONNX opu;
- jak zapsat rejection reason pro unsupported variantu.

## Implementacni Plan

1. Rozsirit `OnnxExportPatternRegistry`.
   - Pridat composite reduction recognizery.
   - Zachovat consumer-count ochrany z activation patternu.
   - Neexportovat pattern, pokud mezivystup je graph output.

2. Doplnit static-param helper coverage.
   - Pokud je potreba, prejmenovat helpery v importeru na jasnejsi `staticIntVectorInput` / `staticScalarInput`.
   - Sjednotit vsechny compile-time parametry pres jednu branu.

3. Pridat hardening test tridy.
   - `OnnxStaticParameterHardeningTest`
   - `OnnxCompositeReductionExportPatternTest`
   - rozsireni `OnnxShapeBroadcastHardeningTest`

4. Rozsirit `OnnxCompatibilityFixtureModels`.
   - Pridat model-like fixtures.
   - Regenerovat `.onnx` soubory.
   - Overit byte-for-byte harness.

5. Aktualizovat `OnnxCoverageMatrix`.
   - Export status pro nove recognizery.
   - Round-trip evidence rows.
   - Limitations text bez GPU overclaimu.

6. Aktualizovat docs.
   - `docs/onnx-coverage.md` generovat z matrix.
   - Doplnit workflow priklady do existujici ONNX dokumentace nebo zalozit novy docs soubor, pokud je tema uz prilis velke.

## Test Plan

Minimalni testy:

```bash
./gradlew classes testClasses
./gradlew test --tests 'onnx.*'
./gradlew test --tests SourceTreeHygieneTest
git diff --check
```

Focused testy:

```bash
./gradlew test --tests 'onnx.OnnxCompositeReductionExportPatternTest'
./gradlew test --tests 'onnx.OnnxStaticParameterHardeningTest'
./gradlew test --tests 'onnx.OnnxCompatibilityHarnessTest'
./gradlew test --tests 'onnx.OnnxCoverageMatrixTest'
```

Po zmene fixture builderu:

```bash
./gradlew testClasses
java --add-modules=jdk.incubator.vector \
  -cp build/classes/java/main:build/classes/java/test:<protobuf jar> \
  onnx.OnnxCompatibilityFixtureModels src/test/resources/onnx/compat
```

Po zmene coverage matrix:

```bash
java --add-modules=jdk.incubator.vector \
  -cp build/classes/java/main:<protobuf jar> \
  onnx.OnnxCoverageReport docs/onnx-coverage.md
```

## Success Criteria

- Composite reductions a `GlobalAveragePool` maji canonical export recognizery nebo explicitni dokumentovane odmítnuti, pokud pattern neni bezpecne rozpoznatelny.
- Runtime shape/axes/scalar parametry jsou odmitnuty konzistentni `OnnxUnsupportedException` diagnostikou.
- Compat suite obsahuje model-like fixtures pro reductions, normalization/residual/broadcast, indexing/scatter, conv/global-pool a rejection scenare.
- Pocet `round_trip_tested` rows v coverage matrix roste a `explicitly_classified` rows se snizuji u prioritnich export-supported opu.
- Coverage docs jasne oddeluji ONNX support od GPU lowerability.
- Testy `onnx.*` a `SourceTreeHygieneTest` prochazi.

## Assumptions

- Novy dokument je `todo/66-onnx-interchange-closure-wave-6.md`, protoze nejvyssi existujici todo je `65`.
- Wave 5 commit `756e89b` je vychozi stav ONNX coverage.
- Runtime dynamic shapes zustavaji mimo scope.
- `NonZero` zustava explicitni unsupported audit row.
- Metal/CUDA parity prace zustava mimo tento todo a ma byt planovana oddelene podle backend coverage matrix.
