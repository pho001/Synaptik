# ONNX Inference Breadth Wave 3

## Stav Rozpracovani

Status: `IMPLEMENTED`

Wave 2 uz pokryla prakticke staticke dense inference operace:

- `Pad`
- `Split`
- `Tile`
- `ArgMax`
- `ReduceProd`
- `GlobalAveragePool`

Wave 3 ma navazat na body 3 az 5 z dalsiho doporuceneho postupu:

3. staticke shape-helper operace,
4. dalsi redukcni family,
5. `CumSum` a audit datove dynamickych index opu.

Cilem neni zavadet obecny ONNX runtime engine, dynamic shape executor ani plnou podporu datove zavislych vystupnich tvaru. ONNX zustava import/export vrstva nad Synaptik tensor graph modelem.

## Cil

Rozsirit staticky dense inference subset o operace, ktere casto vypadnou z PyTorch/ONNX exporteru okolo shape skladani, normalizaci, loss-adjacent vypoctu a sekvencnich modelu:

- `ConstantOfShape`
- `Range`
- `ReduceL1`
- `ReduceL2`
- `ReduceLogSum`
- `ReduceLogSumExp`
- `CumSum`
- `NonZero` audit a rozhodnuti, ne plna implementace v teto vlne

## Navrzeny Scope

### 1. Static Shape Helpers

#### `ConstantOfShape`

Semantika ONNX:

- vstup je shape tensor,
- vystup je tensor daneho tvaru,
- hodnota je scalar atribut `value`, default je `0`.

Scope wave 3:

- importovat jen pripad, kde shape vstup je staticky `INT64`/`INT32` constant;
- vytvorit leaf `Tensor` s materializovanou konstantni hodnotou a vystupnim tvarem;
- podporovat `FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `BOOL`;
- export neni nutny, protoze Synaptik zatim nema first-class `constantOfShape` semantic op;
- coverage row bude `import=SUPPORTED`, `export=UNSUPPORTED`, `cpu=SUPPORTED`, GPU status podle toho, ze jde o konstantni leaf, ne native op.

Odmítnout:

- runtime/dynamicky shape input,
- non-scalar `value`,
- `INT64` runtime vystup.

#### `Range`

Semantika ONNX:

- `Range(start, limit, delta)` vytvori 1D sekvenci hodnot.

Scope wave 3:

- importovat pouze staticky constant pripad;
- pokud vsechny vstupy jsou `INT64` shape constants, vysledek zustane importer-internal `int64Constants`;
- pokud vstupy jsou scalar tensor initializers/`Constant` nodes podporovane dtype, vytvorit konstantni leaf `Tensor`;
- podporovat pozitivni i negativni `delta`;
- odmitnout `delta=0`;
- odmitnout runtime `Range`, protoze delka vystupu je datove zavisla.

Priklad podporovaneho shape-helper patternu:

```text
Constant(start=0)
Constant(limit=hidden)
Constant(delta=1)
Range
  -> Gather/Slice/Reshape shape plumbing
```

### 2. Reduction Family

#### `ReduceL1`

Mapovani:

```java
abs(input).sum(axis, keepDims)
```

Multi-axis import:

- stejne jako `ReduceSum`: repeated single-axis lowering;
- `keepdims=false` redukovat osy sestupne;
- `keepdims=true` redukovat osy vzestupne.

#### `ReduceL2`

Mapovani:

```java
input.mul(input).sum(axis, keepDims).sqrt()
```

Poznamka:

- pro multi-axis redukci musi byt `sqrt` az po cele redukci, ne po kazde ose;
- jinak by se zmenila matematika.

#### `ReduceLogSum`

Mapovani:

```java
input.sum(axis, keepDims).log()
```

Poznamka:

- `log` musi byt az po cele redukci pres vsechny osy.

#### `ReduceLogSumExp`

Zakladni mapovani:

```java
input.exp().sum(axis, keepDims).log()
```

Numericka stabilita:

- minimalni wave 3 muze zacit primym `exp -> sum -> log`;
- pokud testy ukazou overflow problem, doplnit stabilni lowering:

```java
m = input.max(axis, true)
log(sum(exp(input - m))) + squeeze_or_keep(m)
```

Rozhodnuti pro wave 3:

- preferovat jednoduchy lowering pro prvni implementaci;
- explicitne dokumentovat, ze stabilni varianta je dalsi hardening, pokud bude potreba.

Export:

- nekanonikalizovat slozene Synaptik grafy zpet na `ReduceL1/L2/LogSum/LogSumExp` v teto vlne;
- export status `UNSUPPORTED`, dokud nebude first-class descriptor nebo canonical export recognizer.

### 3. CumSum

Semantika ONNX:

- `CumSum(data, axis)` pocita prefixove soucty podel osy;
- atributy `exclusive` a `reverse` meni, zda prefix zahrnuje aktualni prvek a smer scanovani.

Scope wave 3:

- zavest first-class Synaptik op `cumSum(axis, exclusive, reverse)`;
- vystup ma stejny shape a dtype jako input;
- podporovat floating dtypes a `INT32`;
- `BOOL` odmítnout;
- axis musi byt staticky scalar `INT32`/`INT64` constant;
- importovat `exclusive=0/1`, `reverse=0/1`;
- exportovat first-class `cumSum` jako ONNX `CumSum`;
- CPU kernel implementovat layout-aware pres logical indexing, podobne jako layout/index kernely;
- backward pro training muze zustat nepodporovany nebo odlozeny, protoze ONNX inference vlna nepotrebuje gradient.

Priklady:

```text
data = [1, 2, 3], axis=0
CumSum(exclusive=0, reverse=0) -> [1, 3, 6]
CumSum(exclusive=1, reverse=0) -> [0, 1, 3]
CumSum(exclusive=0, reverse=1) -> [6, 5, 3]
CumSum(exclusive=1, reverse=1) -> [5, 3, 0]
```

### 4. NonZero Audit

`NonZero` ma datove zavisly vystupni shape:

```text
input shape: [N]
output shape: [rank(input), number_of_nonzero_values]
```

`number_of_nonzero_values` neni znamy ze statickeho shape, ale az z dat. To narazi na aktualni Synaptik predpoklad statickych tensor tvaru v compile graphu.

Scope wave 3:

- neimplementovat full `NonZero`;
- pridat rozhodovaci audit do todo nebo docs:
  - kde se `NonZero` typicky objevuje v exportech,
  - zda ho potrebujeme pro inference modely, ktere chceme nacitat,
  - jake architektonicke zmeny by vyzadoval datove zavisly output shape,
  - zda existuje uzitecny static-only subset.

Mozny static-only subset:

- pokud input je constant initializer, `NonZero` by mohl byt vyhodnocen import-time jako `INT64`/`INT32` constant;
- to je ale shape/data constant folding, ne runtime op.

Rozhodnuti pro wave 3:

- povolit pouze import-time constant folding, pokud je trivialni a bezpecne;
- jinak jen explicitni unsupported diagnostic s dokumentovanym duvodem.

## Implementacni Plan

1. Rozsirit `OnnxCoverageMatrix`.
   - Pridat radky pro `ConstantOfShape`, `Range`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `CumSum`, `NonZero`.
   - `NonZero` muze zustat `UNSUPPORTED` s jasnym duvodem.
   - GPU status nikdy neodvozovat optimismem; u lowering kompozic vypsat mapovane Synaptik opy.

2. Pridat import-time constant helpers.
   - Rozsirit importer o materializaci statickych konstant pro `ConstantOfShape`.
   - Rozsirit importer o staticky `Range`.
   - Zachovat `INT64` jako shape-only constant, ne runtime tensor dtype.

3. Pridat reduction lowering.
   - Rozsirit internal `ReductionKind` nebo zavest helper pro composed reductions.
   - Testovat multi-axis `keepdims=true/false`.
   - U `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp` aplikovat final `sqrt/log` az po cele multi-axis redukci.

4. Zavest `CumSum`.
   - `operations.reduction.cumSum` nebo samostatna category podle finalniho rozhodnuti.
   - `Tensor.cumSum(axis)` a overloady pro `exclusive/reverse`.
   - CPU kernel pro FLOAT64/FLOAT32/BFLOAT16/INT32.
   - Type contract a prepare policy.
   - ONNX import/export.

5. Fixtures.
   - `src/test/resources/onnx/breadth_wave3/` nebo rozsirit `onnx/breadth/`.
   - Builder: `OnnxBreadthWave3FixtureModels` nebo rozsirit `OnnxBreadthFixtureModels`.
   - Pokryt:
     - `ConstantOfShape` float scalar,
     - `ConstantOfShape` bool/int scalar,
     - static `Range` positive/negative delta,
     - `ReduceL1`,
     - `ReduceL2`,
     - `ReduceLogSum`,
     - `ReduceLogSumExp`,
     - `CumSum` normal/exclusive/reverse.

6. Dokumentace.
   - Aktualizovat `docs/onnx.md`.
   - Vysvetlit rozdil mezi:
     - import-time shape constant,
     - constant tensor leaf,
     - runtime tensor op.
   - Vysvetlit proc `NonZero` neni bez obecne dynamic-shape podpory normalni runtime op.

## Implementacni Poznamky

- `ConstantOfShape` je import-time materializace konstantniho leaf tensoru. Nepřibyl first-class runtime op.
- `Range` je import-time folding. Cisty `INT64` pripad zustava shape-only `long[]`; scalar tensor initializers/`Constant` nodes vytvari materializovany tensor leaf podporovane dtype.
- `ReduceL1`, `ReduceL2`, `ReduceLogSum` a `ReduceLogSumExp` se importuji jako kompozice existujicich Synaptik opu. Export techto slozenych vzoru zustava nepodporovany, dokud nebude canonical recognizer nebo first-class descriptor.
- `CumSum` je first-class `operations.reduction.cumSum` s public API `Tensor.cumSum(axis)` a `Tensor.cumSum(axis, exclusive, reverse)`, CPU kernelem pro `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32` a ONNX import/export podporou.
- `CumSum` nema GPU lowering. `OnnxCoverageMatrix` proto ukazuje ONNX/CPU support oddelene od Metal/CUDA supportu.
- `NonZero` zustava unsupported coverage row. Duvod je datove zavisly output shape `rank(input) x number_of_nonzero_values`, ktery aktualni staticky compile graph neumoznuje reprezentovat jako normalni runtime tensor op.

## Verification

- `./gradlew classes testClasses`
- `./gradlew test --tests OnnxWave3CoreOpsExecutionTest --tests OnnxWave2CoreOpsExecutionTest --tests 'onnx.*' --tests DataTypeExecutionCoverageTest --tests PreparedExecutionBuildTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalOperationParityMatrixTest --tests SourceTreeHygieneTest`

## Test Plan

- ONNX:
  - `./gradlew test --tests 'onnx.*'`
- New execution:
  - `CumSumExecutionTest`
  - rozsirene ONNX fixture testy
- Regression:
  - `./gradlew test --tests DataTypeExecutionCoverageTest`
  - `./gradlew test --tests PreparedExecutionBuildTest`
  - `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest`
  - `./gradlew test --tests backend.metal.lowering.MetalOperationParityMatrixTest`
  - `./gradlew test --tests SourceTreeHygieneTest`

## Non-Goals

- obecne dynamic shapes;
- runtime `Range` s datove zavislou delkou;
- full runtime `NonZero`;
- ONNX control flow;
- sparse/sequence/map/optional hodnoty;
- canonical export recognizer pro slozene reduction patterny;
- GPU native implementace pro nove operace, pokud ji samostatne nepridame a neotestujeme.

## Assumptions

- Wave 3 bude stale staticky dense inference subset.
- `ConstantOfShape` a `Range` jsou primarne importer helpers, ne nove backend execution primitiva.
- `CumSum` je jedina pravdepodobna first-class nova runtime operace v teto vlne.
- `NonZero` je architektonicky citlive kvuli datove zavislemu output shape, proto patri nejdriv do auditu.
