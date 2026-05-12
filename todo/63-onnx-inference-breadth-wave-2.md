# ONNX Inference Breadth Wave 2

## Stav Rozpracovani

Status: `IMPLEMENTED`

Wave 1 uz zavedla `OnnxCoverageMatrix`, NN inference subset (`Conv`, `MaxPool`, `AveragePool`, `LayerNormalization`, import-only `BatchNormalization`) a checked-in NN fixtures.

Wave 2 ma rozsirit praktickou statickou dense inference kompatibilitu bez zavadeni obecneho ONNX runtime enginu. ONNX zustava interoperabilitni vrstva; internim zdrojem pravdy zustava Synaptik tensor graph, CPU kernels a backend lowering coverage.

## Cil

Pridat bezne inference operace, ktere se casto objevuji v exportech z PyTorch/ONNX toolchainu a ktere lze bezpecne mapovat na existujici nebo male nove Synaptik semantiky:

- `Pad`
- `Split`
- `Tile`
- `ArgMax`
- `ReduceProd`
- `GlobalAveragePool`

Kazda nova operace musi mit radek v `OnnxCoverageMatrix`, import/export testy, execution testy a jasne GPU status rozliseni. Pokud GPU native coverage neexistuje, zustane v matici explicitne `UNSUPPORTED` nebo `PARTIAL`; ONNX import nesmi predstirat backend paritu.

## Navrzeny Model

### `Pad`

- Importovat jen dense tensor `Pad` se statickymi `pads`.
- Podporovat `mode="constant"` a scalar `constant_value`.
- Default `constant_value=0`.
- Odmitnout `reflect`, `edge`, `wrap`, dynamicke pads a non-scalar constant.
- Pokud Synaptik nema first-class pad op, zavest `operations.layout.pad` + `Tensor.pad(...)` jako funkcionalni layout/value op, ne jako ONNX-only hack.
- CPU kernel vytvori novy output a vyplni padded oblasti konstantou.
- Backward pro floating training: gradient do inputu je slice z outGrad; gradient do constant value se nepodporuje.

### `Split`

- Protoze ONNX `Split` je multi-output node a importer dnes multi-output obecne odmita, nezavadet obecnou multi-output architekturu.
- Implementovat jen import-time lowering pro pattern, kde vsechny vystupy `Split` jsou primo definovane v ONNX graphu:
  - node muze mit vice vystupu;
  - importer ulozi kazdy vystup jako samostatny `Tensor.slice(...)`;
  - split sizes musi byt staticke;
  - axis se normalizuje;
  - output count musi odpovidat `split` sizes nebo rovnomernemu deleni.
- Export Synaptik `slice` zustava `Slice`; neexportovat vice `slice` nodu zpet jako `Split` v teto vlne.

### `Tile`

- Importovat `Tile(data, repeats)` s konstantnim `repeats`.
- Pokud existujici `expand` nestaci, zavest samostatny `tile` op, protoze `Tile` kopiruje opakovane bloky a neni broadcast view.
- CPU kernel musi fungovat pro rank >= 1 a staticke positive repeats.
- Odmitnout nulove/zaporne repeats a dynamicke repeats.
- Export `Tensor.tile(...)` jako ONNX `Tile` jen pokud bude zavedena first-class Synaptik operace.

### `ArgMax`

- Zavest inference-only `argMax(input, axis, keepDims)` s output dtype `INT32`.
- Importovat `axis`, `keepdims`, `select_last_index=0`.
- Odmitnout `select_last_index=1`, dokud nebude explicitni tie policy.
- CPU kernel vraci prvni maximalni index stejne jako ONNX default.
- Export jako `ArgMax` pro first-class Synaptik `argMax`.
- GPU coverage v matici zustane `UNSUPPORTED`, dokud nebude INT32 output compute podlozeny native backend coverage.

### `ReduceProd`

- Zavest `prod` reduction pro floating numeric tensors.
- Importovat static axes a `keepdims`.
- CPU kernel pouzije stejnou axis-normalization strategii jako `ReduceSum/Mean/Max/Min`.
- Backward muze byt odlozeny nebo omezeny na inference; pokud se prida backward, musi byt explicitne testovana zero-input semantika.
- Export jako `ReduceProd`.
- GPU coverage zustane `UNSUPPORTED` nebo `PARTIAL` podle noveho lowering row; neodvozovat automaticky z existujicich sum/mean reductions.

### `GlobalAveragePool`

- Importovat rank-4 NCHW `GlobalAveragePool` jako `mean` pres spatial axes `[2, 3]` s `keepdims=true`.
- Export se v teto vlne nemusi canonicalizovat z `mean(mean(x, 2, true), 3, true)` na `GlobalAveragePool`; staci import.
- CPU execution pouzije existujici reduction kernels.
- Metal/CUDA coverage se odvodi z reduction coverage, ale coverage matrix musi vysvetlit, ze ONNX op je kompozice dvou reductions, ne samostatny native pool primitive.

## Implementacni Plan

1. Rozsirit `OnnxCoverageMatrix`.
   - Pridat nove radky s konzervativnimi statusy.
   - Test musi opet vyzadovat coverage radek pro kazdy importer-supported op.

2. Pridat minimalni nove Synaptik operace jen tam, kde chybi cista interní semantika.
   - Pravdepodobne nove: `pad`, `tile`, `argMax`, `prod`.
   - Bez nove obecne multi-output infrastruktury; `Split` import lowering pouzije existujici `slice`.

3. Rozsirit ONNX importer/exporter.
   - Import: vsechny opy vyse.
   - Export: jen first-class Synaptik opy (`Pad`, `Tile`, `ArgMax`, `ReduceProd`) pokud budou zavedene.
   - `Split` a `GlobalAveragePool` zustanou import-only v teto vlne.

4. Pridat CPU kernels a prepare/type contracts.
   - `pad`: floating/int/bool value copy podle dtype.
   - `tile`: floating/int/bool copy podle dtype.
   - `argMax`: output `INT32`.
   - `prod`: floating reductions.

5. Pridat fixtures.
   - `src/test/resources/onnx/breadth/` s programatickym builderem `OnnxBreadthFixtureModels`.
   - Fixtures: constant pad, split axis 1, tile rank-2, argmax keepdims true/false, reduceprod axis, globalaveragepool.

6. Dokumentace.
   - Aktualizovat `docs/onnx.md` supported subset a non-goals.
   - Explicitne vysvetlit, ze `Split` je special-case multi-output lowering, ne obecna multi-output ONNX podpora.

## Test Plan

- ONNX unit/integration:
  - `./gradlew test --tests 'onnx.*'`
- Novy behavior:
  - `PadExecutionTest`
  - `TileExecutionTest`
  - `ArgMaxExecutionTest`
  - `ReduceProdExecutionTest`
- Regression:
  - `./gradlew test --tests DataTypeExecutionCoverageTest --tests PreparedExecutionBuildTest`
  - `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest`
  - `./gradlew test --tests SourceTreeHygieneTest`

## Implementacni Poznamky

- `Pad`, `Tile`, `ArgMax` a `ReduceProd` maji first-class Synaptik op, Tensor API, CPU kernel, ONNX import/export a coverage row.
- `Split` zustava zamerne import-only special-case lowering na staticke `Slice` vystupy.
- `GlobalAveragePool` zustava import-only lowering na opakovane `Mean` pres spatial axes.
- Breadth fixtures jsou checked-in pod `src/test/resources/onnx/breadth/` a generuje je `OnnxBreadthFixtureModels`.
- `OnnxBreadthFixtureTest` byte-porovnava generovane protobufy proti checked-in souborum a executuje vsechny deklarovane vystupy, vcetne obou vystupu `Split`.

## Assumptions

- Neimplementujeme obecne dynamic shapes, control flow, external data ani plnou multi-output ONNX architekturu.
- `Split` je zamerne special-case importer lowering.
- `ArgMax` vraci `INT32`, protoze runtime `INT64` tensors zustavaji mimo aktualni Synaptik dtype policy.
- GPU native podpora se nepridava automaticky; coverage matrix musi ukazat skutecny stav.
