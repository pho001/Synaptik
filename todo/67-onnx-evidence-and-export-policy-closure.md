# ONNX Evidence And Export Policy Closure

## Stav Rozpracovani

Status: `IMPLEMENTED`

Wave 66 zavrela dalsi ONNX interchange mezery: composite reduction/global-pool export recognizery,
static-param hardening, mini compat modely a jasne oddeleni ONNX supportu od GPU lowerability.

Aktualni zbyvajici problem neni primarne chybejici import, ale dukazni kvalita coverage matrix:

- mnoho export-supported rows je porad jen `EXPLICITLY_CLASSIFIED`;
- nektere rows maji export-name smoke test, ale ne skutecny Synaptik graph -> ONNX export -> ONNX import -> CPU execute round trip;
- `Flatten` je coverage vedene jako export-supported, ale exporter musi dokazat canonical `Flatten` jen tehdy, kdyz `RESHAPE` presne odpovida ONNX flatten tvaru;
- leaf tensor export policy existuje, ale potrebuje explicitni kontrakt testy pro `INPUTS`, `INITIALIZERS` a `TRAINABLE_INPUTS`.

## Cil

Zpresnit ONNX evidence bez rozsirovani scope na GPU backend parity nebo dynamic-shape runtime.

Po teto vlne ma byt jasne:

- ktere export-supported ONNX rows maji skutecny round-trip dukaz;
- ktere rows zustavaji import-only nebo policy-only zamerne;
- ze coverage report neoverclaimuje accelerator podporu;
- ze leaf tensor export policy je testovana jako public ONNX export kontrakt.

## Non-Goals

- Neimplementovat Metal/CUDA lowering.
- Neimplementovat runtime dynamic shapes.
- Neimplementovat `NonZero`.
- Nezavadet ONNX control-flow, sparse, quantized, sequence, map nebo optional coverage.
- Neimplementovat `BatchNormalization` export recognizer v teto vlne.
- Nezavadet obecny `Constant` node export jako novou default policy; leaf values zustavaji graph inputs nebo initializers podle `OnnxLeafTensorPolicy`.

## Implementacni Plan

1. Rozsirit round-trip evidence testy.
   - Doplnit Synaptik graph -> ONNX export -> ONNX import -> CPU execute testy pro unary, compare, select, dtype, NN, layout, index, reduction a probability rows, ktere jsou dnes export-supported, ale jen explicitne klasifikovane.
   - Pouzit CPU jako correctness oracle.
   - Testovat hodnoty, nejen operator names.

2. Zpresnit `Flatten` export.
   - `RESHAPE` exportovat jako `Flatten` pouze tehdy, kdyz vstupni rank je vetsi nez 2, cilovy tvar ma rank 2 a existuje ONNX `axis`, pro ktery:
     `target[0] = product(inputShape[0:axis])` a `target[1] = product(inputShape[axis:rank])`.
   - Vsechny ostatni reshape pripady zustavaji canonical `Reshape`.

3. Doplnit leaf tensor policy testy.
   - `INPUTS`: vsechny leaf tensors jsou graph inputs, zadne initializers.
   - `INITIALIZERS`: vsechny leaf tensors jsou initializers, zadne graph inputs pro leaf storage.
   - `TRAINABLE_INPUTS`: `requiresGrad=true` leaf je graph input, netrainovatelny leaf je initializer.

4. Aktualizovat coverage matrix.
   - Do `roundTripTested(...)` pridat jen rows s novym skutecnym round-trip testem.
   - `Constant` ponechat `PARTIAL` a `EXPLICITLY_CLASSIFIED`, protoze exporter hodnoty serializuje pres initializers, ne pres ONNX `Constant` nodes.
   - `BatchNormalization` ponechat import-only, dokud nebude samostatny canonical export recognizer nebo first-class descriptor.

5. Regenerovat `docs/onnx-coverage.md`.
   - Coverage report zustava generovany z `OnnxCoverageMatrix`.
   - GPU status se nesmi rucne zlepsit.

## Test Plan

```bash
./gradlew test --tests onnx.OnnxPrimitiveRoundTripEvidenceTest
./gradlew test --tests onnx.OnnxExportImportTest
./gradlew test --tests onnx.OnnxCoverageMatrixTest
./gradlew test --tests 'onnx.*'
./gradlew test --tests SourceTreeHygieneTest
git diff --check
```

## Success Criteria

- `OnnxPrimitiveRoundTripEvidenceTest` pokryva hlavni drive explicitne klasifikovane export-supported rows.
- `Flatten` ma realny canonical export round-trip dukaz.
- `OnnxCoverageMatrix` snizi pocet `EXPLICITLY_CLASSIFIED` rows a zvysi `ROUND_TRIP_TESTED` rows.
- `docs/onnx-coverage.md` odpovida generovanemu reportu.
- `Constant`, `BatchNormalization`, shape-only helpery a dynamic-shape rows zustavaji pravdive popsane bez overclaimu.

## Assumptions

- ONNX evidence znamena semantic round trip pres CPU execution, ne GPU execution.
- Leaf policy je soucast export API kontraktu.
- GPU parity se resi v backend todo streamu, zejmena v todo 58.
