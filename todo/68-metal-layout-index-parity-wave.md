# Metal Layout/Index Parity Wave

## Stav Rozpracovani

Status: `IMPLEMENTED`

Tato vlna uzavira dalsi konkretni Metal parity mezeru po ONNX closure: runtime a ONNX cesta dnes pouziva
`GATHER_AXIS`, ale Metal coverage truth historicky podporovala hlavne starsi `GATHER`. Zaroven layout operace
`SLICE`, `CONCAT`, `PAD` a `TILE` byly CPU-only, prestoze jde o staticke shape operace, ktere casto jen zbytecne
rozbiji delsi Metal regiony.

## Cil

Udrzet verejny `Tensor` API logicky a rozsirit compile/prepare/execute Metal cestu tak, aby podporovala jasne
ohraniceny subset layout/index operaci bez ticheho fallbacku:

- `GATHER_AXIS` a `GATHER_AXIS_GRAD` jsou prvni trida v Metal coverage truth;
- `SLICE`, `CONCAT`, `PAD` a `TILE` maji vlastni accelerator DAG opy;
- staticke atributy layout opu se prenasi pres explicitni DAG atributovy ABI payload;
- fallback zustava viditelny pro nepodporovane tvary, dtype, layouty a slice kroky;
- `GATHER_ND`, `SCATTER_ND`, `REDUCE_PROD`, `CUMSUM` a `ARGMAX` zustavaji mimo tuto vlnu.

## Implementacni Plan

1. Sjednotit coverage truth.
   - `GpuLoweringCoverageMatrix` vede Metal `GATHER_AXIS`, `GATHER_AXIS_GRAD`, `SLICE`, `CONCAT`, `PAD` a `TILE`
     jako supported pouze pro podporovany subset.
   - ONNX `Gather` bere Metal status z `GATHER_AXIS`, ne z rucne nastaveneho unsupported stavu.
   - `MetalOperationParityMatrix` a generovane dokumenty odpovidaji kodove pravde.

2. Rozsirit backend-neutral DAG ABI.
   - Pridat DAG typy pro `GATHER_AXIS`, `GATHER_AXIS_GRAD`, `SLICE`, `CONCAT`, `PAD` a `TILE`.
   - Pridat osm integer atributu na DAG uzel pro staticke layout parametry.
   - Nepouzivat float scalar payload pro vektorove layout parametry; scalar zustava pro osu nebo konstantni pad value.

3. Implementovat Metal planner legality.
   - `GATHER_AXIS`: dense FLOAT32/BFLOAT16 value tensor, static leaf INT32 1-D indices, rank 1..4, staticky proverene bounds.
   - `GATHER_AXIS_GRAD`: stejny index contract plus dense FLOAT32/BFLOAT16 gradient.
   - `SLICE`: dense input, rank 1..4, staticke in-bounds start/end, `step=1`.
   - `CONCAT`: 2..5 vstupu, matching dtype/rank, staticka osa, dense vstupy nebo legalni GPU-side layout producer.
   - `PAD`: dense input, non-negative static before/after pads, constant scalar value.
   - `TILE`: dense input, positive static repeats.

4. Implementovat native Metal mapping.
   - `GATHER_AXIS` loweruje na `gatherAlongAxis` s broadcastovanou 1-D index shape.
   - `GATHER_AXIS_GRAD` loweruje na `scatterAlongAxis` add s broadcastovanou 1-D index shape.
   - `SLICE` loweruje jako chain statickych per-dimension `sliceTensor` opu.
   - `CONCAT`, `PAD` a `TILE` loweruji na odpovidajici MPSGraph shape/layout primitiva.
   - Attributed uzly pouzivaji novy `synaptik_apple_mps_compile_partition_dtype_v4` symbol; pokud chybi, fallback je capability-visible.

5. Doplnit testy a dokumenty.
   - Focus testy pro Metal planner/lowering support i explicitni rejection.
   - Regenerovat `docs/metal-operation-parity.md` a `docs/onnx-coverage.md`.
   - Lokalne nemenit ani necommitovat benchmark/calibration profily.

## Test Plan

```bash
./gradlew classes
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.metal.lowering.MetalRegionLowererTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.metal.lowering.MetalOperationParityMatrixTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests onnx.OnnxCoverageMatrixTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
./gradlew metalTest
git diff --check
```

## Success Criteria

- Metal planner prijima podporovane `GATHER_AXIS`, `GATHER_AXIS_GRAD`, `SLICE`, `CONCAT`, `PAD` a `TILE` pripady.
- Nepodporovany layout subset vraci konkretni `UNSUPPORTED_*` reason.
- Coverage matrix a ONNX report neoverclaimuji `GATHER_ND`, `SCATTER_ND`, `REDUCE_PROD`, `CUMSUM` ani `ARGMAX`.
- Native attributed layout/index DAG ma explicitni v4 ABI symbol misto implicitniho parsovani z output shape.

## Assumptions

- Tato vlna cilene podporuje 1-D ONNX `Gather` indices na Metal; multi-rank ONNX gather zustava budouci rozsireni.
- `SLICE` na Metal podporuje pouze `step=1`; strided slice vyzaduje samostatny contract.
- `CONCAT` je omezen poctem peti DAG vstupu, dokud se nerozsiri variadic DAG ABI.
