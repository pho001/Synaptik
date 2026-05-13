# Metal Reduction/Scan Parity Wave

## Stav Rozpracovani

Status: `PLANNED`

Todo 68 uzavrelo dalsi layout/index parity mezeru pro Metal: `GATHER_AXIS`, `GATHER_AXIS_GRAD`,
`SLICE`, `CONCAT`, `PAD` a `TILE` maji coverage truth, planner legality, backend-neutral DAG reprezentaci,
native MPSGraph mapping, native parity testy a dokumentovany fallback pro nepodporovane tvary.

Nasledujici cista mezera v Metal operation parity matrix jsou redukcni a scan operace, ktere uz existuji
v public Tensor/ONNX/CPU vrstve, ale Metal je porad vede jako explicitne unsupported:

- `REDUCE_PROD`
- `ARGMAX`
- `CUMSUM`

Tyto operace patri do stejneho compile-time accelerator lowering proudu jako `SUM`, `MEAN`,
`REDUCE_MIN` a `REDUCE_MAX`. Nejde o graph rewrite ani o runtime policy. Planner ma rozhodnout,
jestli konkretni semanticky op ma legalni Metal lowering; pokud ne, fallback musi zustat viditelny
v trace/reportu se stabilnim duvodem.

## Problem

Aktualni stav je architektonicky jasny, ale parity matrix je stale nekompletni:

- `GpuLoweringCoverageMatrix` vraci pro `REDUCE_PROD`, `ARGMAX` a `CUMSUM` implicitni unsupported row.
- `docs/metal-operation-parity.md` tim spravne ukazuje CPU fallback, ale hot path coverage tim zbytecne
  zkracuje regiony, ktere by Metal dokazal provest jako nativni MPSGraph redukce nebo scan.
- ONNX coverage uz zna `ReduceProd`, `ArgMax` a `CumSum`, ale GPU status se odvozuje z GPU matrix,
  takze import/export support neni to same jako Metal lowerability.
- `ARGMAX` ma indexovy vystup (`INT32` v Synaptiku, protoze runtime nepodporuje `INT64` tensors),
  a proto nesmi byt smichany s FLOAT32/BFLOAT16 compute-output pravidly.
- `CUMSUM` je shape-preserving scan, ne axis-reducing reduction. Vyuziva podobne axis metadata,
  ale ma odlisnou vystupni shape semantiku, `exclusive` a `reverse` atributy.

## Cil

Promovat `REDUCE_PROD`, `ARGMAX` a `CUMSUM` na Metal-supported pouze pro jasne ohraniceny a testem
prokazany subset.

Po teto vlne ma platit:

- `REDUCE_PROD` loweruje jako nativni Metal/MPSGraph produktova redukce pro dense floating inputy.
- `ARGMAX` loweruje jako nativni Metal/MPSGraph indexova redukce a vraci `INT32` Synaptik tensor.
- `CUMSUM` loweruje jako nativni Metal/MPSGraph prefix scan pro statickou osu.
- Planner predem odmita nepodporovane dtype, rank, axis, layout a atributy se stabilnim reason code.
- Coverage docs, ONNX report a Metal parity matrix odpovidaji kodove pravde a neoverclaimuji obecny
  index/scatter nebo dynamic-shape support.

## Non-Goals

- Neimplementovat `GatherND`, `ScatterND`, `ScatterElements` ani dalsi index-write operace.
- Neimplementovat CUDA native support pro tyto operace v teto vlne.
- Neimplementovat `FLOAT64` Metal compute.
- Nezavadet obecne dynamic-shape axis vstupy; osa musi byt compile-time znama.
- Nezavadet public device tensor API.
- Nepridavat migracni nebo compatibility mezivrstvy.

## Cilovy Contract

### REDUCE_PROD

`REDUCE_PROD` je axis-reducing floating operation. Pro vstup `x` a osu `axis` vystup obsahuje
soucin hodnot pres danou osu. `keepDims=true` zachova redukovanou osu jako velikost `1`, `keepDims=false`
ji odstrani z public shape.

Podporovany Metal subset:

- input dtype: `FLOAT32`, `BFLOAT16`;
- output dtype: stejny floating dtype jako input;
- rank: 1..4, pokud native ABI a MPSGraph mapping nepodpori vic;
- layout: dense input nebo legalni GPU-side producer, ktery planner umi drzet v regionu;
- axis: staticky normalizovana osa;
- output shape: presne shodna s CPU/Synaptik shape kontraktem vcetne `keepDims`.

Explicitni odmitnuti:

- `BOOL` a `INT32` inputy jako `UNSUPPORTED_DTYPE`;
- non-dense strided runtime compute jako `UNSUPPORTED_LAYOUT`;
- rank/axis mimo contract jako `UNSUPPORTED_RANK_OR_SHAPE`;
- chybejici native symbol nebo ABI mismatch jako `NATIVE_ABI_MISMATCH`.

### ARGMAX

`ARGMAX` je indexova redukce. Hodnotove porovnava floating nebo integer vstup po ose a vraci index
maximalni hodnoty. Synaptik vystup je `INT32`, i kdyz ONNX defaultne pracuje s `INT64`, protoze aktualni
runtime nema plnohodnotny `INT64` tensor.

Podporovany Metal subset:

- input dtype: `FLOAT32`, `BFLOAT16`, pripadne `INT32` az pokud bude nativne prokazane;
- output dtype: `INT32`;
- rank: 1..4;
- axis: staticky normalizovana osa;
- `keepDims`: podporovat oba CPU/Synaptik rezimy, nebo jasne omezit a testovat podporovany subset;
- `select_last_index=false` zustava jediny podporovany ONNX rezim, protoze CPU/Synaptik tie policy musi byt
  deterministicky shodna.

Klicova semantika:

- tie policy musi odpovidat CPU. Pokud MPSGraph vraci posledni index a CPU prvni index, row nesmi byt
  oznacena jako supported bez kompenzacniho loweringu.
- `INT32` output neni floating compute output. DType role policy musi rozeznat `INDEX_OUTPUT`.
- output buffer binding a readback musi umet `INT32` bez skryteho CPU fallbacku pro samotnou operaci.

Explicitni odmitnuti:

- unsupported tie policy jako `UNSUPPORTED_INDEX_SEMANTICS`;
- `select_last_index=true` jako `UNSUPPORTED_INDEX_SEMANTICS`;
- nepodporovany dtype jako `UNSUPPORTED_DTYPE`;
- non-dense nebo dynamic axis jako `UNSUPPORTED_LAYOUT` nebo `UNSUPPORTED_RANK_OR_SHAPE`.

### CUMSUM

`CUMSUM` je prefix scan. Zachovava shape vstupu a pocita kumulativni soucet po jedne ose. Neni to
axis-reducing operation, prestoze sdili axis metadata s redukcemi.

Podporovany Metal subset:

- input dtype: `FLOAT32`, `BFLOAT16`, pripadne `INT32` jen pokud bude native parity prokazana;
- output dtype: stejny dtype jako input;
- rank: 1..4;
- axis: staticky normalizovana osa;
- `exclusive=false`, `reverse=false` jako minimalni prvni target;
- `exclusive=true` a `reverse=true` pridat pouze pokud native mapping nebo kompozice projde parity testy.

Klicova semantika:

- `exclusive` znamena, ze aktualni prvek neni zahrnut do vlastni prefix hodnoty.
- `reverse` znamena scan od konce osy smerem k zacatku.
- Pokud MPSGraph umi jen inclusive forward scan, ostatni varianty lze podporovat pouze pres explicitni
  GPU-side kompozici (`reverse`/`pad`/`slice`) bez CPU materializace.

Explicitni odmitnuti:

- `BOOL` input jako `UNSUPPORTED_DTYPE`;
- varianty `exclusive/reverse`, ktere nejsou native nebo GPU-side slozitelne, jako `CAPABILITY_MISSING`
  nebo `UNSUPPORTED_INDEX_SEMANTICS` podle skutecne priciny;
- non-dense strided runtime compute jako `UNSUPPORTED_LAYOUT`.

## Implementacni Plan

1. Rozsirit coverage truth.
   - Pridat explicitni Metal rows pro `REDUCE_PROD`, `ARGMAX` a `CUMSUM`.
   - CUDA rows nechat beze zmen nebo explicitne `CAPABILITY_MISSING`, dokud nema native CUDA execution.
   - `docs/gpu-lowering-coverage.md`, `docs/metal-operation-parity.md` a `docs/onnx-coverage.md`
     regenerovat z kodove pravdy.

2. Rozsirit backend-neutral DAG ABI.
   - Pridat `AcceleratorDagNodeType.REDUCE_PROD`, `ARGMAX` a `CUMSUM` s novymi ABI kody.
   - Pro redukce znovu pouzit existujici axis/keepDims metadata model, pokud je uz v `AcceleratorDagNode`.
   - Pro `CUMSUM` pridat nebo vyuzit atributy pro `axis`, `exclusive` a `reverse`.
   - Nepretizit scalar float payload pro boolean atributy; atributy maji byt explicitni integer/flag fields.

3. Doplnit lowerer mapping.
   - `AcceleratorSubgraphLowerer` musi prelozit `Operation.OpType.REDUCE_PROD` na novy DAG node.
   - `ARGMAX` musi propagovat output descriptor jako `INT32`, ne jako input floating dtype.
   - `CUMSUM` musi zachovat input shape a prenest scan atributy.
   - Compound region manifest musi uvest tyto operace jako lowered primitives, ne jako CPU boundary.

4. Implementovat Metal planner legality.
   - `MetalPartitionSupport` musi overit dtype, rank, axis, layout a atributy pred prijetim regionu.
   - Rejection details musi byt konkretni: dtype, layout, rank/axis, tie policy, scan variant nebo ABI.
   - Planner nesmi prijmout `ARGMAX`, pokud downstream/runtime neumi `INT32` output buffer bez CPU-only kroku.

5. Implementovat native MPSGraph mapping.
   - `REDUCE_PROD`: mapovat na MPSGraph product reduction nebo korektni native equivalent.
   - `ARGMAX`: mapovat na MPSGraph argmax a provest kontrolovanou konverzi vystupu na `INT32`, pokud native vraci sirsi index dtype.
   - `CUMSUM`: mapovat na MPSGraph cumulative sum/scan; varianty `exclusive` a `reverse` podporovat jen s dokazatelnou parity.
   - Rozsirit native compile ABI jen jednim cilovym symbolem/verzi, bez prechodneho dual-path contractu.

6. Doplnit native parity testy.
   - Planner/lowerer tests pro pozitivni i negativni pripady.
   - Bridge/native buffer tests pro `REDUCE_PROD`, `ARGMAX` a `CUMSUM`.
   - BF16 parity pro `REDUCE_PROD` a `CUMSUM`, pokud jsou BF16 rows oznacene jako supported.
   - `ARGMAX` parity s tie hodnotami, aby byla prokazana first/last policy.
   - Prepared execution test, ktery dokaze, ze region zustane na Metal a nevznikne skryta CPU materializace.

7. Aktualizovat reporty a benchmark gates.
   - Metal parity matrix ma ukazat podporovany subset a explicitni fallback pro zbytek.
   - ONNX coverage ma zustat pravdiva: import/export support neznamena automaticky GPU support.
   - Pridat nebo rozsireni hot-path coverage targetu pouze tehdy, kdyz tyto operace realne odstrani CPU boundary
     v reprezentativnim workloadu.

## Test Plan

```bash
./gradlew classes
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.metal.lowering.MetalRegionLowererTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.metal.lowering.MetalOperationParityMatrixTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests onnx.OnnxCoverageMatrixTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon test --tests OnnxWave3CoreOpsExecutionTest
JAVA_TOOL_OPTIONS=-Djdk.lang.Process.launchMechanism=FORK ./gradlew --no-daemon metalTest
git diff --check
```

## Success Criteria

- `REDUCE_PROD`, `ARGMAX` a `CUMSUM` maji explicitni Metal coverage rows.
- Planner prijima pouze legalni scoped subset a odmita zbytek stabilnim reason code.
- DAG ABI ma samostatne node typy a neprenasi axis/flags pres nejasny scalar hack.
- Native MPSGraph execution pro legalni pripady bezi pres buffer binding.
- `ARGMAX` vraci `INT32` hodnoty shodne s CPU vcetne tie policy.
- `CUMSUM` ma shodne inclusive/exclusive/reverse chovani jen pro varianty oznacene jako supported.
- Dokumenty jsou generovane z coverage truth a neoverclaimuji CUDA, F64, dynamic axis ani obecne index-write operace.

## Assumptions

- Metal MPSGraph ma dostupny ekvivalent pro product reduction, argmax a cumulative sum na podporovanem macOS/SDK.
- Pokud native MPSGraph vraci pro argmax index sirsi nez `INT32`, konverze na `INT32` je soucast Metal op loweringu,
  ne samostatna CPU materializace.
- BF16 product/cumsum parity muze vyzadovat sirsi toleranci nez elementwise BF16, protoze jde o akumulacni operace.
- Prvni implementacni vlna smi podporovat uzsi `CUMSUM` subset (`exclusive=false`, `reverse=false`), pokud coverage
  a rejection details zustanou presne.
