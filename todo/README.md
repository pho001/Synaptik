# Todo

Tento adresar drzi aktualni architektonicke ukoly a navrhy dalsich vetsich refaktoru.

## Stav

Nove a aktivne upravovane plany maji primo v dokumentu sekci `Stav Rozpracovani`.

Pouzivane hodnoty:

- `PLANNED` - navrh je pripraveny, implementace nezacala.
- `ACTIVE DESIGN / PARTIAL IMPLEMENTATION EXISTS` - cast infrastruktury uz existuje, plan popisuje dalsi sjednoceni nebo rozsireni.
- `PARKED / DESIGN ONLY` - zamerne odlozeny nebo slovnikovy navrh, nema blokovat aktualni implementaci.
- `IMPLEMENTED / HISTORICAL` - starsi plan je hotovy nebo slouzi jako historicky kontext.

## Ukoly

- [01-fused-operation-backend-refactor.md](./01-fused-operation-backend-refactor.md)
  - Dokonceny refaktor fused execution. `FusedOperation` je descriptor, codegen bezi pres `FusedExpressionPlan` a runtime executable vznikaji az v prepared metadata.


- [04-tensor-compiledgraph-api-cleanup.md](./04-tensor-compiledgraph-api-cleanup.md)
  - Finalni navrh cisteho rozdeleni odpovednosti mezi `Tensor`, `CompiledGraph`, `PreparedExecution` a `ExecutionProfile`, vcetne odstraneni compile cache a optimizer overloadu z `Tensor` a navaznosti na autotune.

- [05-framework-maturity-roadmap.md](./05-framework-maturity-roadmap.md)
  - Velmi podrobny maturity plan: co chybi, aby Synaptik prestal byt jen experimental framework a byl ferove povazovany nejdriv za serious internal framework a pozdeji za public production-ready framework.

- [06-formalize-broadcast-semantics.md](./06-formalize-broadcast-semantics.md)
  - Detailni navrh, jak formalizovat broadcasting semantics v engine vrstve: shape compatibility rules, rank mismatch contract, reduction interoperability (`keepDims`), backward gradient reduction a odpovidajici test/docs plan.

- [07-memory-planner-stage-2.md](./07-memory-planner-stage-2.md)
  - Detailni navrh druhe etapy memory planneru: vratit cast efektivity po correctness-safe refaktoru `MEM` stage pomoci interval packing, lepsiho release po poslednim backward use, explain tooling a jemnejsi phase-aware reuse politiky.

- [08-compare-select-algebra-roadmap.md](./08-compare-select-algebra-roadmap.md)
  - Kdy a proc ma smysl zavest compare/select algebru do Synaptiku, ktere frameworky ji maji, jaky minimalni op set by daval smysl a proc to zatim neni prvni priorita pro MLP/CNN/transformer minimum.

- [09-expand-stage-2-true-view.md](./09-expand-stage-2-true-view.md)
  - Detailni navrh druhe etapy pro `expand(...)`: prechod z dnesni materialized expand operace na true zero-stride alias view s explicitnim read/write contractem, backend auditem a memory-planner integraci.

- [10-storage-dtype-first-refactor.md](./10-storage-dtype-first-refactor.md)
  - Must-have refaktor storage vrstvy: odstranit `double` bridge z `TensorStorage`, prejit na dtype-first pristup napric `Tensor`, `TensorRemap`, `TensorLayoutTransform`, plannerem a backend fallbacky a pripravit cistou pudu pro `BOOL` dtype a compare/select algebru.

- [11-bool-dtype-foundation.md](./11-bool-dtype-foundation.md)
  - Konkretni implementacni batch pro zavedeni `DataType.BOOL` a `BoolStorage`: typed access v `Tensor`, remap/layout podpora, planner/memory accounting a regression test matrix. Je to bezprostredni prerequisite pred compare/select operation layer.

- [12-loss-api-roadmap.md](./12-loss-api-roadmap.md)
  - Roadmapa loss API po zavedeni `softmax`, `logSoftmax`, `nllLoss` a `crossEntropyLoss`: co je hotove, proc byla prvni verze dense-target, kdy ma smysl specializovany `CROSS_ENTROPY_LOSS`, a v jakem poradi resit index-based targets, `ignore_index`, reduction modes, class weights a label smoothing.

- [13-fused-optimizer-target-redesign.md](./13-fused-optimizer-target-redesign.md)
  - Cílový redesign fused optimizeru a fused codegenu po zavedení true view/offset modelu, `where`, compare-select algebry a indexing kernelů. Dokument záměrně jde rovnou na finální architekturu bez compatibility vrstev, helper přechodů a mezistavů. Doplnen follow-up pro unified direct fused executor family (`BF16`/`F32`/`F64`) s `Vector API` jako primarni path a generated ASM jako fallback.

- [14-attention-hot-path-specialization.md](./14-attention-hot-path-specialization.md)
  - Odlozeny navrh attention hot-path optimalizaci. `scaledDotProductAttention(...)` ma zustat kompozicni, zatimco backend/planner mohou dostat shape-specialized runtime paths a tuning knobs pro `matmul`, `where` a `softmax` casti bez navratu k obri monoliticke fused attention implementaci.

- [15-onnx-import-export-integration.md](./15-onnx-import-export-integration.md)
  - Detailni navrh ONNX exportu a importu a jejich zacleneni do Synaptiku: kam patri IR mapping, jak oddelit public I/O surface od graph/runtime jadra, jak resit operator coverage, shape/dtype kompatibilitu, fallbacky a jak ONNX workflow navazat na `Tensor`, `CompiledGraph` a budouci `tuning` vrstvu.

- [17-fused-runtime-routing-and-asm-f32-analysis.md](./17-fused-runtime-routing-and-asm-f32-analysis.md)
  - Detailni rozbor fused runtime routing regrese a rozdilu `ASM/F32` vs `ASM/F64`: presne execution pruchody, root cause v capability modelu direct backendu, analyza `pow` specializace v ASM vector path a navrh ciloveho reseni pro resolver i autotune candidate filtering.

- [18-fused-backend-split-asm-scalar-direct-vector.md](./18-fused-backend-split-asm-scalar-direct-vector.md)
  - Cílový implementační návrh na odstranění duplicitní ASM vector větve: `ASM` bude scalar-only generated backend, `direct` bude jediná vector/runtime family. Dokument detailně popisuje, co smazat, co ponechat, jak upravit prepared contract a jak vyčistit autotune surface bez mezivrstev a compatibility hacků.

- [23-tuning-target-architecture.md](./23-tuning-target-architecture.md)
  - Jeden autoritativní cílový návrh přepisu tuning vrstvy: rozdělení na benchmark, per-graph autotune a platform calibration, oddělení `PlatformRuntimeProfile` a `GraphExecutionPolicy`, pravidla pro scoring, persistence a finální skládání do `ExecutionProfile`.

- [24-high-performance-backend-roadmap.md](./24-high-performance-backend-roadmap.md)
  - Velmi detailní výkonová roadmapa backendu: oddělení storage/compute/backend resolveru, GEMM hardening, BF16 real performance path, fused throughput backend, layout/materialization planner, scheduler refinement, conv runtime decision a performance observability.

- [22-execution-overhead-analysis.md](./22-execution-overhead-analysis.md)
  - Detailní rozpad current execution overheadu: compile, prepare, execute, per-step CPU dispatch, root sync, zeroGrad a trace-only náklady. Dokument určuje, co je třeba samostatně změřit, než se začne optimalizovat runtime hot path.

- [26-conv2d-calibration-and-dispatch-roadmap.md](./26-conv2d-calibration-and-dispatch-roadmap.md)
  - Detailni cilovy plan pro spravnou `conv2d` kalibraci: oddeleni `lowering` vs `GEMM dispatch` vs budouci Java/backend families, navrh workload bucketu, score policy a seznam `conv2d` knobu, ktere opravdu dava smysl kalibrovat.

- [27-cpu-backend-planning-refactor.md](./27-cpu-backend-planning-refactor.md)
  - Cílový refaktor CPU planning vrstvy: compile-time planning zůstává mimo execute, family-specific planning se vytahuje z centralizovaných tříd a backend structure se zarovnává s execution families.

- [28-graph-execution-isolation-and-immutability.md](./28-graph-execution-isolation-and-immutability.md)
  - Detailni roadmapa na oddeleni semantickeho `Tensor` graphu od compiled/prepared/run state: immutable compiled nodes, per-run execution state, autograd build bez globalniho graph mutating state a backend prepare boundary mimo `CompiledGraph`.

- [29-semantic-forward-canonicalization-before-autograd.md](./29-semantic-forward-canonicalization-before-autograd.md)
  - Detailni plan na rozdeleni compile pipeline na semantic-safe forward canonicalization, autograd build nad uz kanonizovanym forward graphen a nasledny joint optimizer nad `forward + backward` graphen.

- [30-strided-materialization-planning.md](./30-strided-materialization-planning.md)
  - Detailni plan na nahrazeni hrubeho `boolean stridedPath` explicitnim compile-time layout rozhodnutim, ktere umi volit mezi `KEEP_STRIDED`, selektivni materializaci vstupu a bezruntime policy branching na elementwise strided path.

- [31-matmul-prepared-executable-refactor.md](./31-matmul-prepared-executable-refactor.md)
  - Detailni cilovy plan na rozdeleni `matmul` prepare-time execution path: `MATMUL` zustane jeden semanticky op, ale `prepare` bude vytvaret konkretni dtype/backend-specific executable a runtime prijde o pozdni branching na `F64/F32/BF16`, BLAS a continuation cesty.

- [39-backend-architecture-cleanup-and-reorganization.md](./39-backend-architecture-cleanup-and-reorganization.md)
  - Navazujici backend cleanup po graph architecture cleanupu: cisty package ownership pro `backend`, rozdeleni generic prepare/lowering/partition kontraktu od CPU/Metal/CUDA implementaci, prejmenovani Apple target-level a bridge kodu na Metal bez aliasu a mezivrstev, sjednoceni CPU package rootu, registry cleanup a performance non-regression gates.

- [40-new-autotune-architecture-rewrite.md](./40-new-autotune-architecture-rewrite.md)
  - Plan cisteho prepisu autotune architektury: oddelit graph autotune, platform calibration a benchmark, nahradit stage-order candidate space standardnim `graphPolicy=current`, zavest strukturovana candidate metadata/persistence a ponechat CSE/piecewise/memory zmeny jen jako explicitni research rezim.

- [41-tuning-package-reorganization-and-calibration-ux.md](./41-tuning-package-reorganization-and-calibration-ux.md)
  - Plan ciste reorganizace `tuning` balicku kolem kalibrace/autotune/benchmark ownershipu: sjednoceny registry runtime calibration families, odstraneni mrtvych/duplicitnich knobu, versioned per-platform/per-dtype historie kalibraci a citelny barevny live progress se Synaptik bannerem.

- [42-public-api-javadocs.md](./42-public-api-javadocs.md)
  - Plan dokumentacni faze pro doplneni kvalitnich JavaDoc komentaru ke vsem public/protected Java API v `src/main/java` bez zmen runtime chovani, refaktoru nebo prejmenovani symbolu.

- [43-metal-mps-capability-sdpa-and-native-build.md](./43-metal-mps-capability-sdpa-and-native-build.md)
  - Plan cleanupu Metal/MPS hranice: dtype-aware legality pro planner, bezpecne povoleni direct forward SDPA, explicitni `nativeBuild`/`metalTest` workflow pro shim a vedome odlozeni zero-copy memory modelu do samostatne architektonicke faze.

- [44-cpu-execution-regions-fusion-and-graph-autotune.md](./44-cpu-execution-regions-fusion-and-graph-autotune.md)
  - Plan redesignu regioningu a fusion: `PART` vytvari accelerator ownership regiony a CPU natural execution regiony, `FUSE` uvnitr regionu tvori execution units/fused loops, graph autotune konsoliduje offload, CPU region, CPU fusion a accelerator region policy bez zasahu do platform calibration knobu.

- [45-metal-zero-copy-regions-sdpa-and-stress-workloads.md](./45-metal-zero-copy-regions-sdpa-and-stress-workloads.md)
  - Plan dalsi Metal vykonove etapy: observabilita transferu, explicitni storage residency, shared-buffer/zero-copy cesta, vetsi Metal ownership regiony, SDPA training coverage a vetsi transformer stress workloady bez skrytych CPU/GPU round-tripu.

- [46-metal-native-buffer-zero-copy-execution.md](./46-metal-native-buffer-zero-copy-execution.md)
  - Follow-up k fazi 45, ktery dodava skutecnou native Metal buffer-binding cestu: `execute_partition_f32_buffers` ABI, realne `MTLBuffer` ownership/lifetime, zapnuti `supportsBufferBindings()` az po testech, Metal device-to-CPU materializer a overeni Metal-to-Metal handoffu bez Java `float[]` copy-back.

- [47-accelerator-buffer-policy-and-prepared-inputs.md](./47-accelerator-buffer-policy-and-prepared-inputs.md)
  - Plan zobecneni buffer-binding rozhodovani z Metal-only cesty na spolecny accelerator model pro Metal i budouci CUDA. Soucasti je per-backend `OFF/AUTO/REQUIRE` policy, reuse prepared contiguous input path pres `CpuNodeExecutionPlan.apply(...)`, oddeleni execution-local prepared input uploadu od semantic device-current residency a autotune knob pro volbu buffer path per graf.

- [48-optimizer-stage-split-and-fixpoint-cleanup.md](./48-optimizer-stage-split-and-fixpoint-cleanup.md)
  - Status: `PLANNED`. Plan rozdeleni optimizer pipeline na jasne stage `AR`, `CF`, `CSE`, `DCE` s fixpoint/max-iteration guardem a cost/size kontrolou.

- [49-region-aware-lowering-after-partitioning.md](./49-region-aware-lowering-after-partitioning.md)
  - Status: `PLANNED`. Plan presunu backend-sensitive loweringu az za partitioning, aby region owner mohl rozhodnout, ktere high-level opy lowerovat a ktere ponechat backend primitive/library vrstve.

- [50-shared-cost-model-vocabulary.md](./50-shared-cost-model-vocabulary.md)
  - Status: `IMPLEMENTED_FOUNDATION`. Plan sjednoceni slovniku pro estimated work, compute cost, copy cost, materialization cost a fallback cost napric optimizerem, partitioningem a backend routingem.

- [51-compiled-tensor-descriptor-index.md](./51-compiled-tensor-descriptor-index.md)
  - Status: `IMPLEMENTED`. Immutable `CompiledTensorDescriptor` index je compile-time zdroj pravdy pro dtype, shape, strides, storage offset, layout fakta a `requiresGrad` snapshot.

- [52-metal-backend-mpsgraph-first-region-lowering.md](./52-metal-backend-mpsgraph-first-region-lowering.md)
  - Status: `IMPLEMENTED`. Metal regiony se defaultne loweruji jako jeden MPSGraph DAG (`METAL_GRAPH_REGION`), fused elementwise zustava metadata, CPU `Operation.OpType.FUSED` se na Metal neprenasi a `MIN`/`MAX`/scalar `POW` jsou doplneny do DAG/native coverage.

- [53-graph-backend-cleanup-boundaries.md](./53-graph-backend-cleanup-boundaries.md)
  - Status: `PLANNED`. Cleanup plan pro jasne hranice mezi graph semantics, optimizer stages, backend lowering, capability truth, layout descriptors, autograd rules, fusion pojmy a route/copy evidence.

- [54-trace-gates-cleanup-and-evidence-contract.md](./54-trace-gates-cleanup-and-evidence-contract.md)
  - Status: `PLANNED`. Cleanup plan pro canonical trace/gates evidence schema, backend-neutral route/copy fields, gate policy skupiny a report contract testy.

- [55-cost-explanation-trace-report-rendering.md](./55-cost-explanation-trace-report-rendering.md)
  - Status: `IMPLEMENTED_FOUNDATION`. Plan napojeni `CostExplanation` z todo 50 do benchmark JSON/text reportu, compile trace a coverage/gate vystupu bez zmeny score rozhodovani.

- [56-metal-route-cost-adapter.md](./56-metal-route-cost-adapter.md)
  - Status: `IMPLEMENTED_FOUNDATION`. Plan report-only adapteru pro Metal prepare-time route rozhodnuti (`MPS_GRAPH`, `TENSOR_ARRAY`, `CPU_FALLBACK`, `UNAVAILABLE_REQUIRED`) pres shared cost vocabulary.

- [57-autotune-benchmark-cost-adapter.md](./57-autotune-benchmark-cost-adapter.md)
  - Status: `PLANNED`. Plan adapteru pro benchmark/autotune measured cost explanations, winner selection evidence a oddeleni measured runtime costu od compile-time odhadu.

- [58-metal-full-backend-parity-closure.md](./58-metal-full-backend-parity-closure.md)
  - Status: `PLANNED`. Jeden velky closure plan pro dotazeni Metal backendu: zavrit vsechny aktualni CPU boundary zdroje, unsupported Metal rows, scoped dtype/layout omezeni, training/backward mezery, index/loss/scatter/conv-pool backward coverage, output-copy strategii a regression gates.

- [59-training-aware-metal-partition-closure.md](./59-training-aware-metal-partition-closure.md)
  - Status: `IMPLEMENTED`. Greedy Metal partitioning je phase-aware, trace rozlisuje phase/structural/lowerer rejection, SDPA Q/K/V a unmasked backward outGrad umi GPU-side layout legalization producenty a causal/masked SDPA backward je planner-truth odmítnuty jako `UNSUPPORTED_MASK_SEMANTICS`.

- [60-canonical-composite-tensor-dag.md](./60-canonical-composite-tensor-dag.md)
  - Status: `IMPLEMENTED`. Public `softmax`, `logSoftmax` a `scaledDotProductAttention` skladaji canonical primitive DAG, default pipeline attention DAG neprepisuje zpet na `SPECIAL` a starsi backend-special op test fixtures pouzivaji explicitni konstrukci pres `TensorPrimitiveBuilder`.

- [61-configuration-model-consolidation.md](./61-configuration-model-consolidation.md)
  - Status: `IMPLEMENTED_FOUNDATION`. Prime cilove rozdeleni konfigurace na `CompileConfig` a `RuntimeConfig`; compile vrstva samostatne vlastni `SemanticCanonicalizationConfig`, `GraphOptimizationConfig`, uzky `BackendPlanningConfig`, `RegionOptimizationConfig` a `MemoryPlanningConfig`; soucasti jsou defaultni profily, nove `GraphExecutionPolicy`/`ExecutionProfile`, compile-side autotune model, sjednoceny planning service a zachovani explicitniho backend planning kontraktu i pri `noGraphOptimization()`.

- [62-onnx-scatter-elements-and-scatternd-plan.md](./62-onnx-scatter-elements-and-scatternd-plan.md)
  - Status: `IMPLEMENTED`. `ScatterElements` je implementovany jako funkcionalni rank-preserving axis scatter s ONNX import/export, forward `none/add/mul/max/min` a backward `none/add`; `ScatterND` je implementovany pro inference jako tuple-index scatter s ONNX import/export. `ScatterND` backward ceka na `GatherND`.

- [63-onnx-inference-breadth-wave-2.md](./63-onnx-inference-breadth-wave-2.md)
  - Status: `IMPLEMENTED`. ONNX dense inference breadth pro `Pad`, `Split`, `Tile`, `ArgMax`, `ReduceProd` a `GlobalAveragePool` se statickymi parametry, coverage matrix rows a fixtures.

- [64-onnx-inference-breadth-wave-3.md](./64-onnx-inference-breadth-wave-3.md)
  - Status: `IMPLEMENTED`. ONNX staticke shape helpery, dalsi redukcni family, `CumSum` a `NonZero` audit bez zavadeni runtime dynamic-shape enginu.

- [65-onnx-wave-4-activation-roundtrip-model-harness.md](./65-onnx-wave-4-activation-roundtrip-model-harness.md)
  - Status: `IMPLEMENTED_FOUNDATION`. ONNX activation/math breadth, round-trip hardening, coverage report, activation export recognizery a checked-in compatibility fixture harness pro male model-like grafy.

- [66-onnx-interchange-closure-wave-6.md](./66-onnx-interchange-closure-wave-6.md)
  - Status: `IMPLEMENTED`. Dalsi ONNX closure vlna: composite reduction/global-pool export recognizery, sirsi static-param hardening, mini model suite expansion, coverage evidence upgrade, GPU parity audit bez overclaimu a workflow docs.

- [67-onnx-evidence-and-export-policy-closure.md](./67-onnx-evidence-and-export-policy-closure.md)
  - Status: `IMPLEMENTED`. ONNX evidence closure bez GPU scope: `ROUND_TRIP_TESTED` rows narostly na 70, `EXPLICITLY_CLASSIFIED` zustal jen `Constant`, canonical `Flatten` export je zpresneny pro reshape pattern a `OnnxLeafTensorPolicy` ma explicitni kontrakt testy.
