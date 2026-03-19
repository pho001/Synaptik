# Optimizer (Graph/optimizer)

## Cíl

- Transformovat topologicky seřazený graf Tensor uzlů před execution.
- Snížit počet uzlů, odstranit redundantní výrazy a zlepšit locality/throughput.
- Zachovat korektnost forward i backward části grafu.

## Hlavní části

- Orchestrátor:
  - [src/Graph/optimizer/GraphOptimizer.java](src/Graph/optimizer/GraphOptimizer.java)
  - [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java)
  - [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java)
- Pravidla:
  - [src/Graph/optimizer/rules/AlgebraicRewritingRule.java](src/Graph/optimizer/rules/AlgebraicRewritingRule.java)
  - [src/Graph/optimizer/rules/CommonSubexpressionEliminationRule.java](src/Graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
  - [src/Graph/optimizer/rules/FuseElementWiseRule.java](src/Graph/optimizer/rules/FuseElementWiseRule.java)
  - [src/Graph/optimizer/rules/MemoryOptimizerRule.java](src/Graph/optimizer/rules/MemoryOptimizerRule.java)
- Fused codegen:
  - [src/Graph/codegen/DFusedOperationGenerator.java](src/Graph/codegen/DFusedOperationGenerator.java)
  - [src/Operations/FusedOperation.java](src/Operations/FusedOperation.java)

## Datový tok

1) `Tensor.compute(...)` vytvoří `CompiledGraph`.
2) `CompiledGraph.compile()` sestaví `finalGraph` a zavolá `optimizer.optimize(...)`.
3) Pravidla běží sekvenčně nad topologicky seřazeným seznamem uzlů.
4) Po optimalizaci se pro compiled graph předvyřeší execution metadata (backend + CPU kernel cache).
5) `CompiledGraph.execute()` běží nad optimalizovaným pořadím.

Soubory:
- [src/Tensor/Tensor.java](src/Tensor/Tensor.java)
- [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java)

## API kontrakt pravidla

Rozhraní pravidla je:
- [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java)

Pravidlo:
- přijme `List<Tensor>` (topologicky seřazený graf),
- vrátí nový `List<Tensor>` odpovídající transformovanému grafu.

Pravidla musí:
- zachovat závislosti a pořadí výpočtu,
- nepokazit gradient flow,
- vracet konzistentní topologické pořadí.

## Stručně k pravidlům

- `AlgebraicRewritingRule`
  - lokální algebraické simplifikace výrazů.
- `CommonSubexpressionEliminationRule`
  - slučování ekvivalentních podvýrazů (volitelně strict safety).
- `FuseElementWiseRule`
  - seskupení element-wise uzlů do fused clusteru podle cost modelu.
  - respektuje hranice fází (forward/backward), materialization body a shared-expensive politiku.
- `MemoryOptimizerRule`
  - úpravy pro lepší memory behavior/reuse.

## Fused operace

`FuseElementWiseRule` může nahradit cluster jedním `FusedOperation` uzlem.

- Fused uzel má `OpType.FUSED`.
- Runtime execution jde přes `CpuFusedKernel`, který volá `op.apply(...)`.
- Bytecode fused `apply` je generován ASM generátorem.

Soubory:
- [src/Graph/optimizer/rules/FuseElementWiseRule.java](src/Graph/optimizer/rules/FuseElementWiseRule.java)
- [src/Backend/kernels/cpu/CpuFusedKernel.java](src/Backend/kernels/cpu/CpuFusedKernel.java)
- [src/Graph/codegen/DFusedOperationGenerator.java](src/Graph/codegen/DFusedOperationGenerator.java)

## Napojení na benchmark/autotune

Optimizer stage order i tuning knobs řídí benchmark framework:
- [src/Benchmark/OptimizerBenchmarkFramework.java](src/Benchmark/OptimizerBenchmarkFramework.java)
- [src/Benchmark/OptimizerCandidateFactory.java](src/Benchmark/OptimizerCandidateFactory.java)
- [src/Benchmark/TuningKnobs.java](src/Benchmark/TuningKnobs.java)

Autotune je 2-fázový:
- Phase 1: hrubý screening kandidátů.
- Phase 2: přesné přeměření finalistů.

Vítězné profily se persistují:
- `config/optimizer-profile.json` (runtime training profil),
- `build/optimizer-autotune/best-profile-training.json`,
- `build/optimizer-autotune/best-profile-inference.json`.

`RECOMMENDED` kandidát je profilově přepisován vítězem autotune training profilu.

## Jak přidat nové pravidlo

1) Přidej třídu do `src/Graph/optimizer/rules/` implementující `OptimizationRule`.
2) Implementuj transformaci `List<Tensor> -> List<Tensor>`.
3) Zaregistruj pravidlo v `OptimizerFactory` nebo přímo při skládání `GraphOptimizer`.
4) Ověř korektnost:
- numerická shoda proti baseline,
- zachování gradientů,
- regresní test pro edge cases.

## Poznámky k build/runtime

- Projekt používá ASM (`org.ow2.asm`).
- CPU vector path používá `jdk.incubator.vector`.
- Při lokálním spuštění benchmarku bez Gradle je nutné mít ASM na classpath.
