# Optimizer

Optimizer je čistě graph-level vrstva. Transformuje topologicky seřazený graph před runtime prepare fází. Nevykonává kernels a nerozhoduje hot-path dispatch v okamžiku běhu.

Jeho kontrakt je jednoduchý:

- vstup: `List<Tensor>` v topological order
- výstup: sémanticky ekvivalentní `List<Tensor>` stále v topological order

## Reading Guide

Tento dokument je pro tebe, pokud řešíš:

- kdy přidat nový operation descriptor a kdy jen rewrite
- jak dnes vypadá `AR` stage family
- jaké patterny se lowerují do specializovaných primitiv
- kde jsou hranice CSE / FUSE / MEM
- jak neporušit forward/backward correctness

Související dokumentace:

- graph lifecycle: [../README.md](../README.md)
- operation descriptors: [../../operations/README.md](../../operations/README.md)
- backend families: [../../backend/README.md](../../backend/README.md)
- tuning/autotune: [../../tuning/README.md](../../tuning/README.md)

## Main Components

- orchestrace
  - [GraphOptimizer.java](../../graph/optimizer/GraphOptimizer.java)
  - [OptimizationRule.java](../../graph/optimizer/OptimizationRule.java)
  - [OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java)
- shared graph rewrite support
  - [OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java)
- top-level stages
  - [rewrite/RewriteRule.java](../../graph/optimizer/rewrite/RewriteRule.java)
  - [rules/CommonSubexpressionEliminationRule.java](../../graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
  - [rules/FuseElementWiseRule.java](../../graph/optimizer/rules/FuseElementWiseRule.java)
  - [rules/MemoryOptimizerRule.java](../../graph/optimizer/rules/MemoryOptimizerRule.java)
- fusion support
  - [fusion/FusedCostModel.java](../../graph/optimizer/fusion/FusedCostModel.java)
  - [fusion/FusedExternalInputCollector.java](../../graph/optimizer/fusion/FusedExternalInputCollector.java)
  - [fusion/FusedPrecisionResolver.java](../../graph/optimizer/fusion/FusedPrecisionResolver.java)
  - [fusion/FusedSignatureBuilder.java](../../graph/optimizer/fusion/FusedSignatureBuilder.java)

## Stage Model

Veřejný optimizer stage order dnes používá:

- `AR`
- `CSE`
- `FUSE`
- `MEM`

Mapování na implementace je centralizované v [OptimizerFactory.java](../../graph/optimizer/OptimizerFactory.java).

Výchozí preset reality:

- `OptimizerConfig.noOptimization()`
  - žádný stage
- `OptimizerConfig.trainingDefaults()`
  - `AR -> CSE -> MEM`
- `OptimizerConfig.inferenceDefaults()`
  - `AR -> CSE -> FUSE -> MEM`

To je důležité:

- training default dnes standardně nezapíná `FUSE`
- inference default ho zapíná

## Core Design Rule

Optimizer nesmí být "druhá runtime vrstva". Co má zůstat runtime rozhodnutí:

- scalar/vector/parallel dispatch
- BLAS vs Java path u konkrétní prepared matmul recipe
- chunk sizing
- approximation policy

Co naopak patří do optimizeru:

- algebraic cleanup
- lowering do specializovaných graph primitiv
- structural CSE
- fusion cluster formation
- memory planning

## Rule Contract

Každé pravidlo musí zachovat:

- dependency ordering
- reachability od sinků
- forward/backward phase boundaries
- dtype a shape semantiku
- gradient correctness

Pravidlo smí:

- nahradit node jiným node
- přepsat input edge
- znovu sestavit topological closure ze zachovaných sinků

K tomu slouží hlavně [OptimizerGraphSupport.java](../../graph/optimizer/OptimizerGraphSupport.java).

## `AR`: Rewrite Family

`AR` není jedna malá algebraic pass. Je to composite rewrite stage.

Dnešní delegate order v [rewrite/RewriteRule.java](../../graph/optimizer/rewrite/RewriteRule.java) je:

1. volitelný `PiecewiseLoweringRewrite`
2. `AlgebraicRewrite`
3. `LinearLoweringRewrite`
4. `LossLoweringRewrite`
5. `ReductionLoweringRewrite`
6. `AttentionLoweringRewrite`
7. `AttentionBackwardLoweringRewrite`
8. volitelný `Conv2dLoweringRewrite`

Tahle order není náhodná:

- canonicalization/import cleanup běží před ostatními specializacemi
- algebraic cleanup nejdřív zjednoduší lokální tvar grafu
- strukturální lowering na specializovaná primitiva běží potom
- `conv2d` lowering zůstává explicitně policy-controlled

## `PiecewiseLoweringRewrite`

Tahle pass je dnes schválně opt-in. Slouží hlavně jako repair/canonicalization vrstva pro:

- importované grafy
- ručně rozpadlé patterny

Nepředpokládá se, že interní `Tensor` builders na ni budou spoléhat pro běžný forward graph.

Aktuálně umí rozpoznat:

- canonical sigmoid
  - `1 / (1 + exp(-x)) -> sigmoid(x)`
- relu-like `where`
  - `where(x > 0, x, 0) -> relu(x)`
- clamp-like `where`
  - `where(x < t, t, x) -> clampMin(t)`
  - `where(x > t, t, x) -> clampMax(t)`

Config:

- [PiecewiseLoweringConfig.java](../../config/optimizer/PiecewiseLoweringConfig.java)

Default:

- všechno vypnuté

To je důležité i dokumentačně:

- pokud `Tensor.relu()` už vytváří `relu` primitivum, rewrite nic nedělá
- její role je canonicalize/import cleanup, ne normální forward construction

## `AlgebraicRewrite`

Sem patří lokální numerické zjednodušení. Je to úmyslně užší vrstva než "jakýkoli sémantický lowering".

Typické příklady:

- identity elimination
- scalar canonicalization
- lokální constant folding, kde je bezpečný
- přepisy typu `pow(x, 2) -> x * x`

Naopak sem dnes nepatří:

- attention pattern recognition
- softmax backward lowering
- cross-entropy lowering
- view/access rewrite

## `LinearLoweringRewrite`

Rozpoznává pattern:

- `matmul(input, weight) + bias`

a nahrazuje ho:

- `LINEAR(input, weight, bias)`

Podmínky jsou čistě shape/semantics based:

- `weight` musí odpovídat lineární vrstvě
- `bias` musí být 1D bias vector
- výstupní shape musí odpovídat batch prefixu vstupu + `outFeatures`

Smysl:

- backend dostane explicitní structured primitive
- bias epilog a packed weights mohou žít v jedné family
- nemusí se znovu hádat pattern až v runtime

## `LossLoweringRewrite`

Tohle je dnes jedna z nejdůležitějších rewrite families, protože nahrazuje reálně používané loss patterny specializovanými primitivy.

Aktuálně loweruje:

- forward cross-entropy-from-indices pattern do `CROSS_ENTROPY_LOSS_INDICES`
- backward pattern do `CROSS_ENTROPY_LOSS_INDICES_GRAD`

Rozpoznávaný forward tvar je zhruba:

- `neg(gather(logSoftmax(logits), targetIndices))`
- případně následovaný `sum()` nebo `mean()`

Backward tvar rozpoznává rozpadlý softmax/scatter-based gradient pattern a nahrazuje ho specializovaným grad primitive.

To je přesně správná vrstva pro takový přepis:

- je to graph semantics problém
- ne backend runtime heuristika
- backend pak může mít výrazně čistší specializovaný kernel family

## `ReductionLoweringRewrite`

Tahle pass loweruje backward patterny pro structured reduction families.

Aktuálně:

- softmax backward pattern -> `SOFTMAX_GRAD`
- log-softmax backward pattern -> `LOG_SOFTMAX_GRAD`

Rozpoznává se backward graph tvar, ne forward API call.

To je důležité:

- veřejná tensor surface může gradient stále skládat přes tensor ops
- optimizer ho později může přepsat na specializované primitivum

Tím zůstává:

- veřejné API čisté
- backend rychlý

## `AttentionLoweringRewrite`

Rozpoznává forward scaled dot-product attention pattern:

- `scores = q.matmul(k^T)`
- optional scale přes `mulScalar`
- optional mask přes `where(mask, scores, fill)`
- `softmax(scores)`
- `softmax(scores).matmul(v)`

Pokud pattern sedí, přepíše ho na:

- `SCALED_DOT_PRODUCT_ATTENTION`

Mask fill scalar je ověřovaný podle dtype. Rewrite není obecné "snaž se uhodnout attention za každou cenu". Je to poměrně úzká a kontrolovaná pattern detekce.

## `AttentionBackwardLoweringRewrite`

Tahle pass se dívá do backward části graphu a nahrazuje rozpadlé backward patterny specializovaným primitive:

- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`

Umí rozpoznat gradient paths pro:

- query
- key
- value

Používá přitom index nad forward attention nodes, aby backward pattern nespárovala špatně.

To je podstatné:

- nejde o lokální rewrite na jednom uzlu
- je to structured backward lowering opřený o znalost forward primitive

## `Conv2dLoweringRewrite`

`conv2d` lowering není vždy zapnutý. Je řízený explicitní policy:

- `OFF`
- `HEURISTIC`
- `ALWAYS`

Rewrite převádí:

- `CONV2D`

na:

- `CONV2D_GEMM`

pokud to dovolí policy.

Důležitá hranice:

- compile-time rewrite rozhodne, jestli graf bude mít direct nebo GEMM lowered conv primitive
- runtime pak ještě pořád řeší konkrétní backend compute detail uvnitř zvolené family

## `CSE`: Common Subexpression Elimination

`CommonSubexpressionEliminationRule` nepoužívá naivní string comparison. Pracuje se structural signatures.

Signatura zohledňuje:

- `Operation.OpType`
- forward/backward fázi
- input signatures
- explicitní operation parametry
- safety metadata

To je důležité třeba pro:

- `sum(axis, keepDims=false)` vs `sum(axis, keepDims=true)`
- různé `permute(...)`
- `pow` s různými exponenty
- scalar-parameter ops

`noop` a fused nodes zůstávají záměrně CSE boundaries.

## `FUSE`: Elementwise Fusion

`FuseElementWiseRule` vytváří fused clusters jen tam, kde dává smysl model:

- jedna output-space loop
- lokální per-element compute

Do fused compute algebra dnes patří:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

Nepatří tam:

- indexing
- reductions
- matmul
- structured losses
- special gradient kernels

View/access ops se nepovažují za compute nodes. Mohou se absorbovat jako external input access metadata.

## `MEM`: Memory Planning

`MemoryOptimizerRule` je compile-time planner, ne runtime allocator.

Jeho role:

- analyzovat liveness
- přiřadit reusable slots
- snížit peak memory footprint
- vracet explain/summary data

Policy jde přes:

- [MemoryConfig.java](../../config/optimizer/MemoryConfig.java)
- [MemoryPlannerPolicy.java](../../graph/optimizer/memory/MemoryPlannerPolicy.java)

Typické knoby:

- oddělené forward/backward pools
- cross-phase reuse
- larger-buffer reuse
- minimum reusable buffer size

## Example: Forward Lowering

```java
Tensor logits = x.linear(w, b);
Tensor loss = logits.crossEntropyLossIndices(targets, 1);
```

V ideálním runtime graphu po `AR` už můžeš mít:

- `LINEAR`
- `CROSS_ENTROPY_LOSS_INDICES`

místo rozpadlé kombinace:

- `MATMUL`
- `ADD`
- `LOG_SOFTMAX`
- `GATHER`
- `NEG`
- `MEAN`

## Example: Backward Lowering

Veřejný autograd builder může složit backward přes běžné tensor operace. Po `AR` se ale může přepsat na:

- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
- `CROSS_ENTROPY_LOSS_INDICES_GRAD`

To je klíčový design pattern projektu:

- forward/backward formulas se mohou skládat čistě přes `Tensor` operace
- optimizer je pak může nahradit strukturovanými primitivy

## Config Surface

Primární veřejný optimizer config je:

- [OptimizerConfig.java](../../config/optimizer/OptimizerConfig.java)

Obsahuje:

- `stageOrder`
- `rewrite`
- `cse`
- `fuse`
- `memory`

Z toho plyne důležitá zásada:

- tuning nesmí vymýšlet druhý skrytý optimizer config model
- vše, co má být graph policy, musí být vyjádřitelné přes `OptimizerConfig`

## Adding A New Rewrite

Správný postup:

1. rozhodni, jestli vůbec má vzniknout nový rewrite
   - není to jen práce pro nový operation descriptor?
   - není to spíš runtime/backend knob?
2. pokud je to rewrite:
   - umísti ji do `graph.optimizer.rewrite`, pokud patří do `AR` family
   - nebo do `graph.optimizer.rules`, pokud jde o samostatnou top-level stage
3. implementuj `OptimizationRule`
4. používej `OptimizerGraphSupport` pro edge rewrite a closure rebuild
5. registruj ji v `RewriteRule` nebo `OptimizerFactory`
6. přidej testy pro:
   - forward correctness
   - backward correctness
   - dtype coverage
   - broadcast/layout invariants

## When Not To Add A Rewrite

Rewrite nepřidávej, pokud:

- veřejný `Tensor` builder má rovnou vytvářet správné primitivum
- jde jen o backend-specific dispatch rozhodnutí
- jde o tuning knob, ne graph transformaci
- pattern je benchmark-only syntetika bez reálného graph významu

## Common Mistakes

- míchat graph policy s runtime policy
- lowerovat pattern, který má být rovnou canonical primitive ve veřejném API
- ignorovat backward section a přepsat jen forward tvar
- dělat rewrite jen podle labelu node místo `Operation.OpType` a parametrů
- spoléhat na rewrite jako opravu interní nekonzistence builderů

## Related Modules

- graph lifecycle: [../README.md](../README.md)
- operations: [../../operations/README.md](../../operations/README.md)
- backend: [../../backend/README.md](../../backend/README.md)
- tuning: [../../tuning/README.md](../../tuning/README.md)
