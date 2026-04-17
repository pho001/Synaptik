# Synaptik

Synaptik je Java framework pro tensory, explicitní compiled-graph execution a reverse-mode autodiff. Není postavený jako "eager-only" knihovna ani jako benchmark-first experiment. Základní kontrakt je:

- veřejné `Tensor` API skládá graf
- `CompiledGraph` z něj vytvoří explicitní execution artifact
- optimizer provede čistě graph-level transformace
- `PreparedExecution` naváže runtime politiku a backend metadata
- backend spustí konkrétní CPU kernel path

Projekt dnes cílí primárně na CPU backend. GPU backendy existují jen jako scaffolding.

## Reading Guide

Pokud chceš pochopit projekt rychle:

1. začni v [src/main/java/tensor/README.md](src/main/java/tensor/README.md)
2. pak [src/main/java/operations/README.md](src/main/java/operations/README.md)
3. potom [src/main/java/graph/README.md](src/main/java/graph/README.md)
4. nakonec [src/main/java/backend/README.md](src/main/java/backend/README.md) a [src/main/java/tuning/README.md](src/main/java/tuning/README.md)

Pokud řešíš konkrétní problém:

- veřejné tensor API: [src/main/java/tensor/API.md](src/main/java/tensor/API.md)
- optimizer a rewrite/fusion pravidla: [src/main/java/graph/optimizer/README.md](src/main/java/graph/optimizer/README.md)
- tuning workflow, persistence a candidate search: [src/main/java/tuning/README.md](src/main/java/tuning/README.md)
- numerics drift a A/B porovnání: [src/main/java/numerics/README.md](src/main/java/numerics/README.md)

## Highlights

- explicitní tensor metadata: shape, strides, storage offset, dtype
- compiled/prepared execution pipeline místo implicitního runtime dispatch chaosu
- reverse-mode autodiff nad grafem složeným z `Tensor` operací
- optimizer stage model s rewrite, CSE, fusion a memory planning
- CPU backend s prepared metadata, dispatch hints a family-specific executory
- ASM fused backend pro hot elementwise fused subgrafy
- specializovaná primitiva pro strukturované kernel families
  - `LINEAR`
  - `SOFTMAX` / `LOG_SOFTMAX` a jejich gradienty
  - `SCALED_DOT_PRODUCT_ATTENTION` forward/backward
  - `CROSS_ENTROPY_LOSS_INDICES`
  - `CONV2D_GEMM`
- tuning stack pro benchmark, autotune a platform calibration

## Requirements

- JDK 25
- Gradle 9.4.1 kompatibilní prostředí nebo přibalený Gradle Wrapper
- macOS, Linux nebo Windows

Poznámka k Vector API:

- build i runtime používají `jdk.incubator.vector`
- Gradle wrapper přidává `--add-modules=jdk.incubator.vector` do compile/test/run tasků

## Build And Run

Základní příkazy:

- `./gradlew classes`
- `./gradlew test`
- `./gradlew run`

Na Windows použij `gradlew.bat`.

### Main CLI

Hlavní CLI entrypoint je [src/main/java/synaptik/app/Main.java](src/main/java/synaptik/app/Main.java).

Podporované flow:

```bash
./gradlew run --args="full f64"
./gradlew run --args="calibrate f64"
./gradlew run --args="autotune f64"
./gradlew run --args="benchmark-winner f64"
./gradlew run --args="benchmark-stage-space f64"
```

Význam:

- `full`
  - convenience flow `calibrate -> autotune -> benchmark-winner`
  - vhodný pro lokální iteraci
  - ne pro nejčistší performance čísla
- `calibrate`
  - hledá platform runtime defaults pro zvolený dtype/mode
- `autotune`
  - hledá graph-level winner pro workload `abc_sequence_matmul_*`
- `benchmark-winner`
  - porovnává baseline proti uloženému winner profilu
- `benchmark-stage-space`
  - benchmarkuje explicitní stage-order kandidáty

### Numerics CLI

Numerics harness běží přes [src/main/java/numerics/NumericsCli.java](src/main/java/numerics/NumericsCli.java).

Příklad:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR,CSE,FUSE \
  -Dnumerics.size=200000 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

## Quick Start

### 1. Jednoduchý forward/backward výpočet

```java
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import tensor.Tensor;

Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b");
a.setRequiresGrad(true);
b.setRequiresGrad(true);

Tensor y = a.add(b).mul(0.5).sum();

ExecutionProfile profile = new ExecutionProfile(
        "demo",
        "demo",
        y.getDataType(),
        ExecutionMode.FORWARD_BACKWARD,
        OptimizerConfig.trainingDefaults(),
        RuntimeConfig.trainingDefaults()
);

y.compute(profile);
System.out.println(y.scalarAsDouble());
System.out.println(java.util.Arrays.toString(a.getGradient().toDoubleArrayCopy()));
System.out.println(java.util.Arrays.toString(b.getGradient().toDoubleArrayCopy()));
```

Co se tady reálně stane:

1. `Tensor` API složí DAG z primitiv
2. `compute(profile)` interně zavolá `CompiledGraph.compile(...)`
3. optimizer aplikuje stage order z `OptimizerConfig`
4. `prepare(...)` předpočítá backend metadata a dispatch hints
5. `PreparedExecution` spustí forward nebo `FORWARD_BACKWARD`

### 2. Explicitní compile/prepare reuse

Tohle je správný pattern, když chceš měřit runtime bez opakovaného compile/prepare overheadu:

```java
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tensor.Tensor;

Tensor out = input.linear(weight, bias).relu().sum();
CompiledGraph graph = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults());
PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());

prepared.execute(ExecutionMode.FORWARD_BACKWARD);
prepared.execute(ExecutionMode.FORWARD_BACKWARD);
```

Použij to pro:

- benchmarky steady-state execution
- tracing hot paths
- opakované inference/training běhy nad stejným grafem

### 3. Broadcasting a select API

```java
Tensor scores = query.matmul(key.transpose());
Tensor masked = Tensor.where(mask, scores, Tensor.scalar(-1.0e9, scores.getDataType()));
Tensor probs = masked.softmax(1);
Tensor out = probs.matmul(value);
```

Důležité:

- `where` vyžaduje `BOOL` condition
- binary ops i compare ops používají standardní broadcasting
- autograd redukuje broadcasted gradient zpět do původních shape operandů

## Project Structure

- [src/main/java/tensor/](src/main/java/tensor/)
  - veřejná tensor surface, metadata, execution helpers, graph traversal, primitive builders
- [src/main/java/tensor/ops/](src/main/java/tensor/ops/)
  - tematicky rozdělené public graph builders
  - `binary`, `unary`, `compare`, `select`, `layout`, `linalg`, `conv`, `pool`, `reduction`, `loss`, `normalization`
- [src/main/java/tensor/options/](src/main/java/tensor/options/)
  - konfigurační helper records/enums pro veřejné higher-level ops
- [src/main/java/operations/](src/main/java/operations/)
  - canonical operation descriptors používané v grafu
- [src/main/java/graph/](src/main/java/graph/)
  - compile, prepare, run orchestrace
- [src/main/java/graph/optimizer/](src/main/java/graph/optimizer/)
  - rule pipeline, rewrite family, fusion support, memory planning
- [src/main/java/backend/](src/main/java/backend/)
  - runtime dispatch a backend-specific execution integration
- [src/main/java/backend/kernels/cpu/](src/main/java/backend/kernels/cpu/)
  - CPU kernel families
  - `elementwise`, `reduction`, `linalg`, `nn`, `index`, `layout`, `fused`, `grad`
- [src/main/java/tuning/](src/main/java/tuning/)
  - benchmark, autotune, platform calibration, reporting, persistence, search
- [src/main/java/numerics/](src/main/java/numerics/)
  - numerics A/B harness
- [src/test/java/](src/test/java/)
  - execution, regression, rewrite, tuning a benchmark kontrakty

## Core Architecture

### Tensor Layer

`Tensor` je veřejný graph node i runtime container. Nese:

- `Operation` descriptor
- seznam input tensorů
- storage/data views
- `requiresGrad`, `gradient`, backward marker
- shape/stride metadata

Veřejné API dnes primárně staví graf přes helper vrstvy v `tensor.ops.*`, ne přes ručně psaný kód přímo uvnitř `Tensor.java`.

### Operation Layer

`operations.*` nejsou backend kernels. Jsou to graph-level deskriptory.

Příklady:

- `add`, `mul`, `relu`
- `linear`
- `softmax`
- `scaledDotProductAttention`
- `crossEntropyLossIndices`

Stejný descriptor:

- definuje typ uzlu v grafu
- je vstupem pro optimizer
- je klíčem pro backend kernel resolution

### Graph Layer

`CompiledGraph` provádí:

1. topological closure nad forward grafem
2. build backward grafu, pokud existují trainable leaf inputs
3. optimizer stages podle `OptimizerConfig`
4. rozdělení na forward/backward section

`PreparedExecution` pak provádí:

- runtime-specific prepare
- build prepared metadata pro každý node
- vlastní execution loop
- optional traced run

### Backend Layer

CPU backend dnes používá prepared metadata místo opakovaného runtime rozhodování nad `Tensor`.

Prepare fáze řeší zejména:

- compute contract
- dtype conversion/materialization rozhodnutí
- dispatch hints
- reduction hints
- matmul hints
- fused executable preparation
- workspace allocation pro vybrané op families

### Optimizer Layer

Optimizer stage order je explicitní. Dnešní hlavní stage family:

- `AR`
  - composite rewrite family
- `CSE`
  - structural common subexpression elimination
- `FUSE`
  - elementwise fused cluster formation
- `MEM`
  - liveness-aware memory planning

Rewrite family dnes zahrnuje víc než jen algebraic cleanup. Obsahuje i lowering do specializovaných primitiv, například:

- `matmul + bias -> linear`
- `softmax` / `logSoftmax` backward pattern lowering
- attention forward/backward lowering
- cross-entropy-from-indices lowering
- volitelný piecewise import canonicalization
- `conv2d -> conv2dGemm` podle policy

## Execution Profiles, Calibration And Persistence

Runtime se neřídí jen jedním "optimizer profile" JSONem. Dnešní flow rozlišuje:

- built-in defaults v kódu
- platform runtime profile
  - výstup platform calibration
- graph-specific best profile
  - výstup autotune nad konkrétním workloadem
- finální `ExecutionProfile`
  - assembled runnable artifact

Preferovaný layout je:

- `profiles/platform/<platform-id>/calibration/...`
- `profiles/platform/<platform-id>/reports/...`
- `profiles/platform/<platform-id>/tuning/...`

Kompatibilní fallbacky do `build/...` stále existují, ale nejsou preferovaný long-term layout.

## Real Usage Patterns

### Kdy použít jen `Tensor.compute(profile)`

Použij pro:

- jednoduché integration testy
- sanity check lokálního grafu
- malé demo programy

Nepoužívej jako jediný benchmark harness, protože při každém běhu znovu provede compile/prepare.

### Kdy držet `CompiledGraph`

Použij pro:

- investigation optimizer výstupu
- compile trace / prepare trace
- opakované připravování stejného grafu pod různými runtime configy

### Kdy držet `PreparedExecution`

Použij pro:

- steady-state benchmark
- hot-path tracing
- výkonové experimenty nad stejným grafem a stejnou runtime policy

## Module Docs

- tensor: [src/main/java/tensor/README.md](src/main/java/tensor/README.md)
- tensor API: [src/main/java/tensor/API.md](src/main/java/tensor/API.md)
- operations: [src/main/java/operations/README.md](src/main/java/operations/README.md)
- graph: [src/main/java/graph/README.md](src/main/java/graph/README.md)
- optimizer: [src/main/java/graph/optimizer/README.md](src/main/java/graph/optimizer/README.md)
- backend: [src/main/java/backend/README.md](src/main/java/backend/README.md)
- tuning: [src/main/java/tuning/README.md](src/main/java/tuning/README.md)
- numerics: [src/main/java/numerics/README.md](src/main/java/numerics/README.md)

## Testing

Základní test flow:

- `./gradlew test`
- `./gradlew classes`

Typické oblasti pokrytí:

- execution correctness
- dtype coverage
- broadcast kontrakty
- rewrite/lowering correctness
- fused execution
- tuning/search/report contracts

## Development Notes

- fused kernels se dnes generují ASM codegenem během prepare fáze
- CPU je jediný plně implementovaný backend
- CUDA/OpenCL jsou scaffolding
- project počítá s JDK 25 a Vector API
- tuning docs jsou zdroj pravdy pro benchmark/autotune/calibration flow, ne starší benchmark-only utility vrstva

## License

V repozitáři zatím není licenční soubor. Před veřejnou distribucí je potřeba ho doplnit.
