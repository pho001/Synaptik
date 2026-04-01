# Fused Operation Backend Refactor

## Stav

Dokonceno.

Aktualni implementovany stav:

- `operations.FusedOperation` je descriptor
- `operations.FusedOperationFactory` vyrabi descriptor z fused clusteru
- `graph.codegen.FusedExpressionPlan` je compiler IR
- `graph.codegen.CompiledFusedKernelFactory` vytvari compiled runtime executable
- `graph.codegen.FusedKernelGeneratorRouter` routuje F32/F64/F16 codegen
- `backend.kernels.cpu.CpuFusedKernel` vykonava prepared fused kernel z metadata

Zbytek dokumentu je zachovan jako navrhove rozhodnuti a historicky plan, ale vysledna architektura uz je v kodu.

## Cil

Presunout fused execution z dnesniho modelu:

- `operations.FusedOperation` = descriptor + compiler + class loader + runtime wrapper

na cilovy model:

- `operations.FusedOperation` = pouze descriptor fused uzlu
- `graph/codegen` a nova fusion compile vrstva = priprava compiled fused executable
- `CompiledGraph.prepare(...)` = misto, kde se fused executable vytvari a uklada do prepared metadata
- `backend/kernels/cpu` = pouze spusti prepared fused executable

Tento refaktor ma byt dokoncen tak, aby fused path respektovala stejnou architekturu jako bezne operace:

- `Tensor` drzi operation descriptor
- `CompiledGraph` drzi serazeny graph
- `PreparedExecution` drzi runtime-ready kroky
- backend kernel pouze provadi execution

## Proc je to potreba

Aktualni stav v [FusedOperation.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/operations/FusedOperation.java):

- constructor ma side effects
- pri vytvoreni operation descriptoru uz probiha ASM codegen
- dochazi k class loadingu a reflection instanciaci
- `FusedOperation` drzi `compiledInstance`
- `CpuFusedKernel` musi sahat zpet do operation vrstvy a vytahovat executable objekt

To je architektonicky spatne, protoze:

- descriptor a executable jsou pomichane
- `operations` vrstva znovu nese runtime/arithmetic responsibility
- prepared metadata neobsahuji vsechny runtime zavislosti fused node
- neni zde smysluplna cache compiled fused executable
- invalidace se neda navrhnout ciste

## Cilovy tvar

### 1. `operations.FusedOperation` zustane, ale jen jako descriptor

Ve [FusedOperation.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/operations/FusedOperation.java) ponechat pouze:

- `expression`
- `precisionMode` nebo `precisionHint`
- `lowCostHint`
- `schedulerSignature`
- `clusterSize`
- `externalInputCount`
- `dispatchComplexity`
- `dispatchScale`

Ponechat metadata API:

- `opType()`
- `isElementWise()`
- `getPreferredBackend()`
- `supportsBackend()`
- `getExpression()`
- gettery na fused metadata

Odstranit z `FusedOperation`:

- `CLASS_COUNTER`
- `compiledInstance`
- ASM codegen
- `CustomClassLoader`
- reflection konstruktor generated class
- compile profiler hook
- `apply(...)` delegaci na generated instance

Po refaktoru ma constructor `FusedOperation` pouze:

- validovat vstup
- spocitat descriptor metadata

## 2. Pridat fusion helper tridy

Vytvorit novy balicek:

- `src/main/java/graph/fusion/`

Do nej rozdelit dnesni statickou logiku z `FusedOperation`:

### `FusedExternalInputCollector`

Zodpovednost:

- najit external inputs fused clusteru v deterministickem poradi

API:

```java
public final class FusedExternalInputCollector {
    public static List<Tensor> collect(List<Tensor> cluster) { ... }
}
```

Presunout sem logiku z dnesni `findExternalInputs(...)`.

### `FusedPrecisionResolver`

Zodpovednost:

- rozhodnout precision mode pro fused executable

API:

```java
public final class FusedPrecisionResolver {
    public static int resolve(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) { ... }
}
```

Presunout sem logiku z `resolvePrecisionMode(...)`.

### `FusedCostModel`

Zodpovednost:

- low-cost hint
- dispatch complexity
- dispatch scale

API:

```java
public final class FusedCostModel {
    public static boolean resolveLowCostHint(List<Tensor> cluster) { ... }
    public static int estimateDispatchComplexity(List<Tensor> cluster) { ... }
    public static int resolveDispatchScale(int dispatchComplexity) { ... }
}
```

### `FusedSignatureBuilder`

Zodpovednost:

- deterministicka scheduler/compile signature fused clusteru

API:

```java
public final class FusedSignatureBuilder {
    public static String buildSchedulerSignature(List<Tensor> cluster, int precisionMode) { ... }
}
```

## 3. Zavest runtime executable kontrakt mimo `operations`

Soucasny [FusedCompiledOperation.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/operations/FusedCompiledOperation.java) nema byt v `operations`.

Vytvorit novy balicek:

- `src/main/java/backend/kernels/cpu/fused/`

A tam novy runtime kontrakt:

```java
public interface CompiledFusedKernel {
    void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    );

    default void applyRangeVector(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    ) {
        applyRangeScalar(inputs, out, startInclusive, endExclusive, options);
    }
}
```

`FusedExecutionOptions` muze zustat tam, kde je ted, ale dlouhodobe je cistsi presun do:

- `backend/kernels/cpu/fused/FusedExecutionOptions`

To neni nutne udelat v prvnim kroku.

## 4. Upravit `FusedOperationGenerator`

Generator nema generovat `Operation` implementaci.

Ma generovat compiled runtime executable.

### Co se ma zmenit

V [FusedOperationGenerator.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/graph/codegen/FusedOperationGenerator.java) a [HFusedOperationGenerator.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/graph/codegen/HFusedOperationGenerator.java):

- generovana trida ma implementovat `backend.kernels.cpu.fused.CompiledFusedKernel`
- nema implementovat `operations.Operation`
- nema obsahovat `opType()`, `getExpression()`, `supportsBackend()`, `isElementWise()`
- nema potrebovat cluster/runtime metadata v constructoru
- idealne ma byt bezstavova nebo mit minimalni constructor

To znamena:

- ponechat range execution methods
- odstranit operation-descriptor API z generated class

### Router

[FusedOperationGeneratorRouter.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/graph/codegen/FusedOperationGeneratorRouter.java) muze zustat jako dtype dispatch vrstva.

Jen zmenit semantiku:

- vystup = bytecode pro `CompiledFusedKernel`

ne:

- bytecode pro generated `Operation`

## 5. Pridat `FusedKernelCompiler`

Vytvorit novou sluzbu:

- `src/main/java/graph/codegen/FusedKernelCompiler.java`

Zodpovednost:

- dostat `FusedOperation` descriptor
- dostat cluster/root/external inputs
- vyrobit compiled executable
- resit compile cache
- pripadne profiler compile hook

Navrzene API:

```java
public final class FusedKernelCompiler {
    public CompiledFusedKernel compile(
            FusedOperation descriptor,
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputsInOrder
    ) { ... }
}
```

Alternativa:

```java
public final class FusedKernelCompiler {
    public CompiledFusedKernel compile(FusedKernelCompileRequest request) { ... }
}
```

To je cistsi, pokud bude request obsahovat vice compile-time detailu.

## 6. Zavest compile cache

Aktualne cache neexistuje. [FusedOperation.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/operations/FusedOperation.java) pri kazdem vytvoreni fused op generuje novou class.

Pridat compile cache do `FusedKernelCompiler`.

### Cache key

Minimum:

- scheduler signature
- precision mode

Lepší:

- scheduler signature
- precision mode
- version generatoru

Navrzene typy:

```java
public record FusedKernelCacheKey(
        String signature,
        int precisionMode
) {}
```

```java
private final ConcurrentHashMap<FusedKernelCacheKey, Class<? extends CompiledFusedKernel>> cache;
```

Nebo:

```java
private final ConcurrentHashMap<FusedKernelCacheKey, Constructor<? extends CompiledFusedKernel>> cache;
```

Pokud generated class nema stav, je mozne cacheovat i rovnou instanci.

### Doporučení

Cacheovat class nebo constructor handle, ne `Tensor`.

To je rychle a ciste:

- descriptor zustane immutable
- executable state bude sdilena mezi prepared plans
- `Tensor` nebude drzet runtime compiled state

## 7. Rozsirit prepared metadata

Soucasna prepared metadata pro node neumi drzet fused executable.

Rozsirit [CompiledNodeExecutionMetadata.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/graph/execution/CompiledNodeExecutionMetadata.java):

Varianta A:

```java
public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        CompiledFusedKernel fusedKernel
) { ... }
```

Varianta B:

```java
public sealed interface CompiledCpuExecutable permits CompiledFusedKernel {}
```

a metadata pak:

```java
public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        CompiledCpuExecutable cpuExecutable
) { ... }
```

Doporuceni:

- pro prvni krok pouzit jednodussi variantu A

## 8. Upravit `CompiledGraph.prepare(...)`

Tohle je nejdulezitejsi presun odpovednosti.

V [CompiledGraph.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/graph/CompiledGraph.java):

- rozpoznat `Operation.OpType.FUSED`
- pri prepare-time zavolat `FusedKernelCompiler`
- ulozit compiled fused executable do metadata

Tedy v `prepareMetadata(...)`:

1. resolve backend
2. resolve `CpuKernel`
3. pokud `operation.opType() == FUSED`
   - najit `FusedOperation descriptor`
   - zavolat `FusedKernelCompiler`
   - vratit metadata s `fusedKernel`
4. jinak vratit bezne metadata

To je spravne misto, protoze:

- je to presne prepare-time responsibility
- metadata uz z definice drzi runtime-ready stav
- sjednoti to fused path s ostatnimi uzly

## 9. Upravit `CpuFusedKernel`

V [CpuFusedKernel.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/backend/kernels/cpu/CpuFusedKernel.java):

Odstranit:

- `fused.getCompiledInstance()`
- fallback na `compiled.apply(...)`
- zavislost na generated `Operation`

Novy model:

- `op` je pouze `FusedOperation` descriptor
- executable prijde z prepared metadata
- kernel pouze rozhodne scalar/vector/parallel path a spusti executable

Pseudo:

```java
CompiledFusedKernel executable = requireFusedKernel(metadata);
ResolvedDispatchHints hints = requireDispatchHints(context);
FusedExecutionOptions options = ...
switch (mode) {
    case SCALAR -> executable.applyRangeScalar(...)
    case VECTOR -> executable.applyRangeVector(...)
    case PARALLEL -> ...
}
```

To je cisty executor model.

## 10. Profiling presunout z descriptoru do compileru/kernelu

Compile profiler hook dnes zije v `FusedOperation` constructoru.

To je spatne.

Spravne:

- compile profiling v `FusedKernelCompiler`
- runtime profiling v `CpuFusedKernel`

`FusedOperation` descriptor nema mit side effects.

## 11. Co menit v jakem poradi

Doporucene poradi kvuli bezpecnemu refaktoru:

### Faze 1: Extract without behavior change

1. vytvorit:
   - `FusedExternalInputCollector`
   - `FusedPrecisionResolver`
   - `FusedCostModel`
   - `FusedSignatureBuilder`
2. presunout logiku z `FusedOperation` do helperu
3. `FusedOperation` zatim muze stale drzet `compiledInstance`

Cil:

- zmensit tridu bez zmeny chovani

### Faze 2: Introduce compiled runtime artifact

4. pridat `backend.kernels.cpu.fused.CompiledFusedKernel`
5. upravit generator(y), aby generovaly tento kontrakt
6. pridat `FusedKernelCompiler`
7. pridat cache compileru

### Faze 3: Move compile to prepare phase

8. odstranit compile side effects z `FusedOperation`
9. rozsirit `CompiledNodeExecutionMetadata`
10. v `CompiledGraph.prepare(...)` generovat fused executable
11. upravit `CpuFusedKernel`, aby cetl executable z metadata

### Faze 4: Cleanup

12. smazat [FusedCompiledOperation.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/operations/FusedCompiledOperation.java) z `operations`
13. odstranit `CustomClassLoader` usage z operation vrstvy
14. upravit dokumentaci

## 12. Referencni implementace

Tato sekce schvalne obsahuje kompletni navrh klicovych trid a uprav. Neni to minimalni patch, ale citelny blueprint, podle ktereho lze refaktor provest.

### 12.1 `operations.FusedOperation`

```java
package operations;

import backend.ComputeBackend;
import graph.fusion.FusedCostModel;
import graph.fusion.FusedExternalInputCollector;
import graph.fusion.FusedPrecisionResolver;
import graph.fusion.FusedSignatureBuilder;
import tensor.Tensor;

import java.util.List;

public final class FusedOperation implements Operation {
    private final String expression;
    private final int precisionMode;
    private final boolean lowCostHint;
    private final String schedulerSignature;
    private final int clusterSize;
    private final int externalInputCount;
    private final int dispatchComplexity;
    private final int dispatchScale;

    public FusedOperation(List<Tensor> cluster, Tensor root) {
        this(cluster, root, FusedExternalInputCollector.collect(cluster));
    }

    public FusedOperation(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        if (cluster == null || cluster.isEmpty()) {
            throw new IllegalArgumentException("Fused cluster cannot be null/empty.");
        }
        if (root == null) {
            throw new IllegalArgumentException("Fused root cannot be null.");
        }

        this.expression = "fused(" + cluster.size() + ")";
        this.precisionMode = FusedPrecisionResolver.resolve(cluster, root, externalInputsInOrder);
        this.lowCostHint = FusedCostModel.resolveLowCostHint(cluster);
        this.schedulerSignature = FusedSignatureBuilder.buildSchedulerSignature(cluster, precisionMode);
        this.clusterSize = cluster.size();
        this.externalInputCount = externalInputsInOrder == null ? 0 : externalInputsInOrder.size();
        this.dispatchComplexity = FusedCostModel.estimateDispatchComplexity(cluster);
        this.dispatchScale = FusedCostModel.resolveDispatchScale(dispatchComplexity);
    }

    @Override
    public OpType opType() {
        return OpType.FUSED;
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return expression;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return false;
    }

    public int getPrecisionMode() {
        return precisionMode;
    }

    public boolean isLowCostHint() {
        return lowCostHint;
    }

    public String getSchedulerSignature() {
        return schedulerSignature;
    }

    public int getClusterSize() {
        return clusterSize;
    }

    public int getExternalInputCount() {
        return externalInputCount;
    }

    public int getDispatchComplexity() {
        return dispatchComplexity;
    }

    public int getDispatchScale() {
        return dispatchScale;
    }
}
```

### 12.2 `graph.fusion.FusedExternalInputCollector`

```java
package graph.fusion;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class FusedExternalInputCollector {
    private FusedExternalInputCollector() {}

    public static List<Tensor> collect(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return List.of();
        }

        Set<Tensor> clusterSet = new LinkedHashSet<>(cluster);
        Set<Tensor> external = new LinkedHashSet<>();

        for (Tensor tensor : cluster) {
            List<Tensor> parents = tensor.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor parent : parents) {
                if (!clusterSet.contains(parent)) {
                    external.add(parent);
                }
            }
        }

        return new ArrayList<>(external);
    }
}
```

### 12.3 `graph.fusion.FusedPrecisionResolver`

```java
package graph.fusion;

import graph.codegen.FusedDTypeOps;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class FusedPrecisionResolver {
    private FusedPrecisionResolver() {}

    public static int resolve(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        DataType target = root != null ? root.getDataType() : DataType.FLOAT64;
        if (target == null) {
            target = DataType.FLOAT64;
        }

        List<Tensor> all = new ArrayList<>();
        if (cluster != null) {
            all.addAll(cluster);
        }
        if (externalInputsInOrder != null) {
            all.addAll(externalInputsInOrder);
        }
        if (root != null) {
            all.add(root);
        }

        for (Tensor tensor : all) {
            if (tensor == null) {
                continue;
            }
            DataType dataType = tensor.getDataType();
            if (dataType == DataType.FLOAT64) {
                target = DataType.FLOAT64;
                break;
            }
            if (dataType == DataType.FLOAT32 && target == DataType.FLOAT16) {
                target = DataType.FLOAT32;
            }
        }

        return switch (target) {
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case FLOAT16 -> FusedDTypeOps.MODE_F16;
        };
    }
}
```

### 12.4 `graph.fusion.FusedCostModel`

```java
package graph.fusion;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class FusedCostModel {
    private FusedCostModel() {}

    public static boolean resolveLowCostHint(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return false;
        }

        for (Tensor tensor : cluster) {
            if (tensor == null || tensor.getOperation() == null) {
                continue;
            }
            Operation.OpType type = tensor.getOperation().opType();
            if (type == null) {
                return false;
            }
            switch (type) {
                case ADD, SUB, MUL, MIN, MAX, NEG, MUL_SCALAR, RELU, NOOP -> {
                }
                default -> {
                    return false;
                }
            }
        }
        return true;
    }

    public static int estimateDispatchComplexity(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return 1;
        }
        int total = 0;
        for (Tensor tensor : cluster) {
            if (tensor == null || tensor.getOperation() == null) {
                continue;
            }
            total += tensor.getOperation().isCheap() ? 1 : 4;
        }
        return Math.max(1, total);
    }

    public static int resolveDispatchScale(int dispatchComplexity) {
        int normalized = (Math.max(1, dispatchComplexity) + 7) / 8;
        return Math.max(1, Math.min(8, normalized));
    }
}
```

### 12.5 `graph.fusion.FusedSignatureBuilder`

```java
package graph.fusion;

import tensor.Tensor;

import java.util.List;

public final class FusedSignatureBuilder {
    private FusedSignatureBuilder() {}

    public static String buildSchedulerSignature(List<Tensor> cluster, int precisionMode) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("fused:pm=").append(precisionMode).append('|');
        if (cluster != null) {
            for (Tensor tensor : cluster) {
                if (tensor == null || tensor.getOperation() == null) {
                    continue;
                }
                sb.append(tensor.getOperation().opType()).append(',');
            }
        }
        return sb.toString();
    }
}
```

### 12.6 `backend.kernels.cpu.fused.CompiledFusedKernel`

```java
package backend.kernels.cpu.fused;

import operations.FusedExecutionOptions;
import tensor.Tensor;

import java.util.List;

public interface CompiledFusedKernel {
    void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    );

    default void applyRangeVector(
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    ) {
        applyRangeScalar(inputs, out, startInclusive, endExclusive, options);
    }
}
```

### 12.7 `graph.codegen.FusedKernelCacheKey`

```java
package graph.codegen;

public record FusedKernelCacheKey(
        String signature,
        int precisionMode
) {}
```

### 12.8 `graph.codegen.FusedKernelCompiler`

```java
package graph.codegen;

import backend.kernels.cpu.FusedExecutionProfiler;
import backend.kernels.cpu.fused.CompiledFusedKernel;
import operations.FusedOperation;
import tensor.Tensor;
import utils.CustomClassLoader;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public final class FusedKernelCompiler {
    private static final AtomicInteger CLASS_COUNTER = new AtomicInteger();

    private final ConcurrentHashMap<FusedKernelCacheKey, Constructor<? extends CompiledFusedKernel>> cache =
            new ConcurrentHashMap<>();

    public CompiledFusedKernel compile(
            FusedOperation descriptor,
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputsInOrder
    ) {
        if (descriptor == null) {
            throw new IllegalArgumentException("descriptor cannot be null");
        }

        FusedKernelCacheKey key = new FusedKernelCacheKey(
                descriptor.getSchedulerSignature(),
                descriptor.getPrecisionMode()
        );

        try {
            Constructor<? extends CompiledFusedKernel> ctor = cache.computeIfAbsent(
                    key,
                    ignored -> compileConstructor(descriptor, cluster, root, externalInputsInOrder)
            );
            return ctor.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to instantiate compiled fused kernel", e);
        }
    }

    @SuppressWarnings("unchecked")
    private Constructor<? extends CompiledFusedKernel> compileConstructor(
            FusedOperation descriptor,
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputsInOrder
    ) {
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        try {
            int id = CLASS_COUNTER.incrementAndGet();
            String binaryName = "backend.kernels.cpu.fused.GeneratedFusedKernel" + id;
            String internalName = binaryName.replace('.', '/');

            byte[] bytecode = FusedOperationGeneratorRouter.generate(
                    internalName,
                    cluster,
                    root,
                    externalInputsInOrder,
                    descriptor.getPrecisionMode()
            );

            CustomClassLoader loader = new CustomClassLoader();
            Class<?> generatedClass = loader.define(binaryName, bytecode);
            Constructor<?> ctor = generatedClass.getDeclaredConstructor();

            if (FusedExecutionProfiler.enabled()) {
                FusedExecutionProfiler.recordCompile(
                        descriptor.getSchedulerSignature(),
                        descriptor.getExpression(),
                        descriptor.getClusterSize(),
                        descriptor.getExternalInputCount(),
                        descriptor.getPrecisionMode(),
                        descriptor.isLowCostHint(),
                        System.nanoTime() - t0
                );
            }

            return (Constructor<? extends CompiledFusedKernel>) ctor;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to compile fused kernel constructor", e);
        }
    }
}
```

### 12.9 Uprava `FusedOperationGenerator`

Generator ma generovat tridu, ktera implementuje `backend.kernels.cpu.fused.CompiledFusedKernel`, ne `operations.Operation`.

Klicova zmena na urovni generatoru:

```java
cw.visit(
        V21,
        ACC_PUBLIC | ACC_FINAL | ACC_SUPER,
        internalClassName,
        null,
        "java/lang/Object",
        new String[]{"backend/kernels/cpu/fused/CompiledFusedKernel"}
);
```

Generated trida ma obsahovat:

- bezparametricky constructor
- `applyRangeScalar(...)`
- `applyRangeVector(...)`

Generated trida uz nema obsahovat:

- `apply(...)`
- `opType()`
- `isElementWise()`
- `supportsBackend()`
- `getExpression()`

### 12.10 Rozsireni `CompiledNodeExecutionMetadata`

```java
package graph.execution;

import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.fused.CompiledFusedKernel;

import java.util.Objects;

public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        CompiledFusedKernel fusedKernel
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
    }
}
```

### 12.11 Uprava `CompiledGraph.prepareMetadata(...)`

```java
private static final graph.codegen.FusedKernelCompiler FUSED_KERNEL_COMPILER =
        new graph.codegen.FusedKernelCompiler();

private CompiledNodeExecutionMetadata prepareMetadata(
        Tensor tensor,
        CpuExecutionPlanner planner,
        backend.runtime.RuntimeConfig runtimeConfig
) {
    ComputeBackend backend = tensor.resolveBackend();
    if (backend != ComputeBackend.CPU) {
        return new CompiledNodeExecutionMetadata(backend, null, null, null);
    }

    Operation operation = tensor.getOperation();
    CpuKernel kernel = CpuKernelRegistry.resolve(operation.opType());
    if (kernel == null) {
        throw new IllegalStateException("Missing CPU kernel for opType=" + operation.opType());
    }

    CpuNodeExecutionPlan cpuPlan = CPUBackend.buildExecutionPlan(
            operation,
            tensor.getPrevTensors(),
            tensor,
            planner,
            runtimeConfig.blasConfig()
    );

    CompiledFusedKernel fusedKernel = null;
    if (operation.opType() == Operation.OpType.FUSED) {
        FusedOperation fused = (FusedOperation) operation;
        List<Tensor> cluster = List.of();
        List<Tensor> externalInputs = graph.fusion.FusedExternalInputCollector.collect(cluster);
        fusedKernel = FUSED_KERNEL_COMPILER.compile(fused, cluster, tensor, externalInputs);
    }

    return new CompiledNodeExecutionMetadata(backend, kernel, cpuPlan, fusedKernel);
}
```

Poznamka:

V realne implementaci je potreba, aby optimizer pri vytvareni fused uzlu ulozil i cluster metadata tak, aby je slo v `prepare(...)` znovu pouzit. Bez toho nebude mozne compiler nakrmit puvodnim clusterem. Toto je nutny predpoklad refaktoru.

### 12.12 Uprava `CpuFusedKernel`

```java
package backend.kernels.cpu;

import backend.kernels.cpu.fused.CompiledFusedKernel;
import graph.codegen.FusedVectorOps;
import operations.FusedExecutionOptions;
import operations.FusedOperation;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
    @Override
    public CpuKernelCostClass costClass(Operation op) {
        if (op instanceof FusedOperation fused) {
            return fused.isLowCostHint() && fused.getDispatchScale() == 1
                    ? CpuKernelCostClass.LOW
                    : CpuKernelCostClass.MEDIUM;
        }
        return CpuKernel.super.costClass(op);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        FusedOperation fused = requireDescriptor(op);
        CompiledFusedKernel executable = requireExecutable(context);
        ResolvedDispatchHints hints = requireDispatchHints(context);

        FusedExecutionOptions options = new FusedExecutionOptions(
                context.useFastExpApprox(),
                context.useFastTanhApprox()
        );

        int length = node.getFlatDataSize();
        CpuExecutionMode mode = hints.mode();
        CpuKernelCostClass costClass = fused.isLowCostHint() && fused.getDispatchScale() == 1
                ? CpuKernelCostClass.LOW
                : CpuKernelCostClass.MEDIUM;
        String schedulerKey = fused.getSchedulerSignature();
        boolean recommendVector = FusedVectorOps.isRecommended(fused.getPrecisionMode());
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;

        switch (mode) {
            case SCALAR -> {
                executable.applyRangeScalar(inputs, node, 0, length, options);
                recordProfile(fused, mode, length, 1, false, false, t0);
            }
            case VECTOR -> {
                if (recommendVector) {
                    executable.applyRangeVector(inputs, node, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, true, t0);
                } else {
                    executable.applyRangeScalar(inputs, node, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, false, t0);
                }
            }
            case PARALLEL -> runParallel(
                    executable, inputs, node, hints, context.planner().lowCostNsPerElementThreshold(),
                    options, false, fused, mode, costClass, schedulerKey
            );
            case PARALLEL_VECTOR -> runParallel(
                    executable, inputs, node, hints, context.planner().lowCostNsPerElementThreshold(),
                    options, recommendVector, fused, mode, costClass, schedulerKey
            );
        }
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }

    private static void runParallel(
            CompiledFusedKernel executable,
            List<Tensor> inputs,
            Tensor node,
            ResolvedDispatchHints hints,
            double lowCostNsPerElementThreshold,
            FusedExecutionOptions options,
            boolean preferVector,
            FusedOperation fused,
            CpuExecutionMode mode,
            CpuKernelCostClass costClass,
            String schedulerKey
    ) {
        int length = node.getFlatDataSize();
        int chunkSize = preferVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (length + chunkSize - 1) / chunkSize;
        boolean useCommonPool = CpuSchedulerAdvisor.shouldUseCommonPool(
                costClass,
                schedulerKey,
                length,
                lowCostNsPerElementThreshold
        );
        long t0 = System.nanoTime();
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, length);
            if (preferVector) {
                executable.applyRangeVector(inputs, node, start, end, options);
            } else {
                executable.applyRangeScalar(inputs, node, start, end, options);
            }
        }, useCommonPool);
        long elapsed = System.nanoTime() - t0;
        CpuSchedulerAdvisor.recordSample(schedulerKey, length, elapsed);
        if (FusedExecutionProfiler.enabled()) {
            FusedExecutionProfiler.recordRun(
                    fused.getSchedulerSignature(),
                    mode,
                    length,
                    chunks,
                    useCommonPool,
                    preferVector,
                    elapsed
            );
        }
    }

    private static FusedOperation requireDescriptor(Operation op) {
        if (op instanceof FusedOperation fused) {
            return fused;
        }
        throw new IllegalStateException("CpuFusedKernel requires FusedOperation descriptor");
    }

    private static CompiledFusedKernel requireExecutable(CpuKernelContext context) {
        CompiledFusedKernel executable = context.executionMetadata().fusedKernel();
        if (executable == null) {
            throw new IllegalStateException("Missing compiled fused kernel in execution metadata");
        }
        return executable;
    }

    private static ResolvedDispatchHints requireDispatchHints(CpuKernelContext context) {
        ResolvedDispatchHints hints = context.dispatchHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedDispatchHints for fused execution");
        }
        return hints;
    }

    private static void recordProfile(
            FusedOperation fused,
            CpuExecutionMode mode,
            int length,
            int chunks,
            boolean useCommonPool,
            boolean preferVector,
            long startedNs
    ) {
        if (!FusedExecutionProfiler.enabled()) {
            return;
        }
        FusedExecutionProfiler.recordRun(
                fused.getSchedulerSignature(),
                mode,
                length,
                chunks,
                useCommonPool,
                preferVector,
                System.nanoTime() - startedNs
        );
    }
}
```

### 12.13 Duvod, proc je tento navrh spravny

Po teto zmene bude fused path konecne konzistentni se zbytkem systemu:

- `Tensor` drzi pouze operation descriptor
- `FusedOperation` je descriptor fused uzlu
- compiled fused executable vznikne v `prepare(...)`
- prepared metadata drzi runtime executable reference
- `CpuFusedKernel` funguje jako bezny backend executor

Toto je spravny model jak z hlediska architektury, tak z hlediska cache a invalidace.

## Minimalni pravidlo po refaktoru

Po dokonceni refaktoru musi platit:

- `operations.FusedOperation` neni compiler
- `operations.FusedOperation` neni runtime executable
- generated fused class neni `Operation`
- prepared metadata jsou jedine misto, kde zije runtime fused executable

## Stav po dokonceni

Po dokonceni bude fused path respektovat stejna pravidla jako normalni operace:

- `Tensor` drzi descriptor
- `CompiledGraph` drzi serazeny graph
- `prepare(...)` vytvari runtime executable metadata
- backend kernel jen spousti executable

To je cilovy architektonicky stav.
