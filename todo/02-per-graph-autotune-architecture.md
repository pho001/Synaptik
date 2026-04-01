# Per-Graph Autotune Architecture

## Stav a zamer

Synaptik ma implicitne:

- pokud uzivatel nic neuvede, vybrat defaultni execution profil podle:
  - architektury
  - dtype
  - rezimu (`FORWARD` / `FORWARD_BACKWARD`)

Soucasne chceme explicitne podporit:

- per-graph autotuning konkretniho vypoctu

Napriklad:

```java
AutotuneResult tuned = T7.compile().autotune();
T7.compute(tuned.profile());
```

nebo ekvivalentni pohodlnejsi API.

Tento dokument je zamerne navrzenejsi a mene finalni nez fused refaktor. Duvod:

- benchmark/autotune framework bude pravdepodobne zahozan
- chceme zachovat tuning knobs, ale prepsat workflow od nuly tak, aby se prizpusoboval `Tensor`, `CompiledGraph`, backendu a compileru, ne naopak

## Zakladni principy

### 1. Defaulty jsou automaticke

Pokud uzivatel nic neudela:

```java
T7.compute();
```

system ma:

1. odhadnout mode
2. resolve defaultni `ExecutionProfile`
3. spustit graph s timto profilem

### 2. Autotune je explicitni

Autotune neni implicitni side effect `compute()`.

Ma byt explicitni:

```java
AutotuneResult result = T7.compile().autotune();
```

To je dulezite, protoze autotune:

- je drahy
- meri vice kandidatu
- muze chtit persistovat vysledek
- je vázan na graph + mode + dtype + hardware

### 3. Autotune je vlastnost compiled graphu, ne tensoru

`Tensor` muze mit convenience wrapper:

```java
T7.autotune()
```

ale skutecne vlastnictvi ma byt na `CompiledGraph`.

Spravne jadro:

```java
CompiledGraph graph = T7.compile();
AutotuneResult result = graph.autotune();
```

## Cilovy model profilů

Uz mame:

- [ExecutionProfile.java](/Users/phujka/IdeaProjects/ComputationalGraph/src/main/java/config/profile/ExecutionProfile.java)

To je spravny zaklad.

Autotune ma vracet:

```java
public record AutotuneResult(
        ExecutionProfile profile,
        AutotuneSummary summary,
        boolean persisted
) {}
```

Minimalne:

- `profile`
- nejaka summary / score / candidate source
- informace, zda byl profil persistovan

## Default profile resolution

Potrebujeme sluzbu:

```java
public interface ExecutionProfileResolver {
    ExecutionProfile resolveDefaults(DataType dataType, ExecutionMode mode);
}
```

Prakticka implementace:

```java
public final class DefaultExecutionProfileResolver implements ExecutionProfileResolver
```

Priority resolveru:

1. explicitne predany profil
2. persisted per-graph tuned profil
3. persisted hardware bucket profil
4. arch default
5. built-in fallback

## Per-graph autotune

Autotune neni globalni benchmark mode. Ma to byt sluzba nad jednim compiled graphem.

### API navrh

```java
public interface GraphAutotuner {
    AutotuneResult autotune(CompiledGraph graph, AutotuneSpec spec);
}
```

`CompiledGraph` convenience API:

```java
public AutotuneResult autotune() { ... }
public AutotuneResult autotune(AutotuneSpec spec) { ... }
```

## `AutotuneSpec`

Potrebujeme lehky popis, jak agresivne a v jakem modu ladit.

```java
public record AutotuneSpec(
        ExecutionMode mode,
        DataType dataType,
        int warmupIters,
        int measureIters,
        int maxCandidates,
        boolean persistIfImproved
) {
    public static AutotuneSpec quickInference() { ... }
    public static AutotuneSpec quickTraining() { ... }
    public static AutotuneSpec thoroughInference() { ... }
    public static AutotuneSpec thoroughTraining() { ... }
}
```

## Co bude autotune ladit

Necele benchmark framework kandidaty jako dnes.

Ma ladit:

### 1. Compile-time knobs

- stage order
- `CSE` strict/aggressive
- `FUSE` config

### 2. Runtime knobs

- `CpuKernelConfig`
- `ApproximationConfig`
- `BlasConfig`

### 3. Fused compiler / backend knobs

Po refaktoru fused path:

- fused dispatch preferences
- vector/scalar preference policy
- scheduler thresholds
- pripadne backend-specific fused compile policy

## Co zahodit z dnesniho benchmark/autotune frameworku

Dnesni framework je vhodne zahodit jako workflow.

Ponechat dává smysl pouze:

- koncept tuning knobs
- cteni/zapis profilu
- nektere utility pro score/measurement, pokud jsou ciste oddelene

Zahodit:

- globalne orientovany candidate framework
- benchmark-first workflow
- konstrukce, kde backend/compiler/Tensor prizpusobujeme benchmark vrstve

Nova vrstva ma byt:

- graph-first
- profile-first
- backend-aware

## Navrhovana nova architektura

### `config.profile`

- `ExecutionProfile`

### `autotune`

Novy balicek:

- `GraphAutotuner`
- `AutotuneSpec`
- `AutotuneResult`
- `AutotuneCandidate`
- `AutotuneEvaluator`
- `AutotunePersistence`
- `AutotuneProfileStore`

### `graph`

`CompiledGraph` ma poskytovat:

- `autotune()`
- `autotune(spec)`
- `execute(profile)`

## Navrhovany tok

### Implicitni execution

```java
ExecutionProfile profile = resolver.resolveDefaults(dtype, mode);
CompiledGraph.compile(root, profile.optimizer())
        .prepare(profile.runtime())
        .execute(mode);
```

### Explicitni autotune

```java
CompiledGraph graph = CompiledGraph.compile(root, optimizerConfig);
AutotuneResult result = graph.autotune(AutotuneSpec.quickInference());
graph.prepare(result.profile().runtime()).execute(result.profile().mode());
```

### Komfortni API

```java
AutotuneResult tuned = T7.compile().autotune();
T7.compute(tuned.profile());
```

nebo:

```java
T7.compute();
T7.compute(profile);
```

## Persistovana data

Profil uz mame ve spravnem tvaru.

Dlouhodobe potrebujeme jeste per-graph identity:

```java
public record GraphProfileKey(
        String graphFingerprint,
        DataType dataType,
        ExecutionMode mode,
        String hardwareBucket
) {}
```

To umozni:

- ulozit tuned profil pro konkretni graph
- neplest ho s globalnim defaultem

## Co zatim neni dodefinovane

Tyto casti zustavaji zamerne otevrene:

- jak presne fingerprintovat graph
- jak reprezentovat candidate search space
- jak moc sdilet tuning mezi podobnymi graphy
- jak moc zapojit numerics postcheck
- jak sloucit globalni hardware defaults a per-graph tuned overrides

Tyto body maji byt doreseny az po:

- fused backend refaktoru
- cistém oddeleni compile/runtime metadata
- redukci benchmark-first zavislosti

## Minimalni smer implementace

Pokud bych mel definovat nejmensi rozumny prvni krok:

1. pridat `ExecutionProfileResolver`
2. pridat `AutotuneSpec`
3. pridat `AutotuneResult`
4. pridat `CompiledGraph.autotune(...)`
5. implementovat jednoduchy candidate search jen nad:
   - `OptimizerConfig`
   - `RuntimeConfig`
6. persistovat vysledny `ExecutionProfile`

Benchmark/autotune framework pak muze byt prepsan podle tohoto modelu, ne naopak.

## Pracovni zavěr

Autotune ma byt:

- explicitni
- per-graph
- profilovy
- backend/compiler aware

Nema byt:

- globalni benchmark workflow, kteremu se zbytek systemu prizpusobuje

To je aktualne nejrozumnejsi smer.

## Referencni implementace

Tato sekce obsahuje kompletni navrh klicovych API a trid. Neni to minimalni patch, ale konzistentni blueprint pro cisty prepis autotune vrstvy.

### 1. `tuning.profile.ExecutionProfileResolver`

```java
package tuning.profile;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;

public interface ExecutionProfileResolver {
    ExecutionProfile resolveDefaults(DataType dataType, ExecutionMode mode);

    default ExecutionProfile resolveDefaults(Tensor root, ExecutionMode mode) {
        if (root == null) {
            throw new IllegalArgumentException("root cannot be null");
        }
        return resolveDefaults(root.getDataType(), mode);
    }
}
```

### 2. `tuning.profile.DefaultExecutionProfileResolver`

```java
package tuning.profile;

import backend.runtime.ExecutionMode;
import benchmark.OptimizerProfileIO;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.nio.file.Path;

public final class DefaultExecutionProfileResolver implements ExecutionProfileResolver {
    private static final Path PROFILE_F32 = Path.of("config", "optimizer-profile-f32.json");
    private static final Path PROFILE_F64 = Path.of("config", "optimizer-profile-f64.json");

    @Override
    public ExecutionProfile resolveDefaults(DataType dataType, ExecutionMode mode) {
        ExecutionProfile builtIn = builtInDefaults(dataType, mode);
        Path path = profilePathFor(dataType);
        return OptimizerProfileIO.loadExecutionProfileOrDefault(path, builtIn);
    }

    private static Path profilePathFor(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> PROFILE_F64;
            case FLOAT32, FLOAT16 -> PROFILE_F32;
        };
    }

    private static ExecutionProfile builtInDefaults(DataType dataType, ExecutionMode mode) {
        OptimizerConfig optimizer = mode == ExecutionMode.FORWARD
                ? OptimizerConfig.inferenceDefaults()
                : OptimizerConfig.trainingDefaults();
        RuntimeConfig runtime = mode == ExecutionMode.FORWARD
                ? RuntimeConfig.inferenceDefaults()
                : RuntimeConfig.trainingDefaults();

        return new ExecutionProfile(
                "built-in-" + dataType.name().toLowerCase() + "-" + mode.name().toLowerCase(),
                "built-in",
                dataType,
                mode,
                optimizer,
                runtime
        );
    }
}
```

### 3. `tuning.autotune.AutotuneSpec`

```java
package tuning.autotune;

import backend.runtime.ExecutionMode;
import tensor.DataType;

public record AutotuneSpec(
        ExecutionMode mode,
        DataType dataType,
        int warmupIters,
        int measureIters,
        int maxCandidates,
        boolean persistIfImproved
) {
    public AutotuneSpec {
        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        warmupIters = Math.max(0, warmupIters);
        measureIters = Math.max(1, measureIters);
        maxCandidates = Math.max(1, maxCandidates);
    }

    public static AutotuneSpec quickInference(DataType dataType) {
        return new AutotuneSpec(ExecutionMode.FORWARD, dataType, 2, 5, 16, true);
    }

    public static AutotuneSpec quickTraining(DataType dataType) {
        return new AutotuneSpec(ExecutionMode.FORWARD_BACKWARD, dataType, 2, 5, 16, true);
    }

    public static AutotuneSpec thoroughInference(DataType dataType) {
        return new AutotuneSpec(ExecutionMode.FORWARD, dataType, 10, 25, 64, true);
    }

    public static AutotuneSpec thoroughTraining(DataType dataType) {
        return new AutotuneSpec(ExecutionMode.FORWARD_BACKWARD, dataType, 10, 25, 64, true);
    }
}
```

### 4. `tuning.autotune.AutotuneCandidate`

```java
package tuning.autotune;

import config.profile.ExecutionProfile;

public record AutotuneCandidate(
        String name,
        ExecutionProfile profile
) {
    public AutotuneCandidate {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name cannot be blank");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
    }
}
```

### 5. `tuning.autotune.AutotuneSummary`

```java
package tuning.autotune;

public record AutotuneSummary(
        int candidateCount,
        String winnerName,
        double winnerScoreMs,
        String reason
) {}
```

### 6. `tuning.autotune.AutotuneResult`

```java
package tuning.autotune;

import config.profile.ExecutionProfile;

public record AutotuneResult(
        ExecutionProfile profile,
        AutotuneSummary summary,
        boolean persisted
) {
    public AutotuneResult {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        if (summary == null) {
            throw new IllegalArgumentException("summary cannot be null");
        }
    }
}
```

### 7. `tuning.autotune.AutotuneProfileStore`

```java
package tuning.autotune;

import config.profile.ExecutionProfile;

public interface AutotuneProfileStore {
    ExecutionProfile load(GraphProfileKey key);
    void save(GraphProfileKey key, ExecutionProfile profile);
}
```

### 8. `tuning.autotune.GraphProfileKey`

```java
package tuning.autotune;

import backend.runtime.ExecutionMode;
import tensor.DataType;

public record GraphProfileKey(
        String graphFingerprint,
        DataType dataType,
        ExecutionMode mode,
        String hardwareBucket
) {
    public GraphProfileKey {
        if (graphFingerprint == null || graphFingerprint.isBlank()) {
            throw new IllegalArgumentException("graphFingerprint cannot be blank");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        if (hardwareBucket == null || hardwareBucket.isBlank()) {
            throw new IllegalArgumentException("hardwareBucket cannot be blank");
        }
    }
}
```

### 9. `tuning.autotune.GraphFingerprint`

```java
package tuning.autotune;

import tensor.Tensor;

public interface GraphFingerprint {
    String fingerprint(Tensor root);
}
```

Minimalni prvni implementace muze byt zalozena na topologickem poradi:

```java
package tuning.autotune;

import tensor.Tensor;

public final class StructuralGraphFingerprint implements GraphFingerprint {
    @Override
    public String fingerprint(Tensor root) {
        StringBuilder sb = new StringBuilder(1024);
        for (Tensor tensor : root.topologicalSort()) {
            if (tensor.getOperation() == null) {
                sb.append("leaf:");
            } else {
                sb.append(tensor.getOperation().opType()).append(':');
            }
            int[] shape = tensor.getShapeUnsafe();
            for (int dim : shape) {
                sb.append(dim).append('x');
            }
            sb.append('|');
        }
        return Integer.toHexString(sb.toString().hashCode());
    }
}
```

Toto neni kryptograficky fingerprint, ale jako prvni krok staci.

### 10. `tuning.autotune.AutotuneCandidateFactory`

```java
package tuning.autotune;

import backend.runtime.ExecutionMode;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

public final class AutotuneCandidateFactory {
    private AutotuneCandidateFactory() {}

    public static List<AutotuneCandidate> createBaseCandidates(
            RuntimeConfig defaultRuntime,
            DataType dataType,
            ExecutionMode mode
    ) {
        List<AutotuneCandidate> out = new ArrayList<>();

        out.add(candidate("NO_OPT", dataType, mode,
                OptimizerConfig.noOptimization(),
                defaultRuntime));

        out.add(candidate("TRAINING_DEFAULTS", dataType, mode,
                OptimizerConfig.trainingDefaults(),
                defaultRuntime));

        out.add(candidate("INFERENCE_DEFAULTS", dataType, mode,
                OptimizerConfig.inferenceDefaults(),
                defaultRuntime));

        out.add(candidate("INFERENCE_NO_FUSE", dataType, mode,
                new OptimizerConfig(
                        List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.MEM),
                        CseConfig.aggressiveDefaults(),
                        FuseConfig.inferenceDefaults()
                ),
                defaultRuntime));

        out.add(candidate("INFERENCE_SMALL_FUSE_CLUSTER", dataType, mode,
                OptimizerConfig.inferenceDefaults().withFuse(
                        FuseConfig.inferenceDefaults().withMaxClusterNodes(32)
                ),
                defaultRuntime));

        return out;
    }

    private static AutotuneCandidate candidate(
            String name,
            DataType dataType,
            ExecutionMode mode,
            OptimizerConfig optimizer,
            RuntimeConfig runtime
    ) {
        return new AutotuneCandidate(
                name,
                new ExecutionProfile(name, name, dataType, mode, optimizer, runtime)
        );
    }
}
```

### 11. `tuning.benchmark` bridge

Autotune nema znat nizsi measurement detaily. Potrebuje jen evaluator.

```java
package tuning.autotune;

import graph.CompiledGraph;

public interface AutotuneEvaluator {
    double evaluate(CompiledGraph graph, AutotuneCandidate candidate, AutotuneSpec spec);
}
```

Prvni implementace muze merit jednoduse:

```java
package tuning.autotune;

import graph.CompiledGraph;

public final class SimpleExecutionEvaluator implements AutotuneEvaluator {
    @Override
    public double evaluate(CompiledGraph graph, AutotuneCandidate candidate, AutotuneSpec spec) {
        long totalNs = 0L;
        for (int i = 0; i < spec.warmupIters(); i++) {
            graph.prepare(candidate.profile().runtime()).execute(spec.mode());
        }
        for (int i = 0; i < spec.measureIters(); i++) {
            long t0 = System.nanoTime();
            graph.prepare(candidate.profile().runtime()).execute(spec.mode());
            totalNs += (System.nanoTime() - t0);
        }
        return totalNs / 1_000_000.0 / spec.measureIters();
    }
}
```

Tohle je zamerne jednoduche. Pozdeji se to muze napojit na novou `tuning.benchmark.measure` vrstvu.

### 12. `tuning.autotune.GraphAutotuner`

```java
package tuning.autotune;

import graph.CompiledGraph;

public interface GraphAutotuner {
    AutotuneResult autotune(CompiledGraph graph, AutotuneSpec spec);
}
```

### 13. `tuning.autotune.DefaultGraphAutotuner`

```java
package tuning.autotune;

import benchmark.OptimizerProfileIO;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

public final class DefaultGraphAutotuner implements GraphAutotuner {
    private final ExecutionProfileResolver resolver;
    private final GraphFingerprint fingerprint;
    private final AutotuneEvaluator evaluator;
    private final AutotuneProfileStore store;
    private final HardwareBucketResolver hardwareBucketResolver;

    public DefaultGraphAutotuner(
            ExecutionProfileResolver resolver,
            GraphFingerprint fingerprint,
            AutotuneEvaluator evaluator,
            AutotuneProfileStore store,
            HardwareBucketResolver hardwareBucketResolver
    ) {
        this.resolver = resolver;
        this.fingerprint = fingerprint;
        this.evaluator = evaluator;
        this.store = store;
        this.hardwareBucketResolver = hardwareBucketResolver;
    }

    @Override
    public AutotuneResult autotune(CompiledGraph graph, AutotuneSpec spec) {
        String graphFingerprint = fingerprint.fingerprint(graph.getRootTensor());
        String hardwareBucket = hardwareBucketResolver.resolve();
        GraphProfileKey key = new GraphProfileKey(
                graphFingerprint,
                spec.dataType(),
                spec.mode(),
                hardwareBucket
        );

        ExecutionProfile defaultProfile = resolver.resolveDefaults(spec.dataType(), spec.mode());
        RuntimeConfig defaultRuntime = defaultProfile.runtime();
        List<AutotuneCandidate> candidates = AutotuneCandidateFactory.createBaseCandidates(
                defaultRuntime,
                spec.dataType(),
                spec.mode()
        );

        AutotuneCandidate winner = candidates.stream()
                .limit(spec.maxCandidates())
                .min(Comparator.comparingDouble(candidate -> evaluator.evaluate(
                        CompiledGraph.compile(graph.getRootTensor(), candidate.profile().optimizer()),
                        candidate,
                        spec
                )))
                .orElseThrow();

        double winnerScoreMs = evaluator.evaluate(
                CompiledGraph.compile(graph.getRootTensor(), winner.profile().optimizer()),
                winner,
                spec
        );

        boolean persisted = false;
        if (spec.persistIfImproved()) {
            store.save(key, winner.profile());
            persisted = true;
        }

        return new AutotuneResult(
                winner.profile(),
                new AutotuneSummary(candidates.size(), winner.name(), winnerScoreMs, "lowest measured mean runtime"),
                persisted
        );
    }
}
```

### 14. `tuning.autotune.HardwareBucketResolver`

```java
package tuning.autotune;

public interface HardwareBucketResolver {
    String resolve();
}
```

Jednoducha implementace:

```java
package tuning.autotune;

import benchmark.OptimizerProfileIO;

public final class DefaultHardwareBucketResolver implements HardwareBucketResolver {
    @Override
    public String resolve() {
        return OptimizerProfileIO.hardwareBucketKey();
    }
}
```

### 15. `CompiledGraph` facade

Navrhovane API:

```java
public AutotuneResult autotune() {
    backend.runtime.ExecutionMode mode = supportsBackward()
            ? backend.runtime.ExecutionMode.FORWARD_BACKWARD
            : backend.runtime.ExecutionMode.FORWARD;
    return autotune(supportsBackward()
            ? tuning.autotune.AutotuneSpec.quickTraining(getRootTensor().getDataType())
            : tuning.autotune.AutotuneSpec.quickInference(getRootTensor().getDataType()));
}

public AutotuneResult autotune(tuning.autotune.AutotuneSpec spec) {
    tuning.autotune.GraphAutotuner autotuner = new tuning.autotune.DefaultGraphAutotuner(
            new tuning.profile.DefaultExecutionProfileResolver(),
            new tuning.autotune.StructuralGraphFingerprint(),
            new tuning.autotune.SimpleExecutionEvaluator(),
            new tuning.autotune.FileAutotuneProfileStore(),
            new tuning.autotune.DefaultHardwareBucketResolver()
    );
    return autotuner.autotune(this, spec);
}

public void execute(config.profile.ExecutionProfile profile) {
    if (profile == null) {
        throw new IllegalArgumentException("profile cannot be null");
    }
    prepare(profile.runtime()).execute(profile.mode());
}
```

### 16. `Tensor` facade

Navrhovane convenience API:

```java
public CompiledGraph compile() {
    backend.runtime.ExecutionMode mode = getRequiresGrad()
            ? backend.runtime.ExecutionMode.FORWARD_BACKWARD
            : backend.runtime.ExecutionMode.FORWARD;
    config.profile.ExecutionProfile profile =
            new tuning.profile.DefaultExecutionProfileResolver().resolveDefaults(this, mode);
    return CompiledGraph.compile(this, profile.optimizer());
}

public void compute() {
    backend.runtime.ExecutionMode mode = getRequiresGrad()
            ? backend.runtime.ExecutionMode.FORWARD_BACKWARD
            : backend.runtime.ExecutionMode.FORWARD;
    config.profile.ExecutionProfile profile =
            new tuning.profile.DefaultExecutionProfileResolver().resolveDefaults(this, mode);
    CompiledGraph.compile(this, profile.optimizer()).execute(profile);
}

public void compute(config.profile.ExecutionProfile profile) {
    if (profile == null) {
        throw new IllegalArgumentException("profile cannot be null");
    }
    CompiledGraph.compile(this, profile.optimizer()).execute(profile);
}

public tuning.autotune.AutotuneResult autotune() {
    return compile().autotune();
}

public tuning.autotune.AutotuneResult autotune(tuning.autotune.AutotuneSpec spec) {
    return compile().autotune(spec);
}
```

### 17. `tuning.autotune.FileAutotuneProfileStore`

```java
package tuning.autotune;

import benchmark.OptimizerProfileIO;
import config.profile.ExecutionProfile;

import java.nio.file.Files;
import java.nio.file.Path;

public final class FileAutotuneProfileStore implements AutotuneProfileStore {
    private final Path root = Path.of("build", "autotune-profiles");

    @Override
    public ExecutionProfile load(GraphProfileKey key) {
        Path path = pathFor(key);
        if (!Files.exists(path)) {
            return null;
        }
        return OptimizerProfileIO.loadExecutionProfileOrDefault(path, null);
    }

    @Override
    public void save(GraphProfileKey key, ExecutionProfile profile) {
        OptimizerProfileIO.saveExecutionProfile(pathFor(key), profile);
    }

    private Path pathFor(GraphProfileKey key) {
        return root.resolve(
                key.mode().name().toLowerCase()
                        + "-" + key.dataType().name().toLowerCase()
                        + "-" + key.graphFingerprint()
                        + ".json"
        );
    }
}
```

### 18. Proc je tento navrh rozumny

Tento model dava Synaptiku:

- implicitni defaulty podle architektury a rezimu
- explicitni per-graph autotune
- ciste oddeleni config DTO od workflow
- graph-first architekturu
- moznost pozdeji napojit novou `tuning/benchmark` vrstvu misto dnesniho benchmark-first frameworku

Hlavni vyhoda:

- `Tensor`, `CompiledGraph`, backend i compiler zustavaji primarni architekturou systemu
- autotune je nadstavba, ne ridici vrstva celeho systemu
