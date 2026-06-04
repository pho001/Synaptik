# 116. cpu1 Fused Elementwise Implementation Plan

## Stav Implementace

Status: `PLANNED`

Tento dokument je zivy implementacni checklist. Pri implementaci se ma menit stav jednotlivych tasku:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` hotovo a overeno

Aktualni stav:

- [x] Faze 0: overeni vstupnich hranic a ochrana pracovniho stromu
- [ ] Faze 1: cpu1 fused IR
- [ ] Faze 2: prepare-time fused plan a dispatch decision
- [ ] Faze 3: cpu1 fused executable unit v `backend.cpu1.exec`
- [ ] Faze 4: scalar/parallel fused runner pro JAVA_ARRAY a MEMORY_SEGMENT
- [ ] Faze 5: trace a prepared artifact integrace
- [ ] Faze 6: prepare dispatcher integrace a runtime config route
- [ ] Faze 7: test coverage
- [ ] Faze 8: scratch buffer pro fused temporaries
- [ ] Faze 9: fused vector/codegen hot path
- [ ] Faze 10: profile IO a tuning knobs
- [ ] Faze 11: parity benchmarky proti staremu CPU fused
- [ ] Faze 12: finalni overeni a odstraneni mezistavu

## Cil

Prenest fused elementwise execution koncept ze stareho `backend.cpu` do `backend.cpu1` tak, aby zapadl do stavajici cpu1 architektury:

```text
graph/region:
  vytvori ExecutionUnitKind.FUSED_ELEMENTWISE

prepare:
  z lowered fused unit vytvori Cpu1PreparedFusedElementwiseUnit
  rozhodne storage kind, layout/access model, launch policy a cost class

execute:
  Cpu1FusedElementwiseExecutableUnit binduje Cpu1TensorView vstupy/vystup
  runner projede prepared fused expression bez Tensor/autograd/stareho CpuKernelContext
```

Cilem neni 1:1 presun stareho CPU fused runtime. Cilem je prenest uzitecne casti:

- fused IR shape
- canonicalizaci
- broadcast/effective-stride pripravu
- seznam podporovanych fusable op semantik
- cost classification myslenku

a nahradit nebo zahodit casti, ktere jsou svazane se starym backendem:

- `PreparedFusedExecutable`
- `CpuKernelContext`
- `TensorInternalAccess` jako primarni kernel argument
- stary ASM generator jako primarni runtime kontrakt; cpu1 misto toho dostane vlastni vector/codegen hot path
- `Operation.OpType.FUSED` jako runtime fasadu pro cpu1

## Non-Goals

- Nezavadet obecnou compatibility vrstvu mezi starym `backend.cpu.fused` a `backend.cpu1`.
- Nekopirovat stary `CpuFusedExecutionArtifact`.
- Neportovat stary ASM generator 1:1 jako compatibility vrstvu.
- Nechat stary CPU fused runtime jako fallback jen po dobu migrace rizeni routy; finalni stav planu musi mit cpu1 route explicitne konfigurovatelnou a otestovanou.
- Nemenit graph optimizer fusion pravidla, pokud neni nutne opravit bug.
- Nemenit public `Tensor` API.
- Nekomitovat lokalni benchmark/profilove artefakty.
- Nezapinat cpu1 fused jako default bez parity testu, benchmarku a trace evidence. Plan obsahuje kompletni route knob a overeni v ramci teto migrace.

## Null Validation Style

V novem `cpu1` fused kodu nepouzivat `Objects` utility pro null validaci.

Preferovany styl je explicitni kontrola:

```java
if (outputNode == null) {
    throw new IllegalArgumentException("outputNode cannot be null");
}
```

Proc:

- Je jasne videt typ vyjimky.
- Prepare/config/IR invarianty failnou primo v miste vstupu.
- Styl je citelnejsi pro verejnejsi prepare metody a immutable prepared objekty.
- Kernel hot path nesmi pridavat null checky do per-element smycek.

## Dulezite Existujici Vstupy

### Graph fused unit uz existuje

Soubor:

```text
src/main/java/graph/compile/planning/region/ElementwiseFusionPlanner.java
```

Relevantni soucasny stav:

```java
static boolean shouldFuseWholePartition(Partition partition, RegionOptimizationContext context) {
    if (partition == null || partition.orderedNodeIds().size() < 2) {
        return false;
    }
    if (partition.outputValueRefs().size() != 1) {
        return false;
    }
    if (partition.target() == PartitionTarget.NONE) {
        return false;
    }
    for (int nodeId : partition.orderedNodeIds()) {
        CompiledNode node = context.compiledNode(nodeId);
        if (!isSubchainFusable(node)) {
            return false;
        }
    }
    return true;
}
```

To znamena: cpu1 nemusi hledat fuzovatelne chainy. Dostane uz lowered region unit.

### PreparedExecutionBuilder uz ma specialni fused step

Soubor:

```text
src/main/java/backend/prepare/PreparedExecutionBuilder.java
```

Relevantni flow:

```java
LoweredExecutionUnit fusedUnit = context.cpuFusedUnitForStart(node.id());
if (fusedUnit != null) {
    addPreparedRegionStep(
            prepareCpuFusedStep(fusedUnit, context, dispatcher),
            context,
            program.forwardBoundaryNodeId(),
            executionSteps,
            forwardSteps,
            backwardSteps,
            coveredNodeIds
    );
    continue;
}
```

Tady neni potreba menit logiku region step. Zmena patri do `BackendPrepareDispatcher.prepareCpuFusedStep(...)`.

### Dnesni fused prepare jde do stareho CPU

Soubor:

```text
src/main/java/backend/prepare/BackendPrepareDispatcher.java
```

Soucasny kod:

```java
public CompiledNodeExecutionMetadata prepareCpuFusedStep(
        CompiledNode outputNode,
        LoweredExecutionUnit loweredUnit,
        BackendPrepareContext context
) {
    if (outputNode == null) {
            throw new IllegalArgumentException("outputNode cannot be null");
        }
    if (loweredUnit == null) {
            throw new IllegalArgumentException("loweredUnit cannot be null");
        }
    if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
    return cpuPreparer.prepareLoweredFusedStep(outputNode, loweredUnit, context);
}
```

Cilova zmena nesmi byt slepe "vzdy cpu1". Nejdriv musi existovat cpu1 fused preparer a testy. Pak lze prepnout trasu nebo pridat jasny policy gate.

## Navrzena Cilova Struktura

```text
src/main/java/backend/cpu1/fused/ir/
  Cpu1FusedAccessKind.java
  Cpu1FusedNodeAttributes.java
  Cpu1NoAttributes.java
  Cpu1ScalarAttribute.java
  Cpu1WhereAttributes.java
  Cpu1FusedInputPlan.java
  Cpu1FusedNodePlan.java
  Cpu1FusedExpressionPlan.java
  Cpu1FusedIrBuilder.java

src/main/java/backend/cpu1/prepare/
  Cpu1FusedElementwisePreparer.java
  Cpu1PreparedFusedElementwiseUnit.java

src/main/java/backend/cpu1/exec/
  Cpu1FusedKernelArgs.java
  Cpu1FusedElementwiseExecutableUnit.java

src/main/java/backend/cpu1/kernels/fused/
  Cpu1FusedElementwiseRangeRunner.java
  Cpu1FusedElementwiseLoops.java
  Cpu1FusedElementwiseDispatch.java

src/main/java/backend/cpu1/kernels/fused/scalar/
  Cpu1FusedScalarInterpreter.java

src/main/java/backend/cpu1/kernels/fused/vector/
  Cpu1FusedVectorPlan.java
  Cpu1FusedVectorInterpreter.java
  Cpu1FusedVectorDispatch.java
  Cpu1FusedVectorFallbackReason.java

src/main/java/backend/cpu1/kernels/fused/codegen/
  Cpu1FusedCodegenKernel.java
  Cpu1FusedCodegenKernelFactory.java
  Cpu1FusedCodegenKernelCacheKey.java

src/main/java/backend/cpu1/kernels/fused/tuning/
  Cpu1FusedTuningClassifier.java

modified:
  src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java
  src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java
  src/main/java/backend/prepare/BackendPrepareDispatcher.java
  src/main/java/config/runtime/FusedExecutionPolicy.java
  src/main/java/config/profile/ExecutionProfileIO.java
```

Proc `Cpu1FusedElementwiseExecutableUnit` patri do `backend.cpu1.exec`:

- vsechny runtime executable jednotky cpu1 jsou tam;
- `prepare` drzi immutable prepared metadata;
- `kernels` drzi konkretni vypocetni smycky;
- fused IR neni executable jednotka, proto zustava pod `backend.cpu1.fused.ir`.

### Naming Decision: `fused`, `vector`, `codegen`

`fused` zustava v cpu1 kernel runtime jako vecny popis typu kernelu: jde o fused elementwise execution, ne o komponentu, ktera fuzovani sama provadi.

`fuser` se v tomto planu nepouziva pro cpu1 runtime balicky. Pokud vznikne komponenta, ktera sklada nebo vybira fuse regiony, patri do graph/lowering vrstvy, napriklad jako `FusionPass`, `Fuser` nebo `FusionPlanner`.

Plan nema zadne cilove soubory pod adresarem pro generovane artefakty. Cilovy balicek je `codegen/`, protoze obsahuje runtime cestu pro tvorbu a cacheovani specializovaneho kernelu. Stejne tak navrzene tridy pouzivaji `Cpu1FusedCodegenKernel*` naming.

## Faze 0: Predimplementacni Kontrola

### Task 0.1: Zkontrolovat dirty tree

Stav: `[x]`

Spustit:

```bash
git status --short
```

Pravidla:

- Nesahat na `.idea/*`.
- Nesahat na `profiles/platform/*`, pokud se explicitne neaktualizuji kanonicke profily.
- Pokud stale existuji rozpracovane MSE zmeny, neprepisovat je.
- Implementacni commit pro fused cpu1 ma stagovat pouze relevantni `src/main/java` a `src/test/java`.

Evidence:

- `git status --short` ukazuje existujici dirty `.idea/*`, `profiles/platform/*` a vice `src/main/java/backend/cpu1/*` souboru; Phase 0 je nechava beze zmen.
- Rozpracovane cpu1/MSE/runtime zmeny jsou pritomne v pracovnim stromu, proto dalsi faze musi editovat jen relevantni fused soubory a nesmi prepisovat aktualni cpu1 zmeny.

### Task 0.2: Overit stare fused soubory jako reference

Stav: `[x]`

Precist:

```text
src/main/java/backend/cpu/fused/ir/FusedIrBuilder.java
src/main/java/backend/cpu/fused/ir/FusedExpressionPlan.java
src/main/java/backend/cpu/fused/ir/FusedNodePlan.java
src/main/java/backend/cpu/fused/ir/FusedExternalInputPlan.java
src/main/java/backend/cpu/fused/exec/InterpretedPreparedFusedExecutable.java
src/main/java/backend/cpu/fused/plan/FusedDispatchPlanner.java
src/main/java/backend/cpu/fused/numeric/FusedNumericContractResolver.java
```

Proc:

- IR builder a canonicalizace jsou primo prenositelne.
- Interpreter je semanticka reference, ale ne runtime struktura k prekopirovani.
- Dispatch planner je inspirace, ale cpu1 musi pouzit `Cpu1DispatchPolicy`/`CpuKernelConfig`.

Evidence:

- `FusedIrBuilder` obsahuje prenositelne ref mapovani, canonicalizaci `pow`/`mulScalar`, extrakci atributu a broadcast/effective-stride access klasifikaci.
- `FusedExpressionPlan`, `FusedNodePlan` a `FusedExternalInputPlan` jsou male immutable plan objekty s kopirovanim listu/poli; pro cpu1 je nutne prepsat null validaci do explicitniho stylu bez `Objects`.
- `InterpretedPreparedFusedExecutable` potvrzuje semantiku op evaluace a storage index vypoctu, ale je svazany s `TensorInternalAccess`, `Tensor`, `CpuKernelContext` a CPU_JAVA_ARRAY-only fallbackem.
- `FusedDispatchPlanner` klasifikuje cheap/non-cheap a contiguous/strided rodiny; cpu1 navazka ma stejnou myslenku vyjadrit pres `Cpu1DispatchPolicy` a `CpuKernelConfig`.
- `FusedNumericContractResolver` podporuje floating/BOOL cesty a odmita INT32/INT64 ve fused numeric contractu; cpu1 plan musi explicitne rozhodnout dtype/storage kontrakt.
- Aktualni graph vrstva uz produkuje `ExecutionUnitKind.FUSED_ELEMENTWISE`; `PreparedExecutionBuilder` vola `BackendPrepareDispatcher.prepareCpuFusedStep(...)`, ktery dnes stale routuje do stareho `cpuPreparer.prepareLoweredFusedStep(...)`.
- Aktualni cpu1 integracni body jsou `Cpu1PreparedArtifact`, `Cpu1ElementwiseExecutableUnit`, `Cpu1KernelArgs`, `Cpu1TraceContributor` a `Cpu1NodePreparer`; zatim neexistuje cpu1 fused prepared unit ani trace vetev.

## Faze 1: cpu1 Fused IR

### Task 1.1: Pridat `Cpu1FusedAccessKind`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedAccessKind.java
```

Kod:

```java
package backend.cpu1.fused.ir;

public enum Cpu1FusedAccessKind {
    DIRECT_CONTIGUOUS,
    OFFSET_CONTIGUOUS,
    DIRECT_STRIDED,
    OFFSET_STRIDED,
    BROADCAST_STRIDED
}
```

Proc:

- Prepare potrebuje predem vedet, jestli jsou vsechny vstupy linearni.
- Runner pak nemusi pri kazdem prvku analyzovat layout.

### Task 1.2: Pridat atributy fused nodu

Stav: `[ ]`

Nove soubory:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedNodeAttributes.java
src/main/java/backend/cpu1/fused/ir/Cpu1NoAttributes.java
src/main/java/backend/cpu1/fused/ir/Cpu1ScalarAttribute.java
src/main/java/backend/cpu1/fused/ir/Cpu1WhereAttributes.java
```

Kod:

```java
package backend.cpu1.fused.ir;

public interface Cpu1FusedNodeAttributes {
}
```

```java
package backend.cpu1.fused.ir;

public enum Cpu1NoAttributes implements Cpu1FusedNodeAttributes {
    INSTANCE
}
```

```java
package backend.cpu1.fused.ir;

public record Cpu1ScalarAttribute(float f32, double f64) implements Cpu1FusedNodeAttributes {
}
```

```java
package backend.cpu1.fused.ir;

public enum Cpu1WhereAttributes implements Cpu1FusedNodeAttributes {
    INSTANCE
}
```

Proc:

- `POW`, `MUL_SCALAR`, `CLAMP_MIN`, `CLAMP_MAX` potrebuji scalar parametr.
- Scalar atribut drzi F32 i F64 reprezentaci, aby F32/BF16 hot path nemusela opakovane castovat z double.
- `WHERE` ma specialni aritu a bool condition, ale nepotrebuje data navic.
- Pouzivame jednoduche typy, zadne abstraktni compatibility vrstvy.

### Task 1.3: Pridat `Cpu1FusedInputPlan`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedInputPlan.java
```

Kod:

```java
package backend.cpu1.fused.ir;

import tensor.DataType;

import java.util.Arrays;

public record Cpu1FusedInputPlan(
        int ref,
        int nodeId,
        DataType dataType,
        int[] shape,
        int[] strides,
        int[] logicalOutputShape,
        int[] logicalOutputDenseStrides,
        int storageOffset,
        int[] effectiveStrides,
        Cpu1FusedAccessKind accessKind
) {
    public Cpu1FusedInputPlan {
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        shape = shape == null ? new int[0] : shape.clone();
        strides = strides == null ? new int[0] : strides.clone();
        logicalOutputShape = logicalOutputShape == null ? new int[0] : logicalOutputShape.clone();
        logicalOutputDenseStrides = logicalOutputDenseStrides == null ? new int[0] : logicalOutputDenseStrides.clone();
        effectiveStrides = effectiveStrides == null ? new int[0] : effectiveStrides.clone();
        if (accessKind == null) {
            throw new IllegalArgumentException("accessKind cannot be null");
        }
    }

    public boolean isLinearAccess() {
        return accessKind == Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                || accessKind == Cpu1FusedAccessKind.OFFSET_CONTIGUOUS;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    @Override
    public int[] logicalOutputShape() {
        return logicalOutputShape.clone();
    }

    @Override
    public int[] logicalOutputDenseStrides() {
        return logicalOutputDenseStrides.clone();
    }

    @Override
    public int[] effectiveStrides() {
        return effectiveStrides.clone();
    }

    @Override
    public String toString() {
        return "Cpu1FusedInputPlan{"
                + "ref=" + ref
                + ", nodeId=" + nodeId
                + ", dataType=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", effectiveStrides=" + Arrays.toString(effectiveStrides)
                + ", accessKind=" + accessKind
                + '}';
    }
}
```

Proc:

- Vstupy fused chainu jsou external inputs.
- Internal nodu nebudeme materializovat.
- `effectiveStrides` resi broadcast uz v prepare.

### Task 1.4: Pridat `Cpu1FusedNodePlan`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedNodePlan.java
```

Kod:

```java
package backend.cpu1.fused.ir;

import operations.Operation;
import tensor.DataType;

import java.util.List;

public record Cpu1FusedNodePlan(
        int index,
        int nodeId,
        Operation.OpType opType,
        List<Integer> inputRefs,
        int outputRef,
        DataType outputType,
        Cpu1FusedNodeAttributes attributes
) {
    public Cpu1FusedNodePlan {
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (inputRefs == null) {
            throw new IllegalArgumentException("inputRefs cannot be null");
        }
        inputRefs = List.copyOf(inputRefs);
        if (outputType == null) {
            throw new IllegalArgumentException("outputType cannot be null");
        }
        attributes = attributes == null ? Cpu1NoAttributes.INSTANCE : attributes;
    }
}
```

Proc:

- Fused node neobsahuje `Tensor`.
- Obsahuje jen op type, refs, dtype a atributy.
- `nodeId` zustava pro trace/debug a validaci.

### Task 1.5: Pridat `Cpu1FusedExpressionPlan`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedExpressionPlan.java
```

Kod:

```java
package backend.cpu1.fused.ir;

import java.util.List;

public record Cpu1FusedExpressionPlan(
        List<Cpu1FusedNodePlan> nodes,
        List<Cpu1FusedInputPlan> inputs,
        int outputRef
) {
    public Cpu1FusedExpressionPlan {
        if (nodes == null) {
            throw new IllegalArgumentException("nodes cannot be null");
        }
        nodes = List.copyOf(nodes);
        if (inputs == null) {
            throw new IllegalArgumentException("inputs cannot be null");
        }
        inputs = List.copyOf(inputs);
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }
    }

    public int inputCount() {
        return inputs.size();
    }

    public int nodeCount() {
        return nodes.size();
    }

    public Cpu1FusedNodePlan outputNode() {
        int nodeIndex = outputRef - inputCount();
        if (nodeIndex < 0 || nodeIndex >= nodes.size()) {
            throw new IllegalStateException("Fused outputRef does not point to an internal node: " + outputRef);
        }
        return nodes.get(nodeIndex);
    }

    public boolean usesOnlyLinearInputs() {
        for (Cpu1FusedInputPlan input : inputs) {
            if (!input.isLinearAccess()) {
                return false;
            }
        }
        return true;
    }
}
```

Proc:

- Prepared unit potrebuje cele expression metadata.
- `usesOnlyLinearInputs()` bude jednoducha dispatch/trace informace.

### Task 1.6: Pridat `Cpu1FusedIrBuilder`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedIrBuilder.java
```

Kod:

```java
package backend.cpu1.fused.ir;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.TensorMetadata;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.function.IntFunction;

public final class Cpu1FusedIrBuilder {
    private Cpu1FusedIrBuilder() {
    }

    public static Cpu1FusedExpressionPlan build(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (compiledNodeResolver == null) {
            throw new IllegalArgumentException("compiledNodeResolver cannot be null");
        }
        if (descriptorIndex == null) {
            throw new IllegalArgumentException("descriptorIndex cannot be null");
        }
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }
        return build(
                orderedNodeIds,
                externalInputNodeIds(orderedNodeIds, compiledNodeResolver),
                compiledNodeResolver,
                descriptorIndex
        );
    }

    public static Cpu1FusedExpressionPlan build(
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (externalInputNodeIds == null) {
            throw new IllegalArgumentException("externalInputNodeIds cannot be null");
        }
        if (compiledNodeResolver == null) {
            throw new IllegalArgumentException("compiledNodeResolver cannot be null");
        }
        if (descriptorIndex == null) {
            throw new IllegalArgumentException("descriptorIndex cannot be null");
        }
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }

        java.util.HashMap<Integer, Integer> refs = new java.util.HashMap<>();
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            refs.put(externalInputNodeIds.get(i), i);
        }

        CompiledTensorDescriptor outputDescriptor = descriptorIndex.byNodeId(orderedNodeIds.getLast());
        List<Cpu1FusedInputPlan> inputPlans = buildInputPlans(externalInputNodeIds, outputDescriptor, descriptorIndex);

        List<Cpu1FusedNodePlan> nodes = new ArrayList<>(orderedNodeIds.size());
        for (int i = 0; i < orderedNodeIds.size(); i++) {
            int nodeId = orderedNodeIds.get(i);
            CompiledNode node = compiledNodeResolver.apply(nodeId);
            if (node == null || node.operation() == null) {
                throw new IllegalArgumentException("Fused node " + nodeId + " does not have an operation");
            }
            int outputRef = externalInputNodeIds.size() + i;
            List<Integer> inputRefs = new ArrayList<>(node.inputIds().size());
            for (int inputNodeId : node.inputIds()) {
                Integer ref = refs.get(inputNodeId);
                if (ref == null) {
                    throw new IllegalStateException("Missing fused input reference for nodeId=" + nodeId);
                }
                inputRefs.add(ref);
            }
            refs.put(nodeId, outputRef);
            CanonicalNode canonical = canonicalize(node.operation(), inputRefs);
            nodes.add(new Cpu1FusedNodePlan(
                    i,
                    nodeId,
                    canonical.opType(),
                    canonical.inputRefs(),
                    outputRef,
                    descriptorIndex.byNodeId(nodeId).dataType(),
                    canonical.attributes()
            ));
        }

        Integer outputRef = refs.get(orderedNodeIds.getLast());
        if (outputRef == null) {
            throw new IllegalStateException("Missing output ref for fused root nodeId=" + orderedNodeIds.getLast());
        }
        return new Cpu1FusedExpressionPlan(nodes, inputPlans, outputRef);
    }

    private static List<Integer> externalInputNodeIds(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        LinkedHashSet<Integer> chainNodeIds = new LinkedHashSet<>(orderedNodeIds);
        LinkedHashSet<Integer> externalInputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = requireNode(compiledNodeResolver, nodeId, "fused unit");
            for (int inputNodeId : node.inputIds()) {
                if (chainNodeIds.contains(inputNodeId)) {
                    continue;
                }
                requireNode(compiledNodeResolver, inputNodeId, "fused unit input");
                externalInputs.add(inputNodeId);
            }
        }
        return List.copyOf(externalInputs);
    }

    private static CompiledNode requireNode(
            IntFunction<CompiledNode> compiledNodeResolver,
            int nodeId,
            String context
    ) {
        CompiledNode node = compiledNodeResolver.apply(nodeId);
        if (node == null) {
            throw new IllegalStateException("Missing compiled node for " + context + " nodeId=" + nodeId);
        }
        return node;
    }

    private static CanonicalNode canonicalize(Operation operation, List<Integer> inputRefs) {
        if (operation instanceof pow p && inputRefs.size() == 1) {
            double exponent = p.getExponent();
            int inputRef = inputRefs.getFirst();
            if (exponent == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        new Cpu1ScalarAttribute(1.0f, 1.0d)
                );
            }
            if (exponent == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1NoAttributes.INSTANCE);
            }
            if (exponent == -1.0d) {
                return new CanonicalNode(Operation.OpType.INV, List.of(inputRef), Cpu1NoAttributes.INSTANCE);
            }
            if (exponent == 2.0d) {
                return new CanonicalNode(Operation.OpType.MUL, List.of(inputRef, inputRef), Cpu1NoAttributes.INSTANCE);
            }
        }
        if (operation instanceof mulScalar m && inputRefs.size() == 1) {
            double scalar = m.getScalar();
            int inputRef = inputRefs.getFirst();
            if (scalar == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        new Cpu1ScalarAttribute(0.0f, 0.0d)
                );
            }
            if (scalar == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1NoAttributes.INSTANCE);
            }
            if (scalar == -1.0d) {
                return new CanonicalNode(Operation.OpType.NEG, List.of(inputRef), Cpu1NoAttributes.INSTANCE);
            }
        }
        return new CanonicalNode(operation.opType(), List.copyOf(inputRefs), extractAttributes(operation));
    }

    private static Cpu1FusedNodeAttributes extractAttributes(Operation operation) {
        if (operation instanceof pow p) {
            return new Cpu1ScalarAttribute(p.getExponentF32(), p.getExponent());
        }
        if (operation instanceof mulScalar m) {
            return new Cpu1ScalarAttribute(m.getScalarF32(), m.getScalar());
        }
        if (operation instanceof clampMin c) {
            return new Cpu1ScalarAttribute(c.getMinValueF32(), c.getMinValue());
        }
        if (operation instanceof clampMax c) {
            return new Cpu1ScalarAttribute(c.getMaxValueF32(), c.getMaxValue());
        }
        if (operation.opType() == Operation.OpType.WHERE) {
            return Cpu1WhereAttributes.INSTANCE;
        }
        return Cpu1NoAttributes.INSTANCE;
    }

    private static List<Cpu1FusedInputPlan> buildInputPlans(
            List<Integer> externalInputNodeIds,
            CompiledTensorDescriptor output,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        int[] outShape = output.shape();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        List<Cpu1FusedInputPlan> plans = new ArrayList<>(externalInputNodeIds.size());
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            int inputNodeId = externalInputNodeIds.get(i);
            CompiledTensorDescriptor input = descriptorIndex.byNodeId(inputNodeId);
            BroadcastPlan plan = BroadcastPlanner.plan(
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides
            );
            if (!Arrays.equals(plan.outShape(), outShape)) {
                throw new IllegalArgumentException("Fused broadcast shape mismatch for external input nodeId=" + inputNodeId);
            }
            int[] effectiveStrides = plan.aEffStrides();
            plans.add(new Cpu1FusedInputPlan(
                    i,
                    inputNodeId,
                    input.dataType(),
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides,
                    input.storageOffset(),
                    effectiveStrides,
                    classifyAccessKind(effectiveStrides, outDenseStrides, input.storageOffset())
            ));
        }
        return List.copyOf(plans);
    }

    private static Cpu1FusedAccessKind classifyAccessKind(
            int[] effectiveStrides,
            int[] denseStrides,
            int storageOffset
    ) {
        boolean broadcast = false;
        for (int stride : effectiveStrides) {
            if (stride == 0) {
                broadcast = true;
                break;
            }
        }
        if (broadcast) {
            return Cpu1FusedAccessKind.BROADCAST_STRIDED;
        }
        if (storageOffset != 0) {
            return Arrays.equals(effectiveStrides, denseStrides)
                    ? Cpu1FusedAccessKind.OFFSET_CONTIGUOUS
                    : Cpu1FusedAccessKind.OFFSET_STRIDED;
        }
        return Arrays.equals(effectiveStrides, denseStrides)
                ? Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                : Cpu1FusedAccessKind.DIRECT_STRIDED;
    }

    private record CanonicalNode(
            Operation.OpType opType,
            List<Integer> inputRefs,
            Cpu1FusedNodeAttributes attributes
    ) {
    }
}
```

Proc:

- Tohle je nejdulezitejsi cast, kterou prebirame ze stareho CPU.
- Je stale prepare-time a backend-local.
- Nepouziva `Tensor`.

## Faze 2: Prepared Fused Unit A Dispatch

### Task 2.1: Pridat `Cpu1PreparedFusedElementwiseUnit`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1PreparedFusedElementwiseUnit.java
```

Kod:

```java
package backend.cpu1.prepare;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;

public final class Cpu1PreparedFusedElementwiseUnit {
    private final String unitId;
    private final List<Integer> orderedNodeIds;
    private final List<Integer> inputNodeIds;
    private final int outputNodeId;
    private final DataType outputDataType;
    private final int elementCount;
    private final int[] outputShape;
    private final Cpu1FusedExpressionPlan plan;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1DispatchDecision dispatchDecision;
    private final boolean approximateExp;
    private final boolean approximateTanh;

    public Cpu1PreparedFusedElementwiseUnit(
            String unitId,
            List<Integer> orderedNodeIds,
            List<Integer> inputNodeIds,
            int outputNodeId,
            DataType outputDataType,
            int elementCount,
            int[] outputShape,
            Cpu1FusedExpressionPlan plan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1LaunchConfig launchConfig,
            Cpu1DispatchDecision dispatchDecision,
            boolean approximateExp,
            boolean approximateTanh
    ) {
        if (unitId == null) {
            throw new IllegalArgumentException("unitId cannot be null");
        }
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (inputNodeIds == null) {
            throw new IllegalArgumentException("inputNodeIds cannot be null");
        }
        if (outputDataType == null) {
            throw new IllegalArgumentException("outputDataType cannot be null");
        }
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (launchPolicy == null) {
            throw new IllegalArgumentException("launchPolicy cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (dispatchDecision == null) {
            throw new IllegalArgumentException("dispatchDecision cannot be null");
        }
        this.unitId = unitId;
        this.orderedNodeIds = List.copyOf(orderedNodeIds);
        this.inputNodeIds = List.copyOf(inputNodeIds);
        this.outputNodeId = outputNodeId;
        this.outputDataType = outputDataType;
        this.elementCount = elementCount;
        this.outputShape = outputShape == null ? new int[0] : outputShape.clone();
        this.plan = plan;
        this.layoutKind = layoutKind;
        this.storageKind = storageKind;
        this.launchPolicy = launchPolicy;
        this.launchConfig = launchConfig;
        this.dispatchDecision = dispatchDecision;
        this.approximateExp = approximateExp;
        this.approximateTanh = approximateTanh;
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
    }

    public String unitId() {
        return unitId;
    }

    public List<Integer> orderedNodeIds() {
        return orderedNodeIds;
    }

    public List<Integer> inputNodeIds() {
        return inputNodeIds;
    }

    public int outputNodeId() {
        return outputNodeId;
    }

    public DataType outputDataType() {
        return outputDataType;
    }

    public int elementCount() {
        return elementCount;
    }

    public int[] outputShape() {
        return outputShape.clone();
    }

    public Cpu1FusedExpressionPlan plan() {
        return plan;
    }

    public Cpu1LayoutKind layoutKind() {
        return layoutKind;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1DispatchDecision dispatchDecision() {
        return dispatchDecision;
    }

    public boolean approximateExp() {
        return approximateExp;
    }

    public boolean approximateTanh() {
        return approximateTanh;
    }

    public boolean containsOp(Operation.OpType opType) {
        for (var node : plan.nodes()) {
            if (node.opType() == opType) {
                return true;
            }
        }
        return false;
    }
}
```

Proc:

- Prepared unit je immutable popis fused vypoctu.
- Exekuce bude bindovat runtime views az v `Cpu1FusedElementwiseExecutableUnit`.
- `launchConfig` je ulozen zvlast pro trace, protoze `Cpu1LaunchPolicy` je interface.

### Task 2.2: Rozsirit `Cpu1DispatchPolicy` o fused rozhodnuti

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java
```

Pridat public metodu:

```java
public Cpu1DispatchDecision decideFusedElementwise(
        Cpu1FusedExpressionPlan plan,
        DataType computeType,
        long elementCount,
        Cpu1PrepareConfig config
) {
    if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
    if (computeType == null) {
            throw new IllegalArgumentException("computeType cannot be null");
        }
    if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
    Operation.OpType representativeOp = fusedRepresentativeOp(plan);
    Cpu1CostClass costClass = classifyFusedElementwise(plan);
    int totalLength = saturatingElementCount(elementCount);
    int vectorWidth = preferredVectorWidth(computeType);
    CpuKernelConfig cpuKernelConfig = cpuKernelConfig(config);
    Cpu1VectorizationKind vectorizationKind = resolveVectorizationKind(
            config,
            cpuKernelConfig,
            costClass,
            config.storageKind(),
            computeType,
            totalLength,
            vectorWidth
    );
    int plannedWorkers = resolvePlannedWorkers(config, cpuKernelConfig, costClass, totalLength);
    int scalarChunkSize = scalarChunkSize(config, cpuKernelConfig, costClass, totalLength, plannedWorkers);
    int vectorChunkSize = vectorChunkSize(config, cpuKernelConfig, costClass, totalLength, vectorWidth, plannedWorkers);
    Cpu1LaunchConfig launchConfig = launchConfig(config, plannedWorkers, vectorizationKind, scalarChunkSize, vectorChunkSize);
    return new Cpu1DispatchDecision(
            representativeOp,
            costClass,
            vectorizationKind,
            launchConfig,
            config.storageKind(),
            scalarChunkSize,
            vectorChunkSize,
            plannedWorkers
    );
}
```

Doplnit private helpery:

```java
private static Operation.OpType fusedRepresentativeOp(Cpu1FusedExpressionPlan plan) {
    boolean expensive = false;
    for (var node : plan.nodes()) {
        if (classifyElementwise(node.opType()) == Cpu1CostClass.EXPENSIVE_ELEMENTWISE) {
            expensive = true;
            break;
        }
    }
    return expensive ? Operation.OpType.EXP : Operation.OpType.ADD;
}

private static Cpu1CostClass classifyFusedElementwise(Cpu1FusedExpressionPlan plan) {
    for (var input : plan.inputs()) {
        if (input.dataType() == DataType.BOOL) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
    }
    for (var node : plan.nodes()) {
        if (node.outputType() == DataType.BOOL) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
        if (classifyElementwise(node.opType()) == Cpu1CostClass.EXPENSIVE_ELEMENTWISE) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
        if (node.opType() == Operation.OpType.WHERE
                || node.opType() == Operation.OpType.GT
                || node.opType() == Operation.OpType.GE
                || node.opType() == Operation.OpType.LT
                || node.opType() == Operation.OpType.LE
                || node.opType() == Operation.OpType.EQ
                || node.opType() == Operation.OpType.NE
                || node.opType() == Operation.OpType.LOGICAL_AND
                || node.opType() == Operation.OpType.LOGICAL_OR
                || node.opType() == Operation.OpType.LOGICAL_NOT) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
    }
    return Cpu1CostClass.CHEAP_ELEMENTWISE;
}
```

Nutne importy:

```java
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
```

Proc:

- Fused chain ma cost podle nejdrazsiho nodu.
- Stejna dispatch decision se pouzije pro fused scalar, vector i codegen hot path.
- Execute nesmi pocitat thresholdy ani znovu rozhodovat scalar/vector route.

Poznamka:

- Pokud `classifyElementwise(...)` zustane private, helpery ve stejne tride ho muzou pouzit.
- Pokud bude implementace potrebovat testovat cost separovane, pridej package-private metodu primo v teto fazi.

### Task 2.3: Pridat `Cpu1FusedElementwisePreparer`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1FusedElementwisePreparer.java
```

Kod:

```java
package backend.cpu1.prepare;

import backend.ComputeBackend;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedIrBuilder;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.lowering.LoweredExecutionUnit;
import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import operations.Operation;
import tensor.DataType;

import java.util.List;

public final class Cpu1FusedElementwisePreparer {
    private final config.runtime.RuntimeConfig runtimeConfig;
    private final Cpu1DispatchPolicy dispatchPolicy = new Cpu1DispatchPolicy();

    public Cpu1FusedElementwisePreparer(config.runtime.RuntimeConfig runtimeConfig) {
        if (runtimeConfig == null) {
            throw new IllegalArgumentException("runtimeConfig cannot be null");
        }
        this.runtimeConfig = runtimeConfig;
    }

    public CompiledNodeExecutionMetadata prepare(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        if (outputNode == null) {
            throw new IllegalArgumentException("outputNode cannot be null");
        }
        if (loweredUnit == null) {
            throw new IllegalArgumentException("loweredUnit cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        validate(outputNode, loweredUnit, context);

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                loweredUnit.orderedNodeIds(),
                context::compiledNode,
                context.descriptorIndex()
        );
        DataType computeType = computeType(plan);
        Cpu1PrepareConfig config = Cpu1PrepareConfig
                .automatic(runtimeConfig, Runtime.getRuntime().availableProcessors(), storageKindFromRuntime())
                .withApproximation(
                        runtimeConfig.approximation().useFastExp(),
                        runtimeConfig.approximation().useFastTanh()
                );
        Cpu1DispatchDecision dispatchDecision = dispatchPolicy.decideFusedElementwise(
                plan,
                computeType,
                outputNode.flatDataSize(),
                config
        );
        Cpu1LaunchConfig launchConfig = dispatchDecision.launchConfig();
        Cpu1PreparedFusedElementwiseUnit preparedUnit = new Cpu1PreparedFusedElementwiseUnit(
                loweredUnit.unitId(),
                loweredUnit.orderedNodeIds(),
                plan.inputs().stream().map(input -> input.nodeId()).toList(),
                outputNode.id(),
                outputNode.dataType(),
                outputNode.flatDataSize(),
                outputNode.shape(),
                plan,
                layoutKind(plan, outputNode),
                dispatchDecision.storageKind(),
                launchPolicy(launchConfig),
                launchConfig,
                dispatchDecision,
                config.useFastExpApprox(),
                config.useFastTanhApprox()
        );
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                preparedUnit.inputNodeIds(),
                new Cpu1PreparedArtifact(preparedUnit),
                inputResidencyRequirement(dispatchDecision),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private void validate(CompiledNode outputNode, LoweredExecutionUnit loweredUnit, BackendPrepareContext context) {
        if (loweredUnit.orderedNodeIds().isEmpty()) {
            throw new IllegalStateException("cpu1 fused unit has no ordered nodes: " + loweredUnit.unitId());
        }
        if (loweredUnit.orderedNodeIds().getLast() != outputNode.id()) {
            throw new IllegalStateException("cpu1 fused output must be last ordered node. outputNodeId="
                    + outputNode.id() + ", orderedNodeIds=" + loweredUnit.orderedNodeIds());
        }
        for (int nodeId : loweredUnit.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null || !node.operation().opType().isFusable()) {
                throw new UnsupportedOperationException("cpu1 fused unit contains non-fusable nodeId=" + nodeId);
            }
            requireSupportedFusedOp(node.operation().opType());
            requireSupportedDType(node.dataType(), "node " + nodeId + " output");
        }
    }

    private static void requireSupportedFusedOp(Operation.OpType opType) {
        switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX,
                    GT, GE, LT, LE, EQ, NE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                    WHERE,
                    NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                    POW, POW_TENSOR, SQRT, ABS, MUL_SCALAR,
                    RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> {
            }
            default -> throw new UnsupportedOperationException("cpu1 fused does not support " + opType);
        }
    }

    private static void requireSupportedDType(DataType dataType, String role) {
        if (dataType != DataType.FLOAT32
                && dataType != DataType.FLOAT64
                && dataType != DataType.BFLOAT16
                && dataType != DataType.BOOL) {
            throw new UnsupportedOperationException("cpu1 fused does not support " + role + " dtype " + dataType);
        }
    }

    private static DataType computeType(Cpu1FusedExpressionPlan plan) {
        DataType result = null;
        for (var input : plan.inputs()) {
            result = promote(result, input.dataType());
        }
        for (var node : plan.nodes()) {
            result = promote(result, node.outputType());
        }
        return result == null || result == DataType.BOOL ? DataType.FLOAT32 : result;
    }

    private static DataType promote(DataType current, DataType next) {
        if (next == null || next == DataType.BOOL) {
            return current;
        }
        if (next == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (next == DataType.FLOAT32) {
            return current == DataType.FLOAT64 ? current : DataType.FLOAT32;
        }
        if (next == DataType.BFLOAT16 && current == null) {
            return DataType.BFLOAT16;
        }
        return current;
    }

    private static Cpu1LayoutKind layoutKind(Cpu1FusedExpressionPlan plan, CompiledNode outputNode) {
        if (outputNode.contiguous() && plan.usesOnlyLinearInputs()) {
            return Cpu1LayoutKind.CONTIGUOUS;
        }
        return switch (outputNode.shape().length) {
            case 2 -> Cpu1LayoutKind.STRIDED_RANK2;
            case 3 -> Cpu1LayoutKind.STRIDED_RANK3;
            case 4 -> Cpu1LayoutKind.STRIDED_RANK4;
            default -> Cpu1LayoutKind.STRIDED_GENERIC;
        };
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        return launchConfig.workerCount() == 1
                ? new Cpu1SingleThreadLaunch(launchConfig)
                : new Cpu1ParallelLaunch(launchConfig);
    }

    private backend.cpu1.storage.Cpu1StorageKind storageKindFromRuntime() {
        return runtimeConfig.cpuStorageProfile() == config.runtime.CpuStorageProfile.CPU_NATIVE
                ? backend.cpu1.storage.Cpu1StorageKind.MEMORY_SEGMENT
                : backend.cpu1.storage.Cpu1StorageKind.JAVA_ARRAY;
    }

    private static InputResidencyRequirement inputResidencyRequirement(Cpu1DispatchDecision decision) {
        return decision.storageKind() == backend.cpu1.storage.Cpu1StorageKind.MEMORY_SEGMENT
                ? InputResidencyRequirement.none()
                : InputResidencyRequirement.cpuReadableAll();
    }
}
```

Proc:

- Preparer je misto, kde se fused lowered unit prelozi na cpu1 prepared unit.
- Dela validate podporovanych fusable op.
- Rozhoduje storage podle runtime CPU storage profile.
- Nepouziva stary `FusedOperationBuilder`.

Poznamka k `runtimeConfig.approximation().useFastExp()`:

- Pred implementaci overit skutecna jmena metod v `ApproximationConfig`.
- Pokud se jmenuji jinak, kod upravit podle aktualniho API.
- Smysl zustava: EXP/TANH aproximacni policy musi jit z runtime configu.

## Faze 3: Executable Unit V `backend.cpu1.exec`

### Task 3.1: Pridat `Cpu1FusedKernelArgs`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedKernelArgs.java
```

Kod:

```java
package backend.cpu1.exec;

import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.offset.Cpu1GenericOffsetPlan;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import tensor.DataType;

import java.util.List;

public final class Cpu1FusedKernelArgs {
    private final Cpu1PreparedFusedElementwiseUnit preparedUnit;
    private final List<Cpu1TensorView> inputs;
    private final Cpu1TensorView output;
    private final Cpu1GenericOffsetPlan[] inputGenericOffsetPlans;
    private Cpu1GenericOffsetPlan outputGenericOffsetPlan;

    public Cpu1FusedKernelArgs(
            Cpu1PreparedFusedElementwiseUnit preparedUnit,
            List<Cpu1TensorView> inputs,
            Cpu1TensorView output
    ) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        if (inputs == null) {
            throw new IllegalArgumentException("inputs cannot be null");
        }
        if (output == null) {
            throw new IllegalArgumentException("output cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.inputs = List.copyOf(inputs);
        this.output = output;
        if (inputs.size() != preparedUnit.inputNodeIds().size()) {
            throw new IllegalArgumentException("Expected " + preparedUnit.inputNodeIds().size()
                    + " fused inputs, got " + inputs.size());
        }
        if (output.dataType() != preparedUnit.outputDataType()) {
            throw new IllegalArgumentException("Fused output dtype " + output.dataType()
                    + " does not match prepared dtype " + preparedUnit.outputDataType());
        }
        if (output.elementCount() != preparedUnit.elementCount()) {
            throw new IllegalArgumentException("Fused output element count " + output.elementCount()
                    + " does not match prepared element count " + preparedUnit.elementCount());
        }
        requireSupportedView(output, "output");
        for (int i = 0; i < inputs.size(); i++) {
            Cpu1TensorView input = inputs.get(i);
            Cpu1FusedInputPlan inputPlan = preparedUnit.plan().inputs().get(i);
            if (input.dataType() != inputPlan.dataType()) {
                throw new IllegalArgumentException("Fused input " + i + " dtype " + input.dataType()
                        + " does not match prepared dtype " + inputPlan.dataType());
            }
            if (input.elementCount() != preparedUnit.elementCount()) {
                throw new IllegalArgumentException("Fused input " + i + " element count " + input.elementCount()
                        + " does not match prepared element count " + preparedUnit.elementCount());
            }
            requireSupportedView(input, "input " + i);
        }
        this.inputGenericOffsetPlans = new Cpu1GenericOffsetPlan[inputs.size()];
    }

    public Cpu1PreparedFusedElementwiseUnit preparedUnit() {
        return preparedUnit;
    }

    public Cpu1TensorView input(int index) {
        return inputs.get(index);
    }

    public List<Cpu1TensorView> inputs() {
        return inputs;
    }

    public Cpu1TensorView output() {
        return output;
    }

    public int elementCount() {
        return preparedUnit.elementCount();
    }

    public Cpu1GenericOffsetPlan inputGenericOffsetPlan(int inputIndex) {
        Cpu1GenericOffsetPlan plan = inputGenericOffsetPlans[inputIndex];
        if (plan == null) {
            plan = Cpu1GenericOffsetPlan.forView(inputs.get(inputIndex));
            inputGenericOffsetPlans[inputIndex] = plan;
        }
        return plan;
    }

    public Cpu1GenericOffsetPlan outputGenericOffsetPlan() {
        if (outputGenericOffsetPlan == null) {
            outputGenericOffsetPlan = Cpu1GenericOffsetPlan.forView(output);
        }
        return outputGenericOffsetPlan;
    }

    private void requireSupportedView(Cpu1TensorView view, String role) {
        if (view.dataType() != DataType.FLOAT32
                && view.dataType() != DataType.FLOAT64
                && view.dataType() != DataType.BFLOAT16
                && view.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("cpu1 fused only supports FLOAT32/FLOAT64/BFLOAT16/BOOL " + role + " views.");
        }
        if (view.storageKind() != preparedUnit.storageKind()) {
            throw new IllegalArgumentException("cpu1 fused " + role + " storage kind " + view.storageKind()
                    + " does not match prepared storage kind " + preparedUnit.storageKind());
        }
    }
}
```

Proc:

- Nepouzivat `Cpu1KernelArgs`, protoze je pevne navazany na `Cpu1PreparedElementwiseUnit`.
- Fused potrebuje plan pro vice internal nodu, bool/numeric scratch arrays a external input metadata.

### Task 3.2: Pridat `Cpu1FusedElementwiseExecutableUnit`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java
```

Kod:

```java
package backend.cpu1.exec;

import backend.cpu1.kernels.fused.Cpu1FusedElementwiseLoops;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;

public final class Cpu1FusedElementwiseExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedFusedElementwiseUnit preparedUnit;

    public Cpu1FusedElementwiseExecutableUnit(Cpu1PreparedFusedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedFusedElementwiseUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return Cpu1ScratchBufferSpec.none();
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.outputNodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.outputNodeId(),
                    preparedUnit.outputDataType(),
                    preparedUnit.elementCount(),
                    "cpu1-fused-" + preparedUnit.outputNodeId()
            );
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            output = Cpu1TensorView.fromTensor(outputTensor);
        }

        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
                NativeTensorStorage nativeInput = context.requireNativeReadable(
                        inputNodeId,
                        CpuMaterializationReason.CPU_CONSUMER
                );
                inputs.add(Cpu1TensorView.fromNativeStorage(tensor, nativeInput)
                        .broadcastToShape(preparedUnit.outputShape()));
            } else {
                inputs.add(Cpu1TensorView.fromTensor(tensor)
                        .broadcastToShape(preparedUnit.outputShape()));
            }
        }

        Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(preparedUnit, inputs, output);
        preparedUnit.launchPolicy().launch(Cpu1FusedElementwiseLoops::computeRange, args);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.outputNodeId(), "cpu1 fused wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(preparedUnit.outputNodeId(), nativeOutput, "cpu1 fused wrote native CPU segment");
        }
    }
}
```

Tento kod predpoklada upravu launch interface ve Fazi 4.1, protoze dnes `Cpu1LaunchPolicy.launch(...)` bere `Cpu1ElementwiseRangeRunner` a `Cpu1KernelArgs`.

Proc:

- Runtime executable patri do `backend.cpu1.exec`.
- Vstupy/vystupy se binduji stejne jako u elementwise.
- Vystup native storage jde pres runtime slot cache, ne per-execute alokace mimo kontext.

## Faze 4: Scalar/Parallel Fused Runner

### Task 4.1: Zobecnit range launch bez rozbiti elementwise

Stav: `[ ]`

Problem:

`Cpu1LaunchPolicy` je dnes navazany na:

```java
void launch(Cpu1ElementwiseRangeRunner kernelRunner, Cpu1KernelArgs args);
```

Fused runner ma jine args. Nechceme kopirovat launch policy.

Zmena v:

```text
src/main/java/backend/cpu1/launch/Cpu1LaunchPolicy.java
```

Nahradit obsah:

```java
package backend.cpu1.launch;

@FunctionalInterface
public interface Cpu1RangeTask {
    void run(int startInclusive, int endExclusive);
}
```

Pozor: `Cpu1RangeTask` ma byt samostatny soubor, ne v `Cpu1LaunchPolicy.java`, protoze public top-level typ musi mit vlastni soubor.

Novy soubor:

```text
src/main/java/backend/cpu1/launch/Cpu1RangeTask.java
```

Kod:

```java
package backend.cpu1.launch;

@FunctionalInterface
public interface Cpu1RangeTask {
    void run(int startInclusive, int endExclusive);
}
```

Upravit `Cpu1LaunchPolicy.java` na:

```java
package backend.cpu1.launch;

public interface Cpu1LaunchPolicy {
    void launch(int elementCount, Cpu1RangeTask task);
}
```

Upravit `Cpu1SingleThreadLaunch.java`:

```java
@Override
public void launch(int elementCount, Cpu1RangeTask task) {
    if (task == null) {
            throw new IllegalArgumentException("task cannot be null");
        }
    task.run(0, elementCount);
}
```

Upravit `Cpu1ParallelLaunch.java`:

```java
@Override
public void launch(int elementCount, Cpu1RangeTask task) {
    if (task == null) {
            throw new IllegalArgumentException("task cannot be null");
        }
    Cpu1RangeLauncher.launch(elementCount, launchConfig, task::run);
}
```

Upravit `Cpu1ElementwiseExecutableUnit.run(...)`:

```java
preparedUnit.launchPolicy().launch(
        args.elementCount(),
        (start, end) -> preparedUnit.kernelRunner().computeRange(args, start, end)
);
```

Nutne pridat getter do `Cpu1PreparedElementwiseUnit`, pokud neexistuje:

```java
public Cpu1ElementwiseRangeRunner kernelRunner() {
    return kernelRunner;
}
```

Proc:

- Launch policy ma resit pouze rozdeleni rozsahu.
- Typ kernel args patri konkretni executable jednotce.
- Elementwise i fused pak pouzivaji stejne chunk/thread tuning.

Riziko:

- Toto je mala breaking zmena v cpu1 internim API.
- Overit vsechny call sites pres:

```bash
rg -n "launch\\(" src/main/java/backend/cpu1 src/test/java/backend/cpu1
```

### Task 4.2: Pridat `Cpu1FusedElementwiseRangeRunner`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseRangeRunner.java
```

Kod:

```java
package backend.cpu1.kernels.fused;

import backend.cpu1.exec.Cpu1FusedKernelArgs;

@FunctionalInterface
public interface Cpu1FusedElementwiseRangeRunner {
    void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive);
}
```

Proc:

- Konkretni fused loops maji vlastni args.
- Interface zustava jednoduchy a inlinovatelny.

### Task 4.3: Pridat `Cpu1FusedElementwiseLoops`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseLoops.java
```

Kod skeleton s plnou semantikou:

```java
package backend.cpu1.kernels.fused;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.fused.ir.Cpu1ScalarAttribute;
import operations.Operation;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class Cpu1FusedElementwiseLoops {
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;
    private static final ValueLayout.OfShort BF16 = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfByte BOOL = ValueLayout.JAVA_BYTE;

    private Cpu1FusedElementwiseLoops() {
    }

    public static void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        int nodeCount = args.preparedUnit().plan().nodeCount();
        double[] numericValues = new double[nodeCount];
        boolean[] boolValues = new boolean[nodeCount];
        for (int index = startInclusive; index < endExclusive; index++) {
            for (Cpu1FusedNodePlan node : args.preparedUnit().plan().nodes()) {
                if (node.outputType() == DataType.BOOL) {
                    boolValues[node.index()] = evalBool(args, node, numericValues, boolValues, index);
                } else {
                    numericValues[node.index()] = evalNumeric(args, node, numericValues, boolValues, index);
                }
            }
            Cpu1FusedNodePlan outputNode = args.preparedUnit().plan().outputNode();
            int outputIndex = outputStorageIndex(args, index);
            if (outputNode.outputType() == DataType.BOOL) {
                storeBool(args.output(), outputIndex, boolValues[outputNode.index()]);
            } else {
                storeNumeric(args.output(), outputIndex, numericValues[outputNode.index()]);
            }
        }
    }

    private static double evalNumeric(
            Cpu1FusedKernelArgs args,
            Cpu1FusedNodePlan node,
            double[] numericValues,
            boolean[] boolValues,
            int index
    ) {
        Operation.OpType op = node.opType();
        return switch (op) {
            case ADD -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    + numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case SUB -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    - numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case MUL -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    * numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case DIV -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    / numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case MIN -> Math.min(
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index),
                    numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index)
            );
            case MAX -> Math.max(
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index),
                    numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index)
            );
            case NEG -> -numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index);
            case INV -> 1.0d / numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index);
            case LOG -> Math.log(numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case EXP -> exp(args, numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case FAST_EXP -> fastExp(args, numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case TANH -> tanh(args, numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case FAST_TANH -> fastTanh(args, numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case POW -> Math.pow(
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index),
                    ((Cpu1ScalarAttribute) node.attributes()).f64()
            );
            case POW_TENSOR -> Math.pow(
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index),
                    numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index)
            );
            case SQRT -> Math.sqrt(numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case ABS -> Math.abs(numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case CONST_SCALAR -> ((Cpu1ScalarAttribute) node.attributes()).f64();
            case MUL_SCALAR -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    * ((Cpu1ScalarAttribute) node.attributes()).f64();
            case RELU -> Math.max(0.0d, numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index));
            case CLAMP_MIN -> Math.max(
                    ((Cpu1ScalarAttribute) node.attributes()).f64(),
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
            );
            case CLAMP_MAX -> Math.min(
                    ((Cpu1ScalarAttribute) node.attributes()).f64(),
                    numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
            );
            case SIGMOID -> {
                double value = numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index);
                yield 1.0d / (1.0d + Math.exp(-value));
            }
            case NOOP -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index);
            case WHERE -> boolRef(args, node.inputRefs().get(0), boolValues, index)
                    ? numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index)
                    : numericRef(args, node.inputRefs().get(2), numericValues, boolValues, index);
            default -> throw new UnsupportedOperationException("cpu1 fused numeric op not supported: " + op);
        };
    }

    private static boolean evalBool(
            Cpu1FusedKernelArgs args,
            Cpu1FusedNodePlan node,
            double[] numericValues,
            boolean[] boolValues,
            int index
    ) {
        Operation.OpType op = node.opType();
        return switch (op) {
            case GT -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    > numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case GE -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    >= numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case LT -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    < numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case LE -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    <= numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case EQ -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    == numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case NE -> numericRef(args, node.inputRefs().get(0), numericValues, boolValues, index)
                    != numericRef(args, node.inputRefs().get(1), numericValues, boolValues, index);
            case LOGICAL_AND -> boolRef(args, node.inputRefs().get(0), boolValues, index)
                    && boolRef(args, node.inputRefs().get(1), boolValues, index);
            case LOGICAL_OR -> boolRef(args, node.inputRefs().get(0), boolValues, index)
                    || boolRef(args, node.inputRefs().get(1), boolValues, index);
            case LOGICAL_NOT -> !boolRef(args, node.inputRefs().get(0), boolValues, index);
            case WHERE -> boolRef(args, node.inputRefs().get(0), boolValues, index)
                    ? boolRef(args, node.inputRefs().get(1), boolValues, index)
                    : boolRef(args, node.inputRefs().get(2), boolValues, index);
            default -> evalNumeric(args, node, numericValues, boolValues, index) != 0.0d;
        };
    }

    private static double numericRef(
            Cpu1FusedKernelArgs args,
            int ref,
            double[] numericValues,
            boolean[] boolValues,
            int index
    ) {
        int inputCount = args.preparedUnit().plan().inputCount();
        if (ref < inputCount) {
            return loadNumeric(args.input(ref), inputStorageIndex(args, ref, index));
        }
        int nodeIndex = ref - inputCount;
        Cpu1FusedNodePlan node = args.preparedUnit().plan().nodes().get(nodeIndex);
        return node.outputType() == DataType.BOOL
                ? (boolValues[nodeIndex] ? 1.0d : 0.0d)
                : numericValues[nodeIndex];
    }

    private static boolean boolRef(
            Cpu1FusedKernelArgs args,
            int ref,
            boolean[] boolValues,
            int index
    ) {
        int inputCount = args.preparedUnit().plan().inputCount();
        if (ref < inputCount) {
            Cpu1TensorView input = args.input(ref);
            if (input.dataType() != DataType.BOOL) {
                throw new UnsupportedOperationException("cpu1 fused bool ref must use BOOL external input.");
            }
            return loadBool(input, inputStorageIndex(args, ref, index));
        }
        int nodeIndex = ref - inputCount;
        if (args.preparedUnit().plan().nodes().get(nodeIndex).outputType() != DataType.BOOL) {
            throw new UnsupportedOperationException("cpu1 fused bool ref points to non-BOOL node.");
        }
        return boolValues[nodeIndex];
    }

    private static int inputStorageIndex(Cpu1FusedKernelArgs args, int inputIndex, int logicalIndex) {
        Cpu1FusedInputPlan plan = args.preparedUnit().plan().inputs().get(inputIndex);
        if (plan.isLinearAccess()) {
            return args.input(inputIndex).storageOffset() + logicalIndex;
        }
        int storageIndex = args.input(inputIndex).storageOffset();
        int remaining = logicalIndex;
        int[] dense = plan.logicalOutputDenseStrides();
        int[] strides = args.input(inputIndex).strides();
        for (int dim = 0; dim < dense.length; dim++) {
            int coord = dense[dim] == 0 ? 0 : remaining / dense[dim];
            remaining = dense[dim] == 0 ? remaining : remaining % dense[dim];
            storageIndex += coord * strides[dim];
        }
        return storageIndex;
    }

    private static int outputStorageIndex(Cpu1FusedKernelArgs args, int logicalIndex) {
        Cpu1TensorView output = args.output();
        if (output.contiguous()) {
            return output.storageOffset() + logicalIndex;
        }
        int storageIndex = output.storageOffset();
        int remaining = logicalIndex;
        int[] dense = tensor.TensorMetadata.computeStrides(output.shape());
        int[] strides = output.strides();
        for (int dim = 0; dim < dense.length; dim++) {
            int coord = dense[dim] == 0 ? 0 : remaining / dense[dim];
            remaining = dense[dim] == 0 ? remaining : remaining % dense[dim];
            storageIndex += coord * strides[dim];
        }
        return storageIndex;
    }

    private static double loadNumeric(Cpu1TensorView view, int storageIndex) {
        return switch (view.storageKind()) {
            case JAVA_ARRAY -> loadArrayNumeric(view, storageIndex);
            case MEMORY_SEGMENT -> loadSegmentNumeric(view, storageIndex);
        };
    }

    private static double loadArrayNumeric(Cpu1TensorView view, int storageIndex) {
        return switch (view.dataType()) {
            case FLOAT64 -> view.float64Array()[storageIndex];
            case FLOAT32 -> view.float32Array()[storageIndex];
            case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(view.bfloat16Array()[storageIndex]);
            case BOOL -> view.boolArray()[storageIndex] == 0 ? 0.0d : 1.0d;
            case INT32 -> view.int32Array()[storageIndex];
            case INT64 -> view.int64Array()[storageIndex];
        };
    }

    private static double loadSegmentNumeric(Cpu1TensorView view, int storageIndex) {
        MemorySegment segment = view.segment();
        return switch (view.dataType()) {
            case FLOAT64 -> segment.get(F64, (long) storageIndex * Double.BYTES);
            case FLOAT32 -> segment.get(F32, (long) storageIndex * Float.BYTES);
            case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(segment.get(BF16, (long) storageIndex * Short.BYTES));
            case BOOL -> segment.get(BOOL, storageIndex) == 0 ? 0.0d : 1.0d;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 fused segment INT storage is not supported.");
        };
    }

    private static boolean loadBool(Cpu1TensorView view, int storageIndex) {
        return switch (view.storageKind()) {
            case JAVA_ARRAY -> view.boolArray()[storageIndex] != 0;
            case MEMORY_SEGMENT -> view.segment().get(BOOL, storageIndex) != 0;
        };
    }

    private static void storeNumeric(Cpu1TensorView output, int storageIndex, double value) {
        switch (output.storageKind()) {
            case JAVA_ARRAY -> storeArrayNumeric(output, storageIndex, value);
            case MEMORY_SEGMENT -> storeSegmentNumeric(output, storageIndex, value);
        }
    }

    private static void storeArrayNumeric(Cpu1TensorView output, int storageIndex, double value) {
        switch (output.dataType()) {
            case FLOAT64 -> output.float64Array()[storageIndex] = value;
            case FLOAT32 -> output.float32Array()[storageIndex] = (float) value;
            case BFLOAT16 -> output.bfloat16Array()[storageIndex] = TensorDTypeOps.toBFloat16Bits((float) value);
            case BOOL -> output.boolArray()[storageIndex] = value == 0.0d ? (byte) 0 : (byte) 1;
            case INT32 -> output.int32Array()[storageIndex] = (int) value;
            case INT64 -> output.int64Array()[storageIndex] = (long) value;
        }
    }

    private static void storeSegmentNumeric(Cpu1TensorView output, int storageIndex, double value) {
        MemorySegment segment = output.segment();
        switch (output.dataType()) {
            case FLOAT64 -> segment.set(F64, (long) storageIndex * Double.BYTES, value);
            case FLOAT32 -> segment.set(F32, (long) storageIndex * Float.BYTES, (float) value);
            case BFLOAT16 -> segment.set(BF16, (long) storageIndex * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
            case BOOL -> segment.set(BOOL, storageIndex, value == 0.0d ? (byte) 0 : (byte) 1);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 fused segment INT storage is not supported.");
        }
    }

    private static void storeBool(Cpu1TensorView output, int storageIndex, boolean value) {
        switch (output.storageKind()) {
            case JAVA_ARRAY -> output.boolArray()[storageIndex] = value ? (byte) 1 : (byte) 0;
            case MEMORY_SEGMENT -> output.segment().set(BOOL, storageIndex, value ? (byte) 1 : (byte) 0);
        }
    }

    private static double exp(Cpu1FusedKernelArgs args, double value) {
        return args.preparedUnit().approximateExp() ? fastExp(args, value) : Math.exp(value);
    }

    private static double fastExp(Cpu1FusedKernelArgs args, double value) {
        return args.preparedUnit().outputDataType() == DataType.FLOAT64
                ? FastTranscendentals.fastExpF64(value)
                : FastTranscendentals.fastExpF32((float) value);
    }

    private static double tanh(Cpu1FusedKernelArgs args, double value) {
        return args.preparedUnit().approximateTanh() ? fastTanh(args, value) : Math.tanh(value);
    }

    private static double fastTanh(Cpu1FusedKernelArgs args, double value) {
        return args.preparedUnit().outputDataType() == DataType.FLOAT64
                ? FastTranscendentals.fastTanhF64(value)
                : FastTranscendentals.fastTanhF32((float) value);
    }
}
```

Proc:

- Toto je scalar interpreted fused cesta.
- Stale fuzuje pametove pruchody: internal nodes nevytvari mezibuffery.
- Neni to finalni maximalne rychla hot path, ale je to cisty cpu1 kontrakt.

Vykonova poznamka:

- `double[] numericValues` a `boolean[] boolValues` se alokuji per chunk/range call. Pro prvni korektni implementaci je to akceptovatelne.
- Faze 8 musi tyto male arrays presunout do `Cpu1ScratchBuffer`.
- Nezapinat jako default nahradu stareho ASM fused pro vykonove kriticke benchmarky bez mereni z Faze 11.

### Task 4.4: Upravit `Cpu1FusedElementwiseExecutableUnit` launch call

Stav: `[ ]`

Po uprave launch policy z Tasku 4.1 ma kod v `run(...)` vypadat takto:

```java
Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(preparedUnit, inputs, output);
preparedUnit.launchPolicy().launch(
        args.elementCount(),
        (start, end) -> Cpu1FusedElementwiseLoops.computeRange(args, start, end)
);
```

Proc:

- Fused executable pouziva stejne chunking a worker policy jako elementwise.
- Hot path nema execute-time registry lookup.

## Faze 5: Artifact A Trace Integrace

### Task 5.1: Upravit `Cpu1PreparedArtifact`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java
```

Pridat import:

```java
import backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnit;
```

Pridat field:

```java
private final Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit;
```

Ve vsech existujicich konstruktorech nastavit novy field na `null`.

Pridat constructor:

```java
public Cpu1PreparedArtifact(Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit) {
    if (preparedFusedElementwiseUnit == null) {
        throw new IllegalArgumentException("preparedFusedElementwiseUnit cannot be null");
    }
    this.preparedUnit = null;
    this.preparedLayoutUnit = null;
    this.preparedReductionUnit = null;
    this.preparedMatmulUnit = null;
    this.preparedMseLossUnit = null;
    this.preparedFusedElementwiseUnit = preparedFusedElementwiseUnit;
    this.executableUnit = new Cpu1FusedElementwiseExecutableUnit(preparedFusedElementwiseUnit);
}
```

Pridat accessor:

```java
public Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit() {
    if (preparedFusedElementwiseUnit == null) {
        throw new IllegalStateException("This cpu1 artifact does not expose a prepared fused elementwise unit");
    }
    return preparedFusedElementwiseUnit;
}
```

Upravit generic executable constructor:

```java
public Cpu1PreparedArtifact(Cpu1ExecutableUnit executableUnit) {
    this.preparedUnit = null;
    this.preparedLayoutUnit = null;
    this.preparedReductionUnit = null;
    this.preparedMatmulUnit = null;
    this.preparedMseLossUnit = null;
    this.preparedFusedElementwiseUnit = null;
    if (executableUnit == null) {
        throw new IllegalArgumentException("executableUnit cannot be null");
    }
    this.executableUnit = executableUnit;
}
```

Upravit traceContribution call:

```java
return Cpu1TraceContributor.traceContribution(
        node,
        preparedLayoutUnit,
        preparedReductionUnit,
        preparedMatmulUnit,
        preparedMseLossUnit,
        preparedFusedElementwiseUnit
);
```

Proc:

- `Cpu1PreparedArtifact` je jednotny wrapper pro cpu1 runtime artifacts.
- Fused je dalsi first-class prepared unit, ne specialni vyjimka mimo model.

### Task 5.2: Upravit `Cpu1TraceContributor`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java
```

Pridat import:

```java
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
```

Upravit signaturu:

```java
public static StepTraceContribution traceContribution(
        CompiledNode node,
        Cpu1PreparedLayoutUnit preparedLayoutUnit,
        Cpu1PreparedReductionUnit preparedReductionUnit,
        Cpu1PreparedMatmulUnit preparedMatmulUnit,
        Cpu1PreparedMseLossUnit preparedMseLossUnit,
        Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit
) {
    if (preparedLayoutUnit != null) {
        return layoutTrace(node, preparedLayoutUnit);
    }
    if (preparedReductionUnit != null) {
        return reductionTrace(preparedReductionUnit);
    }
    if (preparedMatmulUnit != null) {
        return matmulTrace(preparedMatmulUnit);
    }
    if (preparedMseLossUnit != null) {
        return mseLossTrace(preparedMseLossUnit);
    }
    if (preparedFusedElementwiseUnit != null) {
        return fusedElementwiseTrace(preparedFusedElementwiseUnit);
    }
    return StepTraceContribution.empty();
}
```

Pridat helper:

```java
private static StepTraceContribution fusedElementwiseTrace(Cpu1PreparedFusedElementwiseUnit unit) {
    LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
    attrs.put("cpu1KernelId", "CPU1_FUSED_ELEMENTWISE");
    attrs.put("cpu1FusedNodeCount", unit.plan().nodeCount());
    attrs.put("cpu1FusedInputCount", unit.plan().inputCount());
    attrs.put("cpu1StorageKind", unit.storageKind().name());
    attrs.put("cpu1LayoutKind", unit.layoutKind().name());
    attrs.put("cpu1FusedOutputNodeId", unit.outputNodeId());
    attrs.put("cpu1FusedElementCount", unit.elementCount());
    attrs.put("cpu1FusedLaunchWorkers", unit.launchConfig().workerCount());
    attrs.put("cpu1FusedLaunchChunkSize", unit.launchConfig().chunkSize());
    attrs.put("cpu1FusedCostClass", unit.dispatchDecision().costClass().name());
    attrs.put("cpu1FusedRequestedVectorization", unit.dispatchDecision().requestedVectorizationKind().name());
    attrs.put("cpu1FusedApproxExp", unit.approximateExp());
    attrs.put("cpu1FusedApproxTanh", unit.approximateTanh());
    DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
            unit.dispatchDecision().requestedVectorizationKind().name(),
            unit.dispatchDecision().requestedVectorizationKind() == Cpu1VectorizationKind.VECTOR
                    ? vectorWidth(unit.outputDataType())
                    : 1,
            unit.launchConfig().workerCount(),
            unit.launchConfig().chunkSize(),
            unit.launchConfig().chunkSize()
    );
    return new StepTraceContribution(
            "CPU1_FUSED_ELEMENTWISE",
            attrs,
            null,
            null,
            dispatch,
            null,
            null,
            null,
            null
    );
}
```

Proc:

- Fallback/route musi byt videt v trace.
- Trace musi rozlisit scalar/vector/codegen route, storage kind, worker count a fused node count.
- Finalni implementace nesmi nechavat vector trace jako docasnou hodnotu `1`, pokud dispatch zvolil vector route.

## Faze 6: Prepare Dispatcher Integrace

### Task 6.1: Pridat field do `BackendPrepareDispatcher`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/prepare/BackendPrepareDispatcher.java
```

Pridat import:

```java
import backend.cpu1.prepare.Cpu1FusedElementwisePreparer;
```

Pridat field:

```java
private final Cpu1FusedElementwisePreparer cpu1FusedElementwisePreparer;
```

V constructoru:

```java
this.cpu1FusedElementwisePreparer = new Cpu1FusedElementwisePreparer(runtimeConfig);
```

Proc:

- Dispatcher uz vlastni cpu1 MSE a matmul specializovane preparery.
- Fused elementwise je dalsi prepared region step.

### Task 6.2: Pridat runtime route knob do `FusedExecutionPolicy`

Stav: `[ ]`

Soubor:

```text
src/main/java/config/runtime/FusedExecutionPolicy.java
```

Nahradit record:

```java
package config.runtime;

/**
 * Runtime policy for fused elementwise execution.
 *
 * @param allowBackendFallback whether fallback execution is allowed when codegen execution cannot run a fused region
 * @param useCpu1Elementwise whether CPU fused elementwise regions should prepare through the cpu1 fused path
 */
public record FusedExecutionPolicy(
        boolean allowBackendFallback,
        boolean useCpu1Elementwise
) {
    public FusedExecutionPolicy(boolean allowBackendFallback) {
        this(allowBackendFallback, false);
    }

    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(true, false);
    }

    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(true, false);
    }

    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(value, useCpu1Elementwise);
    }

    public FusedExecutionPolicy withUseCpu1Elementwise(boolean value) {
        return new FusedExecutionPolicy(allowBackendFallback, value);
    }
}
```

Proc:

- Route musi byt explicitni runtime volba, ne skryte prepnuti.
- Konstruktor `FusedExecutionPolicy(boolean)` zachova existujici call sites bez adapter vrstvy.
- Default zustava stary CPU fused, dokud profil/test explicitne nezvoli cpu1.

### Task 6.3: Zapojit route do `BackendPrepareDispatcher`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/prepare/BackendPrepareDispatcher.java
```

Upravit `prepareCpuFusedStep(...)`:

```java
public CompiledNodeExecutionMetadata prepareCpuFusedStep(
        CompiledNode outputNode,
        LoweredExecutionUnit loweredUnit,
        BackendPrepareContext context
) {
    if (outputNode == null) {
            throw new IllegalArgumentException("outputNode cannot be null");
        }
    if (loweredUnit == null) {
            throw new IllegalArgumentException("loweredUnit cannot be null");
        }
    if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
    if (runtimeConfig.fused().useCpu1Elementwise()) {
        return cpu1FusedElementwisePreparer.prepare(outputNode, loweredUnit, context);
    }
    return cpuPreparer.prepareLoweredFusedStep(outputNode, loweredUnit, context);
}
```

Proc:

- Finalni plan ma realne zapojeni, ne direct-only preparer test.
- Default zustava kompatibilni.
- cpu1 fused lze zapnout benchmarkem/profilem bez zmeny graph optimizeru.

### Task 6.4: Doplnit profile IO route knob

Stav: `[ ]`

Soubor:

```text
src/main/java/config/profile/ExecutionProfileIO.java
```

Pri cteni runtime fused configu doplnit:

```java
FusedExecutionPolicy fused = new FusedExecutionPolicy(
        findBoolean(json, "fusedAllowBackendFallback", defaultProfile.runtime().fused().allowBackendFallback()),
        findBoolean(json, "fusedUseCpu1Elementwise", defaultProfile.runtime().fused().useCpu1Elementwise())
);
```

Pri zapisu profilu doplnit:

```java
"      \"fusedUseCpu1Elementwise\": " + fused.useCpu1Elementwise() + "\n"
```

Presne umisteni musi respektovat soucasny JSON formatting v `ExecutionProfileIO`.

Proc:

- Tuning/profile system musi umet cpu1 fused route reprodukovat.
- Bez toho by benchmark mohl pouzit jinou route nez produkcni prepare.

## Faze 7: Test Coverage

### Task 7.1: Pridat IR builder testy

Stav: `[ ]`

Novy soubor:

```text
src/test/java/backend/cpu1/fused/Cpu1FusedIrBuilderTest.java
```

Testy:

```java
package backend.cpu1.fused;

import backend.cpu1.fused.ir.Cpu1FusedIrBuilder;
import operations.Operation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1FusedIrBuilderTest {
    @Test
    void canonicalizesPowTwoToMul() {
        // Vytvorit graph: y = x.pow(2).relu()
        // Compile, najit fused ordered node ids, build plan.
        // Assert: prvni pow node v planu ma opType MUL a oba inputRefs jsou stejne.
    }

    @Test
    void buildsBroadcastEffectiveStrides() {
        // Vytvorit graph: y = a([N,M]) + b([M])
        // Build fused plan.
        // Assert: b input ma zero stride v outer dim a stride 1 v inner dim.
    }
}
```

Proc:

- Nejcastejsi bug bude spatne ref mapovani nebo broadcast stride.
- Testovat IR oddelene od execution zkrati debug.

### Task 7.2: Pridat cpu1 fused execution contract test

Stav: `[ ]`

Novy soubor:

```text
src/test/java/backend/cpu1/Cpu1FusedElementwiseExecutionContractTest.java
```

Zakladni pattern:

```java
package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.prepare.Cpu1FusedElementwisePreparer;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.lowering.LoweredExecutionUnit;
import backend.prepare.BackendPrepareContext;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecution;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class Cpu1FusedElementwiseExecutionContractTest {
    @Test
    void fusedReluMulAddMatchesUnfusedF32Array() {
        // a, b, c
        // y = a.mul(b).add(c).relu()
        // compile with fusion enabled
        // prepare fused unit through Cpu1FusedElementwisePreparer
        // execute metadata artifact
        // compare against unfused Java computation
    }

    @Test
    void fusedBroadcastWhereMatchesUnfusedF32Array() {
        // mask bool, x [N,M], bias [M]
        // y = where(mask, x.add(bias), x.mul(-1))
        // compare values
    }

    @Test
    void fusedBf16MatchesWithinTolerance() {
        // BF16 inputs, chain add/mul/relu
        // compare with tolerance reflecting BF16 store
    }

    @Test
    void fusedNativeSegmentMatchesArrayPath() {
        // runtime config CPU_NATIVE
        // ensure artifact is Cpu1PreparedArtifact
        // execute and compare
    }
}
```

Proc:

- Testy musi overit skutecne execution, ne jen prepare.
- Minimalni sada: F32, F64, BF16, BOOL/WHERE, broadcast, native segment.

### Task 7.3: Pridat route test pro config knob

Stav: `[ ]`

Pridat test:

```text
src/test/java/backend/prepare/Cpu1FusedPrepareRoutingTest.java
```

Test:

```java
@Test
void cpu1FusedRouteProducesCpu1PreparedArtifactWhenEnabled() {
    RuntimeConfig config = RuntimeConfig.inferenceDefaults(DataType.FLOAT32)
            .withFused(RuntimeConfig.inferenceDefaults(DataType.FLOAT32).fused().withUseCpu1Elementwise(true));
    PreparedExecution prepared = graph.compile().prepare(config);
    assertTrue(prepared.forwardSteps().stream()
            .anyMatch(step -> step.metadata().artifact() instanceof Cpu1PreparedArtifact artifact
                    && artifact.preparedFusedElementwiseUnit().plan().nodeCount() > 1));
}
```

Pred napsanim overit, zda `RuntimeConfig` ma `withFused(...)`. Pokud nema, pouzit existujici constructor pattern nebo pridat maly immutable `withFused(...)` helper jako soucast teto faze.

Proc:

- Route test je povinna cast kompletni migrace.
- Overuje, ze cpu1 fused neni jen izolovany preparer, ale skutecne zapojitelna prepare route.

## Faze 8: Scratch Buffer Pro Fused Temporaries

### Task 8.1: Rozsirit `Cpu1ScratchBufferSpec` o fused scalar slots

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1ScratchBufferSpec.java
```

Cilovy API doplnek:

```java
public static Cpu1ScratchBufferSpec fusedElementwise(int numericSlots, int boolSlots) {
    return new Cpu1ScratchBufferSpec(
            0,
            0,
            0,
            Math.max(0, numericSlots),
            Math.max(0, boolSlots)
    );
}
```

Pokud aktualni record nema obecne sloty, rozsirit ho explicitne:

```java
private final int fusedNumericSlots;
private final int fusedBoolSlots;
```

Proc:

- `double[] numericValues` a `boolean[] boolValues` nesmi zustat per-range alokace v hot path.
- Scratch buffer je cpu1 pametovy kontrakt pro docasnou pamet.

### Task 8.2: Rozsirit `Cpu1ScratchBuffer`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1ScratchBuffer.java
```

Doplnit pole:

```java
private final double[] fusedNumeric;
private final boolean[] fusedBool;
```

Doplnit accessors:

```java
public double[] fusedNumeric() {
    return fusedNumeric;
}

public boolean[] fusedBool() {
    return fusedBool;
}
```

Pri `allocate(spec)` alokovat:

```java
double[] fusedNumeric = spec.fusedNumericSlots() == 0 ? null : new double[spec.fusedNumericSlots()];
boolean[] fusedBool = spec.fusedBoolSlots() == 0 ? null : new boolean[spec.fusedBoolSlots()];
```

Proc:

- Fused evaluator potrebuje male per-worker temporaries.
- Alokace patri do prepared runtime state, ne do kazdeho chunku.

### Task 8.3: Napojit scratch do `Cpu1FusedKernelArgs`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedKernelArgs.java
```

Upravit konstruktor:

```java
private final Cpu1ScratchBuffer scratchBuffer;

public Cpu1FusedKernelArgs(
        Cpu1PreparedFusedElementwiseUnit preparedUnit,
        List<Cpu1TensorView> inputs,
        Cpu1TensorView output,
        Cpu1ScratchBuffer scratchBuffer
) {
    this.scratchBuffer = scratchBuffer;
    ...
}
```

Pridat:

```java
public double[] fusedNumericScratch() {
    if (scratchBuffer == null || scratchBuffer.fusedNumeric() == null) {
        throw new IllegalStateException("cpu1 fused numeric scratch buffer is not allocated.");
    }
    return scratchBuffer.fusedNumeric();
}

public boolean[] fusedBoolScratch() {
    if (scratchBuffer == null || scratchBuffer.fusedBool() == null) {
        throw new IllegalStateException("cpu1 fused bool scratch buffer is not allocated.");
    }
    return scratchBuffer.fusedBool();
}
```

Proc:

- Kernel smycka dostane prepared scratch bez globalniho stavu.

### Task 8.4: `Cpu1FusedElementwiseExecutableUnit.scratchBufferSpec()`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java
```

Nahradit:

```java
@Override
public Cpu1ScratchBufferSpec scratchBufferSpec() {
    return Cpu1ScratchBufferSpec.none();
}
```

za:

```java
@Override
public Cpu1ScratchBufferSpec scratchBufferSpec() {
    int workerCount = Math.max(1, preparedUnit.launchConfig().workerCount());
    int nodeCount = preparedUnit.plan().nodeCount();
    return Cpu1ScratchBufferSpec.fusedElementwise(
            workerCount * nodeCount,
            workerCount * nodeCount
    );
}
```

V `run(...)` predat scratch:

```java
Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(
        preparedUnit,
        inputs,
        output,
        context.cpu1ScratchBufferForNodeId(preparedUnit.outputNodeId())
);
```

Proc:

- Scratch je alokovany pri prepare/runtime state allocation stejne jako ostatni cpu1 scratch.
- `outputNodeId` je runtime step owner.

### Task 8.5: Pouzit per-worker scratch slice ve fused loops

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseLoops.java
```

Nahradit per-range alokaci:

```java
double[] numericValues = new double[nodeCount];
boolean[] boolValues = new boolean[nodeCount];
```

za:

```java
double[] numericScratch = args.fusedNumericScratch();
boolean[] boolScratch = args.fusedBoolScratch();
int workerSlot = Cpu1RangeLauncher.currentWorkerSlot();
int base = workerSlot * nodeCount;
```

Pokud `Cpu1RangeLauncher` dnes nema `currentWorkerSlot()`, doplnit do launcheru thread-local slot:

```java
private static final ThreadLocal<Integer> CURRENT_WORKER_SLOT = ThreadLocal.withInitial(() -> 0);

public static int currentWorkerSlot() {
    return CURRENT_WORKER_SLOT.get();
}
```

Pri spousteni chunku nastavit slot:

```java
CURRENT_WORKER_SLOT.set(workerIndex);
try {
    task.run(start, end);
} finally {
    CURRENT_WORKER_SLOT.remove();
}
```

Ve fused evaluatoru pak pristupovat pres `base + node.index()`.

Proc:

- Parallel chunks nesmi sdilet stejny scratch index.
- Nechceme alokovat temporaries pro kazdy chunk.

## Faze 9: Fused Vector/Codegen Hot Path

### Task 9.1: Pridat fused dispatch runner

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseDispatch.java
```

Kod:

```java
package backend.cpu1.kernels.fused;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.fused.scalar.Cpu1FusedScalarInterpreter;
import backend.cpu1.kernels.fused.vector.Cpu1FusedVectorDispatch;

public final class Cpu1FusedElementwiseDispatch {
    private Cpu1FusedElementwiseDispatch() {
    }

    public static void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        if (canUseVector(args)) {
            Cpu1FusedVectorDispatch.computeRange(args, startInclusive, endExclusive);
            return;
        }
        Cpu1FusedScalarInterpreter.computeRange(args, startInclusive, endExclusive);
    }

    private static boolean canUseVector(Cpu1FusedKernelArgs args) {
        return args.preparedUnit().dispatchDecision().requestedVectorizationKind() == Cpu1VectorizationKind.VECTOR
                && args.preparedUnit().layoutKind() == Cpu1LayoutKind.CONTIGUOUS
                && args.preparedUnit().plan().usesOnlyLinearInputs();
    }
}
```

Proc:

- Execute stale vola jeden prepared runner.
- Dispatch check je trivialni a pripraveny z prepare metadata.
- Scalar fallback je uvnitr cpu1 fused path, ne stary backend fallback.

### Task 9.2: Presunout scalar interpreter do vlastni tridy

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/scalar/Cpu1FusedScalarInterpreter.java
```

Presunout obsah z `Cpu1FusedElementwiseLoops.computeRange(...)` do:

```java
public final class Cpu1FusedScalarInterpreter {
    private Cpu1FusedScalarInterpreter() {
    }

    public static void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        ...
    }
}
```

`Cpu1FusedElementwiseLoops` ponechat jako compatibility-free delegat pouze pokud jmeno pouzivaji testy; jinak ho nahradit `Cpu1FusedElementwiseDispatch`.

Proc:

- Scalar a vector route nesmi byt smichane v jedne velke tride.
- Organizace odpovida `elementwise` balicku.

### Task 9.3: Pridat vector interpreter kontrakt

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorPlan.java
```

Kod:

```java
package backend.cpu1.kernels.fused.vector;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import tensor.DataType;

public record Cpu1FusedVectorPlan(
        Cpu1FusedExpressionPlan expressionPlan,
        DataType laneType,
        int vectorWidth
) {
}
```

Proc:

- Vector route ma vlastni explicitni plan.
- Plan slouzi i jako codegen kernel cache key.

### Task 9.4: Implementovat `Cpu1FusedVectorInterpreter`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorInterpreter.java
```

Kod:

```java
package backend.cpu1.kernels.fused.vector;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.kernels.fused.scalar.Cpu1FusedScalarInterpreter;

public final class Cpu1FusedVectorInterpreter {
    private Cpu1FusedVectorInterpreter() {
    }

    public static void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        switch (args.preparedUnit().outputDataType()) {
            case FLOAT32 -> computeF32(args, startInclusive, endExclusive);
            case FLOAT64 -> computeF64(args, startInclusive, endExclusive);
            default -> Cpu1FusedScalarInterpreter.computeRange(args, startInclusive, endExclusive);
        }
    }

    private static void computeF32(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        // Implementovat Vector API loop pro podporovane op subsety.
        // Pokud plan obsahuje nepodporovany vector op, spadnout na scalar interpreter pro cely range.
        Cpu1FusedScalarInterpreter.computeRange(args, startInclusive, endExclusive);
    }

    private static void computeF64(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        // Implementovat Vector API loop pro podporovane op subsety.
        // Pokud plan obsahuje nepodporovany vector op, spadnout na scalar interpreter pro cely range.
        Cpu1FusedScalarInterpreter.computeRange(args, startInclusive, endExclusive);
    }
}
```

V ramci Faze 9 musi byt doplnena realna Vector API implementace aspon pro tyto op:

- `ADD`
- `SUB`
- `MUL`
- `DIV`
- `MIN`
- `MAX`
- `NEG`
- `ABS`
- `RELU`
- `CLAMP_MIN`
- `CLAMP_MAX`
- `NOOP`
- `MUL_SCALAR`
- `WHERE` s bool maskou pro F32/F64

Scalar atributy ve vector/generate hot path:

- F32 vector cesta pouziva `((Cpu1ScalarAttribute) node.attributes()).f32()`.
- BF16 compute cesta pouziva take `f32()`, protoze BF16 se pocita pres F32 a materializuje se az pri store.
- F64 vector cesta pouziva `((Cpu1ScalarAttribute) node.attributes()).f64()`.
- Scalar interpreter muze zustat na `f64()`, protoze vraci `double`, ale specializovane F32/BF16 hot paths nesmi zbytecne castovat z double v kazde iteraci.

Povinna Vector API smycka pro F32 contiguous:

```java
private void computeF32(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
    if (!supportsVector(args.preparedUnit().plan())) {
        Cpu1FusedScalarInterpreter.computeRange(args, startInclusive, endExclusive);
        return;
    }
    int i = startInclusive;
    int upper = FloatVector.SPECIES_PREFERRED.loopBound(endExclusive);
    for (; i < upper; i += FloatVector.SPECIES_PREFERRED.length()) {
        // 1. load external input vectors by ref
        // 2. evaluate nodes into local FloatVector[] slots
        // 3. store output vector
    }
    if (i < endExclusive) {
        Cpu1FusedScalarInterpreter.computeRange(args, i, endExclusive);
    }
}
```

Proc:

- Complete migration nesmi koncit u per-element switch interpreteru.
- Vector route je nutna pro paritu s duchem stareho ASM fused.
- Fallback uvnitr vector interpreteru je povoleny jen pro nepodporovany vector op subset; trace musi reportovat fallback reason.

### Task 9.5: Implementovat `Cpu1FusedVectorDispatch`

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorDispatch.java
```

Kod:

```java
package backend.cpu1.kernels.fused.vector;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernelFactory;

public final class Cpu1FusedVectorDispatch {
    private Cpu1FusedVectorDispatch() {
    }

    public static void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        if (Cpu1FusedCodegenKernelFactory.supports(args.preparedUnit())) {
            Cpu1FusedCodegenKernelFactory.kernelFor(args.preparedUnit())
                    .computeRange(args, startInclusive, endExclusive);
            return;
        }
        Cpu1FusedVectorInterpreter.computeRange(args, startInclusive, endExclusive);
    }
}
```

Proc:

- `vector` balicek zustava vector-interpreter/dispatch vrstva.
- Codegen kernel je oddeleny do `codegen/`, ne schovany pod vector nazvem.

### Task 9.6: Pridat codegen kernel kontrakt

Stav: `[ ]`

Nove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelCacheKey.java
```

Minimalni kontrakt:

```java
package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.exec.Cpu1FusedKernelArgs;

public interface Cpu1FusedCodegenKernel {
    void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive);
}
```

Factory skeleton:

```java
package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class Cpu1FusedCodegenKernelFactory {
    private static final Map<Cpu1FusedCodegenKernelCacheKey, Cpu1FusedCodegenKernel> CACHE = new ConcurrentHashMap<>();

    private Cpu1FusedCodegenKernelFactory() {
    }

    public static boolean supports(Cpu1PreparedFusedElementwiseUnit unit) {
        return unit != null
                && unit.plan().usesOnlyLinearInputs()
                && unit.vectorFallbackReason() == Cpu1FusedVectorFallbackReason.NONE;
    }

    public static Cpu1FusedCodegenKernel kernelFor(Cpu1PreparedFusedElementwiseUnit unit) {
        if (unit == null) {
            throw new IllegalArgumentException("unit cannot be null");
        }
        return CACHE.computeIfAbsent(Cpu1FusedCodegenKernelCacheKey.from(unit), key -> generate(unit));
    }

    private static Cpu1FusedCodegenKernel generate(Cpu1PreparedFusedElementwiseUnit unit) {
        // Implementace muze byt ASM nebo jina cpu1-native codegen cesta.
        // Nesmí importovat stary backend.cpu.fused.asm jako compatibility layer.
        throw new UnsupportedOperationException("cpu1 fused codegen kernel generation is not implemented.");
    }
}
```

Cache key skeleton:

```java
package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;

public record Cpu1FusedCodegenKernelCacheKey(
        String dtype,
        String storageKind,
        String layoutKind,
        String expressionSignature
) {
    public static Cpu1FusedCodegenKernelCacheKey from(Cpu1PreparedFusedElementwiseUnit unit) {
        if (unit == null) {
            throw new IllegalArgumentException("unit cannot be null");
        }
        return new Cpu1FusedCodegenKernelCacheKey(
                unit.outputDataType().name(),
                unit.storageKind().name(),
                unit.layoutKind().name(),
                unit.plan().toString()
        );
    }
}
```

Proc:

- Codegen route ma vlastni jmeno a vlastni cache key.
- `codegen/` reprezentuje runtime cestu, ktera kernel vytvari/generuje a pote ho spousti bez interpretace IR v hot path.
- `vector/` reprezentuje Vector API interpreter/dispatch nad fused IR.

### Task 9.7: Pridat vector fallback reason do prepared unit a trace

Stav: `[ ]`

Pridat enum:

```text
src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorFallbackReason.java
```

Kod:

```java
package backend.cpu1.kernels.fused.vector;

public enum Cpu1FusedVectorFallbackReason {
    NONE,
    NON_CONTIGUOUS,
    BROADCAST_STRIDED,
    UNSUPPORTED_DTYPE,
    UNSUPPORTED_OP,
    BOOL_OUTPUT,
    BF16_OUTPUT
}
```

Do `Cpu1PreparedFusedElementwiseUnit` pridat field:

```java
private final Cpu1FusedVectorFallbackReason vectorFallbackReason;
```

Trace attrs:

```java
attrs.put("cpu1FusedVectorFallbackReason", unit.vectorFallbackReason().name());
```

Proc:

- Stary CPU fused trace uz mel vector fallback reason.
- cpu1 musi byt stejne vysvetlitelny.

## Faze 10: Profile IO A Tuning Knobs

### Task 10.1: Pouzit existujici fused thresholdy z `CpuKernelConfig`

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java
```

Pro fused elementwise nepouzivat obecne cheap/transcendental thresholdy, ale fused-specific hodnoty:

```java
private static int fusedVectorMinSize(
        CpuKernelConfig cpuKernelConfig,
        Cpu1CostClass costClass
) {
    return costClass == Cpu1CostClass.EXPENSIVE_ELEMENTWISE
            ? cpuKernelConfig.fusedTranscendentalVectorMinSize()
            : cpuKernelConfig.fusedCheapVectorMinSize();
}

private static int fusedParallelMinSize(
        CpuKernelConfig cpuKernelConfig,
        Cpu1CostClass costClass
) {
    return costClass == Cpu1CostClass.EXPENSIVE_ELEMENTWISE
            ? cpuKernelConfig.fusedTranscendentalParallelMinSize()
            : cpuKernelConfig.fusedCheapParallelMinSize();
}
```

`decideFusedElementwise(...)` musi pouzit tyto hodnoty.

Proc:

- Repo uz ma fused-specific profilove knob hodnoty.
- cpu1 fused tuning nesmi ignorovat existujici kalibraci.

### Task 10.2: Pridat fused cpu1 route do profile JSON

Stav: `[ ]`

Soubor:

```text
src/main/java/config/profile/ExecutionProfileIO.java
```

Viz Task 6.4. Tento task je zde duplicitne uvedeny jako checkpoint pro profile/tuning fazi a musi byt oznacen `[x]` pouze pokud:

- read podporuje `fusedUseCpu1Elementwise`;
- write serializuje `fusedUseCpu1Elementwise`;
- default pro chybejici pole je `false`;
- existuje test pro backward compatible nacitani stareho profilu.

### Task 10.3: Pridat profile IO test

Stav: `[ ]`

Upravit existujici profile IO testy nebo pridat:

```text
src/test/java/config/profile/FusedExecutionPolicyProfileIOTest.java
```

Testy:

```java
@Test
void profileRoundTripsCpu1FusedRouteFlag() {
    RuntimeConfig runtime = RuntimeConfig.inferenceDefaults(DataType.FLOAT32)
            .withFused(RuntimeConfig.inferenceDefaults(DataType.FLOAT32).fused().withUseCpu1Elementwise(true));
    // write profile, read profile, assert loaded.runtime().fused().useCpu1Elementwise()
}

@Test
void missingCpu1FusedRouteFlagDefaultsToFalse() {
    // load minimal/stary profil JSON bez fusedUseCpu1Elementwise
    // assert false
}
```

Proc:

- Route musi byt reprodukovatelna z profilu.
- Stare profily se nesmi rozbit.

## Faze 11: Parity Benchmarky Proti Staremu CPU Fused

### Task 11.1: Pridat benchmark/test harness pro old CPU fused vs cpu1 fused

Stav: `[ ]`

Novy nebo existujici debug benchmark test:

```text
src/test/java/debug/Cpu1FusedParityBenchmarkTest.java
```

Scenare:

- cheap contiguous F32: `relu(a.mul(b).add(c))`
- cheap contiguous F64
- BF16 chain
- broadcast bias: `relu(a.add(bias))`
- where mask: `where(mask, x.mul(scale), fill)`
- transcendental: `tanh(exp(x).add(y))`
- strided input view
- native memory segment F32
- native memory segment F64

Benchmark musi porovnat:

- old CPU fused route: `fusedUseCpu1Elementwise=false`
- cpu1 fused route: `fusedUseCpu1Elementwise=true`

Reportovat:

```text
case,dtype,storage,elements,oldCpuMedianMs,cpu1MedianMs,ratio,cpu1VectorFallbackReason
```

Proc:

- Complete migration musi mit vykonovou evidenci.
- Pokud cpu1 route neni rychlejsi, trace musi rict proc.

### Task 11.2: Pridat correctness parity pro route on/off

Stav: `[ ]`

Test:

```java
@Test
void cpu1FusedRouteMatchesOldCpuFusedRoute() {
    Tensor yOld = graph.compile().prepare(config.withFused(config.fused().withUseCpu1Elementwise(false))).execute();
    Tensor yCpu1 = graph.compile().prepare(config.withFused(config.fused().withUseCpu1Elementwise(true))).execute();
    assertClose(yOld, yCpu1);
}
```

Pokryt:

- F32 exact/tolerance
- F64 tolerance
- BF16 tolerance
- BOOL output exact

Proc:

- Stary CPU fused je pro migraci baseline.
- cpu1 route musi byt semanticky zamenitelna.

### Task 11.3: Aktualizovat dokument podle benchmark vysledku

Stav: `[ ]`

Do tohoto dokumentu doplnit kratkou sekci:

```text
## Benchmark Evidence

Datum:
Platform:
Commit:
Konfigurace:
Souhrn:
```

Proc:

- Plan trackuje presny stav implementace.
- Vykonova rozhodnuti nesmi zustat implicitni.

## Faze 12: Finalni Overeni

### Task 12.1: Kompilace

Stav: `[ ]`

Spustit:

```bash
./gradlew classes
```

### Task 12.2: Cilene testy cpu1 fused

Stav: `[ ]`

Spustit:

```bash
./gradlew test --tests backend.cpu1.Cpu1FusedElementwiseExecutionContractTest --tests backend.cpu1.fused.Cpu1FusedIrBuilderTest
```

### Task 12.3: Regression testy dotcenych oblasti

Stav: `[ ]`

Spustit:

```bash
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest --tests backend.cpu1.Cpu1MseLossExecutionContractTest --tests graph.compile.planning.region.DefaultRegionOptimizerTest --tests graph.compile.planning.partition.CpuNaturalExecutionRegionPlannerTest
```

### Task 12.4: Stare fused regression testy

Stav: `[ ]`

Spustit stare fused testy:

```bash
./gradlew test --tests FusedExecutionModesTest --tests OptimizerFuseTest
```

Proc:

- Route knob je soucast kompletni migrace, proto jsou stare fused regression testy povinne.
- Testy musi projit pro default starou route i pro explicitni cpu1 route, kde dava smysl.

### Task 12.5: Benchmark/test evidence

Stav: `[ ]`

Spustit nebo dolozit:

```bash
./gradlew test --tests debug.Cpu1FusedParityBenchmarkTest
```

Pokud debug benchmark neni vhodny pro CI, vysledky ulozit pouze do dokumentu nebo konzole, ne do `profiles/platform/*`, pokud uzivatel explicitne nechce aktualizovat kanonicke profily.

### Task 12.6: Diff hygiene

Stav: `[ ]`

Spustit:

```bash
git diff --check -- src/main/java src/test/java todo
git status --short
```

Overit:

- zadne `.idea/*` staged;
- zadne `profiles/platform/*` staged;
- zadne scratch `.class` staged;
- dokument aktualizuje checkboxy podle skutecneho stavu.

## Kodove Zmeny - Souhrn Po Souborech

### Nove soubory

- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedAccessKind.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedNodeAttributes.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1NoAttributes.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1ScalarAttribute.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1WhereAttributes.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedInputPlan.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedNodePlan.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedExpressionPlan.java`
- [ ] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedIrBuilder.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1FusedElementwisePreparer.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1PreparedFusedElementwiseUnit.java`
- [ ] `src/main/java/backend/cpu1/exec/Cpu1FusedKernelArgs.java`
- [ ] `src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1RangeTask.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseRangeRunner.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseLoops.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseDispatch.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/scalar/Cpu1FusedScalarInterpreter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorPlan.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorInterpreter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorDispatch.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/vector/Cpu1FusedVectorFallbackReason.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelCacheKey.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/tuning/Cpu1FusedTuningClassifier.java`
- [ ] `src/test/java/backend/cpu1/fused/Cpu1FusedIrBuilderTest.java`
- [ ] `src/test/java/backend/cpu1/Cpu1FusedElementwiseExecutionContractTest.java`
- [ ] `src/test/java/backend/prepare/Cpu1FusedPrepareRoutingTest.java`
- [ ] `src/test/java/config/profile/FusedExecutionPolicyProfileIOTest.java`
- [ ] `src/test/java/debug/Cpu1FusedParityBenchmarkTest.java`

### Upravy existujicich souboru

- [ ] `src/main/java/backend/cpu1/exec/Cpu1ScratchBuffer.java`
- [ ] `src/main/java/backend/cpu1/exec/Cpu1ScratchBufferSpec.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1LaunchPolicy.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1SingleThreadLaunch.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1ParallelLaunch.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1RangeLauncher.java`
- [ ] `src/main/java/backend/cpu1/exec/Cpu1ElementwiseExecutableUnit.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1PreparedElementwiseUnit.java`
- [ ] `src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java`
- [ ] `src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java`
- [ ] `src/main/java/backend/prepare/BackendPrepareDispatcher.java`
- [ ] `src/main/java/config/runtime/FusedExecutionPolicy.java`
- [ ] `src/main/java/config/profile/ExecutionProfileIO.java`
- [ ] profile/config testy podle kompilatoru

## Co Zamerne Neprenest Ze Stareho CPU

### `PreparedFusedExecutable`

Neprenest.

Proc:

- Je navazany na stary CPU fused execution contract.
- Pracuje pres `Tensor` a `CpuKernelContext`.
- V cpu1 chceme `Cpu1TensorView` a prepared executable units.

### `FusedOperation`

Neprenest jako runtime operation.

Proc:

- `cpu1` ma dostat lowered fused execution unit a pripravit vlastni artifact.
- `Operation.OpType.FUSED` je stara CPU fasada pro registry/kernel path.

### Stary ASM generator 1:1

Neprenest 1:1.

Proc:

- Je rozsahove velky a ma vazby na stare numeric/storage contracts.
- Complete migration presto vyzaduje cpu1 vector/codegen hot path ve Fazi 9.
- Pokud benchmarky ukazou, ze ASM je nejlepsi emitter, ma byt zavedena cpu1-native emitter implementace nad `Cpu1FusedExpressionPlan`, ne import stareho `backend.cpu.fused.asm` jako compatibility layer.

### Stary native segment helper

Neprenest.

Proc:

- cpu1 uz ma `Cpu1TensorView`, runtime native slots a `ExecutionContext.requireNativeOutputStorage(...)`.

## Zadny Mimo-Plan Stav

Tento plan nema zamerne zadne sekce mimo hlavni implementaci. Veci, ktere byly v predchozi verzi mimo hlavni implementaci, jsou soucasti povinnych fazi:

- scratch buffer pro fused temporaries je Faze 8;
- vector/codegen hot path je Faze 9;
- route knob a profile IO jsou Faze 6 a Faze 10;
- benchmark parity proti staremu CPU fused je Faze 11;
- finalni overeni je Faze 12.

Pokud implementace narazi na novou prekazku, dokument se ma aktualizovat jako novy task uvnitr techto fazi, ne jako nova samostatna sekce mimo plan.

## Finalni Akceptacni Kriteria

Implementace je hotova, kdyz plati:

- [ ] cpu1 ma vlastni fused IR, neimportuje `backend.cpu.fused.*`
- [ ] cpu1 fused preparer pripravuje `Cpu1PreparedFusedElementwiseUnit`
- [ ] runtime executable je `backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnit`
- [ ] fused execution pouziva `Cpu1TensorView`, ne `TensorInternalAccess` jako primarni runtime kontrakt
- [ ] JAVA_ARRAY F32/F64/BF16 funguje
- [ ] MEMORY_SEGMENT F32/F64/BF16 funguje nebo je explicitne odmitnut v prepareru s testem
- [ ] bool compare/logical/where funguje
- [ ] broadcast vstupy funguji
- [ ] parallel launch funguje pres stejnou cpu1 launch policy
- [ ] fused temporaries nepouzivaji per-range alokace v hot path; jdou pres `Cpu1ScratchBuffer`
- [ ] vector/codegen fused hot path existuje pro podporovany contiguous F32/F64 subset
- [ ] nepodporovane vector cases maji explicitni `Cpu1FusedVectorFallbackReason`
- [ ] `FusedExecutionPolicy.useCpu1Elementwise` existuje a defaultuje na `false`
- [ ] `ExecutionProfileIO` umi route flag nacist i zapsat
- [ ] route-on/route-off correctness parity proti staremu CPU fused je otestovana
- [ ] benchmark parity evidence je doplnena v dokumentu
- [ ] trace ukazuje `CPU1_FUSED_ELEMENTWISE`
- [ ] trace ukazuje storage, node count, launch workers, vectorization a vector fallback reason
- [ ] zadne lokalni profily/IDE soubory nejsou soucasti commitu
- [ ] `./gradlew classes` projde
- [ ] cilene cpu1 fused testy projdou
- [ ] stare fused regression testy projdou
- [ ] dokument neobsahuje zadne migracni kroky mimo tento plan
