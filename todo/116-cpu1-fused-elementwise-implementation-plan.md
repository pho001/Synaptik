# 116. cpu1 Fused Elementwise Implementation Plan

## Stav Implementace

Status: `PLANNED`

Tento dokument je zivy implementacni checklist. Pri implementaci se ma menit stav jednotlivych tasku:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` hotovo a overeno

Aktualni stav:

- [x] Faze 0: overeni vstupnich hranic a ochrana pracovniho stromu
- [x] Faze 1: cpu1 fused IR
- [x] Faze 2: prepare-time fused plan a dispatch decision
- [ ] Faze 2.5: ASM prepare contract alignment
- [x] Faze 3: cpu1 fused executable unit v `backend.cpu1.exec`
- [x] Faze 4: codegen-first fused runner kontrakt pro JAVA_ARRAY a MEMORY_SEGMENT
- [x] Faze 5: trace a prepared artifact integrace
- [x] Faze 6: prepare dispatcher integrace a runtime config route
- [~] Faze 7: test coverage
- [~] Faze 8: ASM/codegen emitter hot path
- [ ] Faze 9: profile IO a tuning knobs
- [ ] Faze 10: parity benchmarky proti staremu CPU fused
- [ ] Faze 11: finalni overeni a odstraneni mezistavu

## Cil

Prenest fused elementwise execution koncept ze stareho `backend.cpu` do `backend.cpu1` tak, aby zapadl do stavajici cpu1 architektury:

```text
graph/region:
  vytvori ExecutionUnitKind.FUSED_ELEMENTWISE

prepare:
  z lowered fused unit vytvori Cpu1PreparedFusedElementwiseUnit
  rozhodne storage kind, layout/access model, launch policy a cost class
  vygeneruje nebo z cache ziska ASM class/template pro structural class signature
  vytvori prepared generated kernel instanci/handle se scalar hodnotami a ulozi ji do prepared unit

execute:
  Cpu1FusedElementwiseExecutableUnit binduje Cpu1TensorView vstupy/vystup
  runner spusti preparedUnit.generatedKernel() bez Tensor/autograd/stareho CpuKernelContext
```

Cilem neni 1:1 presun stareho CPU fused runtime. Cilem je prenest uzitecne casti:

- fused IR shape
- canonicalizaci
- broadcast/effective-stride pripravu
- prepare-time validaci pres `Operation.isFusable()` na konkretnich source operacich
- cost classification myslenku

a nahradit nebo zahodit casti, ktere jsou svazane se starym backendem:

- `PreparedFusedExecutable`
- `CpuKernelContext`
- `TensorInternalAccess` jako primarni kernel argument
- stary ASM generator jako compatibility import; cpu1 misto toho dostane vlastni ASM/codegen emitter nad `Cpu1FusedExpressionPlan`
- `Operation.OpType.FUSED` jako runtime fasadu pro cpu1

## Non-Goals

- Nezavadet obecnou compatibility vrstvu mezi starym `backend.cpu.fused` a `backend.cpu1`.
- Nekopirovat stary `CpuFusedExecutionArtifact`.
- Neportovat stary ASM generator 1:1 jako compatibility vrstvu.
- Nepouzivat node-switch execution pres `switch (opType)` jako produkcni fused hot path.
- Nezavadet zadnou jinou runtime cestu pro cpu1 fused. Pokud nejde vygenerovat ASM kernel,
  prepare/codegen eligibility musi region odmitnout s jasnym `Cpu1FusedCodegenRejectionReason`.
- Neprovadet skrytou contiguous materializaci uvnitr cpu1 fused runtime. Pokud je materializace
  zvolena kvuli layoutu, musi byt explicitni rozhodnuti graph/lowering/memory planning vrstvy.
- Nemenit graph optimizer fusion pravidla, pokud neni nutne opravit bug.
- Nemenit public `Tensor` API.
- Nekomitovat lokalni benchmark/profilove artefakty.
- Nezapinat cpu1 fused jako default bez parity testu, benchmarku a trace evidence. Plan obsahuje kompletni route knob a overeni v ramci teto migrace.

## Operation Metadata Boundary

- `src/main/java/backend/cpu/**` je pro tento plan legacy runtime. Jeho lokalni fused execution cesty, cheap/non-cheap helpery a canonicalizovane `OpType` rozhodovani nejsou precedent pro `backend.cpu1`, graph ani nove shared backend cesty.
- Mimo legacy `backend.cpu` je zdroj operation metadata konkretni `Operation` instance behem prepare/lowering: `arityClass()`, `isFusable()`, `semanticFamily()`, `computationalCost()`, `controlTrait()` a `resultKind()`.
- `Operation.OpType` zustava pouze stabilni identita. Nesmí byt znovu rozsiren na category/fusable/trait/cost/result registry.
- Hot path a prepared runtime nesmi drzet `Operation` ani trait snapshot kvuli dispatch rozhodovani. Prepare ma z `Operation` odvodit konkretni rozhodnuti a runtime uz jen vykonava prepared plan.
- `Cpu1FusedNodePlan` zustava canonicalizovany runtime IR uzel s `opType`, refs, dtype a `scalarParameter`. Nepridavat do nej `Operation` ani trait snapshot, pokud budouci faze neprokaze konkretni potrebu.
- Faze 2 musi klasifikovat fused cost z puvodnich `CompiledNode.operation()` pred tim, nez lowering/canonicalizace ztrati konkretni `Operation` objekt.

## Production Hot Path Contract

Fused elementwise produkcni cesta nesmi vykonavat canonicalizovane nody pres runtime
`switch (opType)`.

Cilovy hot path model je:

```text
prepare:
  Cpu1FusedExpressionPlan
  -> structural Cpu1FusedCodegenClassSignature pro expression/layout/storage/dtype/loop kind
  -> Cpu1FusedCodegenKernelFactory.prepareKernel(...)
  -> concrete generated class/template reused by exact class signature
  -> prepared generated kernel instance/handle with bound scalar values

execute:
  preparedUnit.generatedKernel().computeRange(args, start, end)
```

Generovany kernel ma emitovat primo konkretni smycku pro dany fused vyraz, napr.:

```text
out[i] = max(0, a[i] * b[i] + c[i])
```

ne:

```text
for each node:
  switch (node.opType())
```

cpu1 fused runtime route existuje jen pro ASM-generovane concrete kernels. Aktualni prvni
implementace ma concrete generated kernel pro podporovane `CONTIGUOUS_SCALAR` a
`STRIDED_SCALAR` loop kinds nad `JAVA_ARRAY` F32/F64 subsetem. `CONTIGUOUS_VECTOR`
a `MEMORY_SEGMENT` jsou zatim prepare-time rejected bez fallbacku. Nepodporovane vyrazy musi
prepare/codegen eligibility odmitnout s jasnym rejection reason; execute nesmi mit codegen,
cache lookup, eligibility rozhodovani, scalar interpreter, vector fallback, backend fallback ani
runtime evaluator.

## ASM Codegen Support Layer

cpu1 fused ASM/codegen potrebuje malou support vrstvu pro operace, ktere jsou obtizne,
neprehledne nebo nezadouci emitovat primo jako sekvenci raw ASM instrukci. Tato vrstva neni
fallback, interpreter ani runtime evaluator. Je to codegen-time knihovna pro emitovani
statickych volani z konkretni generated tridy.

Cilove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmMethodEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmCallEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmIntrinsicRegistry.java
src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedGeneratedSupport.java
src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport.java
```

Role:

- `Cpu1FusedAsmMethodEmitter` drzi male ASM utility pro metody generated tridy: locals,
  load/store helpery, labely, loop skeletony a spolecne `MethodVisitor` idiomy.
- `Cpu1FusedAsmCallEmitter` centralizuje `INVOKESTATIC` emitovani, owner internal names a
  descriptor validaci pro support metody.
- `Cpu1FusedAsmIntrinsicRegistry` rozhoduje pri codegen emitovani, jestli se konkretni
  op/dtype/access primitive emituje direct bytecode nebo jako static helper call. Registry je
  compile-time/codegen pomucka, ne runtime dispatch podle `opType`.
- `Cpu1FusedGeneratedSupport` obsahuje stabilni hand-written Java helpery pro storage/dtype/bool
  konverze, ktere generated bytecode vola staticky.
- `Cpu1FusedMathSupport` obsahuje stabilni hand-written Java helpery pro tezsi matematiku a
  pripadne fast approximation implementace.

Direct bytecode vs static helper call policy:

- Emitovat primo v ASM: jednoduche aritmeticke operace `ADD`, `SUB`, `MUL`, `DIV`, unary `NEG`,
  jednoduche comparison nody, `MIN`, `MAX`, `ABS`, `RELU`, `CLAMP_MIN`, `CLAMP_MAX`, linear
  contiguous load/store, scalar field load a jednoduchy `WHERE` select, pokud se bytecode da
  vyjadrit malym a citelnym instrukcnim blokem.
- Emitovat jako static call: tezka matematika `exp`, `log`, `tanh`, `erf`, obecny `pow`,
  fast approximations, BF16 load/store a konverze, boolean load/store konverze, dtype konverze
  mezi F32/F64/BF16/BOOL a jine primitive, ktere by jinak duplikovaly komplexni bitovou logiku
  v emitteru.
- Offset helper je povoleny jen mimo hot inner loop nebo pro jednorazovy setup. Vnitrni
  `STRIDED_SCALAR` loop ma preferovat generated offset math v ASM locals. Pokud by helper call
  mel byt v kazdem prvku, musi task explicitne dolozit, ze nejde o hot-path regresi.
- Volba exact vs fast approximation je soucast codegen planu/signature, protoze meni volany
  helper nebo jeho semantiku.

ASM-only invariant:

- Generated kernel smi volat stabilni static support metody pres `INVOKESTATIC`.
- Generated kernel nesmi volat runtime node evaluator, interpreter, fallback route, backend
  fallback ani per-node `switch (opType)` dispatch.
- Support metody jsou hand-written normal Java, testovane nezavisle, a nejsou operation-specific
  runtime dispatch. Nemaji prijimat `Cpu1FusedExpressionPlan`, node index, `Operation.OpType`,
  `Cpu1FusedNodePlan` ani seznam nodu.
- Emitter muze pouzit registry k vyberu emit strategie pri generovani tridy; generated trida uz
  nema zadne registry lookupy v `computeRange`.

Signature/cache policy:

- `Cpu1FusedCodegenClassSignature` je maly value object nad jedinym canonical stringem.
- `canonicalSignature` musi zachytit vsechny structural class/template dependencies: expression,
  dtype, storage, layout/access, loop kind, approximation policy, support ABI a stabilni helper
  targety, pokud generated bytecode vola support metody.
- Canonical string ma obsahovat napriklad `supportAbi=1` a serazeny seznam helper targetu
  typu `helpers=[backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport.expF32(F)F]`.
- Konkretni scalar hodnoty stale nepatri do structural class signature; scalar hodnoty se binduji
  do prepared kernel instance/fields. Helper targets patri do canonical signature, protoze meni
  bytecode targety a cache reuse.
- Zmena implementace support metody bez zmeny descriptoru obvykle nevyzaduje novou generated
  class. Zmena descriptoru, owneru, aproximacni semantiky nebo ABI kontraktu musi zmenit
  canonical signature, typicky pres `supportAbi=...` nebo helper target string.

Support class skeleton:

```java
package backend.cpu1.kernels.fused.codegen.support;

public final class Cpu1FusedGeneratedSupport {
    public static final int ABI_VERSION = 1;

    private Cpu1FusedGeneratedSupport() {
    }

    public static float bf16ToFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    public static short floatToBf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    public static boolean boolFromByte(byte value) {
        return value != 0;
    }

    public static byte boolToByte(boolean value) {
        return (byte) (value ? 1 : 0);
    }

    public static float f64ToF32(double value) {
        return (float) value;
    }

    public static double f32ToF64(float value) {
        return value;
    }
}
```

```java
package backend.cpu1.kernels.fused.codegen.support;

public final class Cpu1FusedMathSupport {
    private Cpu1FusedMathSupport() {
    }

    public static float expF32(float value) {
        return (float) Math.exp(value);
    }

    public static double expF64(double value) {
        return Math.exp(value);
    }

    public static float logF32(float value) {
        return (float) Math.log(value);
    }

    public static float tanhF32(float value) {
        return (float) Math.tanh(value);
    }

    public static float powF32(float left, float right) {
        return (float) Math.pow(left, right);
    }
}
```

ASM call emitter skeleton:

```java
package backend.cpu1.kernels.fused.codegen.asm;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public final class Cpu1FusedAsmCallEmitter {
    private Cpu1FusedAsmCallEmitter() {
    }

    public static void emitInvokeStatic(
            MethodVisitor mv,
            String owner,
            String name,
            String descriptor
    ) {
        if (mv == null) {
            throw new IllegalArgumentException("mv cannot be null");
        }
        if (owner == null) {
            throw new IllegalArgumentException("owner cannot be null");
        }
        if (name == null) {
            throw new IllegalArgumentException("name cannot be null");
        }
        if (descriptor == null) {
            throw new IllegalArgumentException("descriptor cannot be null");
        }
        mv.visitMethodInsn(Opcodes.INVOKESTATIC, owner, name, descriptor, false);
    }
}
```

Emitter usage example:

```java
private static final String MATH_SUPPORT =
        "backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport";
private static final String GENERATED_SUPPORT =
        "backend/cpu1/kernels/fused/codegen/support/Cpu1FusedGeneratedSupport";

private static void emitExpF32(MethodVisitor mv) {
    // stack before: float value
    Cpu1FusedAsmCallEmitter.emitInvokeStatic(mv, MATH_SUPPORT, "expF32", "(F)F");
    // stack after: float result
}

private static void emitBf16LoadConvert(MethodVisitor mv) {
    // stack before: short bits
    Cpu1FusedAsmCallEmitter.emitInvokeStatic(mv, GENERATED_SUPPORT, "bf16ToFloat", "(S)F");
    // stack after: float result
}
```

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
- Kernel hot path nesmi pridavat null checky do vnitrni vypocetni smycky.

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
  Cpu1FusedScalarParameter.java
  Cpu1FusedInputPlan.java
  Cpu1FusedNodePlan.java
  Cpu1FusedExpressionPlan.java
  Cpu1FusedIrBuilder.java

src/main/java/backend/cpu1/prepare/
  Cpu1FusedElementwisePreparer.java
  Cpu1PreparedFusedElementwiseUnit.java

src/main/java/backend/cpu1/prepare/dispatch/
  Cpu1FusedDispatchDecision.java

src/main/java/backend/cpu1/exec/
  Cpu1FusedKernelArgs.java
  Cpu1FusedElementwiseExecutableUnit.java

src/main/java/backend/cpu1/kernels/fused/
  Cpu1FusedElementwiseRangeRunner.java

src/main/java/backend/cpu1/kernels/fused/codegen/
  Cpu1FusedCodegenKernel.java
  Cpu1FusedCodegenKernelFactory.java
  Cpu1FusedCodegenClassSignature.java
  Cpu1FusedCodegenLoopKind.java
  Cpu1FusedCodegenPlan.java
  Cpu1FusedCodegenRejectionReason.java

src/main/java/backend/cpu1/kernels/fused/codegen/support/
  Cpu1FusedGeneratedSupport.java
  Cpu1FusedMathSupport.java

src/main/java/backend/cpu1/kernels/fused/codegen/asm/
  Cpu1FusedAsmClassEmitter.java
  Cpu1FusedAsmMethodEmitter.java
  Cpu1FusedAsmCallEmitter.java
  Cpu1FusedAsmIntrinsicRegistry.java
  Cpu1FusedAsmExpressionEmitter.java
  Cpu1FusedAsmLoopEmitter.java
  Cpu1FusedGeneratedClassLoader.java

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

Class/template reuse je zalozeny na structural expression/layout/storage/dtype/loop signature.
Signature nesmi obsahovat graph node ids, unit ids ani konkretni runtime scalar hodnoty, pokud
plan pozdeji explicitne neobhaji constant embedding pro konkretni hot path.

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
- Stara interpretovana cesta je pouze semanticka reference, ne runtime struktura k prekopirovani.
- Dispatch planner je inspirace, ale cpu1 musi pouzit `Cpu1DispatchPolicy`/`CpuKernelConfig`.

Evidence:

- `FusedIrBuilder` obsahuje prenositelne ref mapovani, canonicalizaci `pow`/`mulScalar`, extrakci atributu a broadcast/effective-stride access klasifikaci.
- `FusedExpressionPlan`, `FusedNodePlan` a `FusedExternalInputPlan` jsou male immutable plan objekty s kopirovanim listu/poli; pro cpu1 je nutne prepsat null validaci do explicitniho stylu bez `Objects`.
- `InterpretedPreparedFusedExecutable` potvrzuje semantiku op evaluace a storage index vypoctu, ale je svazany s `TensorInternalAccess`, `Tensor`, `CpuKernelContext` a CPU_JAVA_ARRAY-only legacy cestou.
- `FusedDispatchPlanner` klasifikuje cheap/non-cheap a contiguous/strided rodiny uvnitr legacy CPU runtime; cpu1 navazka smi prevzit jen myslenku prepare-time dispatch rozhodnuti a musi ji vyjadrit pres `Operation` metadata, `Cpu1DispatchPolicy` a `CpuKernelConfig`.
- `FusedNumericContractResolver` podporuje floating/BOOL cesty a odmita INT32/INT64 ve fused numeric contractu; cpu1 plan musi explicitne rozhodnout dtype/storage kontrakt.
- Aktualni graph vrstva uz produkuje `ExecutionUnitKind.FUSED_ELEMENTWISE`; `PreparedExecutionBuilder` vola `BackendPrepareDispatcher.prepareCpuFusedStep(...)`, ktery dnes stale routuje do stareho `cpuPreparer.prepareLoweredFusedStep(...)`.
- Aktualni cpu1 integracni body jsou `Cpu1PreparedArtifact`, `Cpu1ElementwiseExecutableUnit`, `Cpu1KernelArgs`, `Cpu1TraceContributor` a `Cpu1NodePreparer`; zatim neexistuje cpu1 fused prepared unit ani trace vetev.

## Faze 1: cpu1 Fused IR

### Task 1.1: Pridat `Cpu1FusedAccessKind`

Stav: `[x]`

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

### Task 1.2: Pridat `Cpu1FusedScalarParameter`

Stav: `[x]`

Novy soubor:

```text
src/main/java/backend/cpu1/fused/ir/Cpu1FusedScalarParameter.java
```

Kod:

```java
package backend.cpu1.fused.ir;

public record Cpu1FusedScalarParameter(boolean present, float f32, double f64) {
    public static final Cpu1FusedScalarParameter NONE =
            new Cpu1FusedScalarParameter(false, 0.0f, 0.0d);

    public static Cpu1FusedScalarParameter of(float f32, double f64) {
        return new Cpu1FusedScalarParameter(true, f32, f64);
    }
}
```

Proc:

- `POW`, `MUL_SCALAR`, `CLAMP_MIN`, `CLAMP_MAX` potrebuji scalar parametr.
- Parametr se vytahne z konkretni `operations` tridy pri prepare/IR build, napriklad z `pow.getExponentF32()` a `pow.getExponent()`.
- Fused runtime plan neobsahuje `Operation`; drzi jen snapshot hodnoty potrebne pro kernel.
- Parametr drzi F32 i F64 reprezentaci, aby F32/BF16 hot path nemusela opakovane castovat z double.
- Bezparametricke operace vcetne `WHERE` pouzivaji `Cpu1FusedScalarParameter.NONE`; specialni arita `WHERE` patri do `opType` a `inputRefs`, ne do atribut objektu.
- Nezavadime obecny `attributes` interface, dokud neexistuje realny non-scalar payload.

### Task 1.3: Pridat `Cpu1FusedInputPlan`

Stav: `[x]`

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

Stav: `[x]`

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
        Cpu1FusedScalarParameter scalarParameter
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
        scalarParameter = scalarParameter == null ? Cpu1FusedScalarParameter.NONE : scalarParameter;
    }
}
```

Proc:

- Fused node neobsahuje `Tensor`.
- Obsahuje jen op type, refs, dtype a scalar parameter snapshot.
- Neobsahuje puvodni `Operation`, proto kernel runtime nemusi castovat `pow`, `mulScalar`, `clampMin` nebo `clampMax`.
- `nodeId` zustava pro trace/debug a validaci.

### Task 1.5: Pridat `Cpu1FusedExpressionPlan`

Stav: `[x]`

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

Stav: `[x]`

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
                    canonical.scalarParameter()
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
                        Cpu1FusedScalarParameter.of(1.0f, 1.0d)
                );
            }
            if (exponent == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (exponent == -1.0d) {
                return new CanonicalNode(Operation.OpType.INV, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (exponent == 2.0d) {
                return new CanonicalNode(Operation.OpType.MUL, List.of(inputRef, inputRef), Cpu1FusedScalarParameter.NONE);
            }
        }
        if (operation instanceof mulScalar m && inputRefs.size() == 1) {
            double scalar = m.getScalar();
            int inputRef = inputRefs.getFirst();
            if (scalar == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        Cpu1FusedScalarParameter.of(0.0f, 0.0d)
                );
            }
            if (scalar == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (scalar == -1.0d) {
                return new CanonicalNode(Operation.OpType.NEG, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
        }
        return new CanonicalNode(operation.opType(), List.copyOf(inputRefs), scalarParameter(operation));
    }

    private static Cpu1FusedScalarParameter scalarParameter(Operation operation) {
        if (operation instanceof pow p) {
            return Cpu1FusedScalarParameter.of(p.getExponentF32(), p.getExponent());
        }
        if (operation instanceof mulScalar m) {
            return Cpu1FusedScalarParameter.of(m.getScalarF32(), m.getScalar());
        }
        if (operation instanceof clampMin c) {
            return Cpu1FusedScalarParameter.of(c.getMinValueF32(), c.getMinValue());
        }
        if (operation instanceof clampMax c) {
            return Cpu1FusedScalarParameter.of(c.getMaxValueF32(), c.getMaxValue());
        }
        return Cpu1FusedScalarParameter.NONE;
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
            Cpu1FusedScalarParameter scalarParameter
    ) {
    }
}
```

Proc:

- Tohle je nejdulezitejsi cast, kterou prebirame ze stareho CPU.
- Je stale prepare-time a backend-local.
- Nepouziva `Tensor`.

Evidence:

- Pridany Phase 1 IR soubory v `src/main/java/backend/cpu1/fused/ir/`.
- `Cpu1FusedIrBuilder` drzi jen `Operation.OpType`, input/output refs, dtype a `Cpu1FusedScalarParameter`; runtime node plan neuchovava `Operation`.
- Scalar snapshoty se berou z konkretnich operaci `pow`, `mulScalar`, `clampMin` a `clampMax`; bezparametricke operace vcetne `WHERE` pouzivaji `Cpu1FusedScalarParameter.NONE`.
- Novy cpu1 fused IR kod pouziva explicitni null validaci bez `Objects.requireNonNull`.
- Overeno: `./gradlew classes`.
- Overeno: kontrola zakazanych starych cpu1 fused atributovych nazvu v novem cpu1 fused kodu a planu bez nalezu.
- Overeno: `rg -n "Objects\\.requireNonNull" src/main/java/backend/cpu1/fused` bez nalezu.
- Overeno: `git diff --check -- src/main/java/backend/cpu1/fused todo/116-cpu1-fused-elementwise-implementation-plan.md`.

## Faze 2: Prepared Fused Unit A Dispatch

Poznamka k aktualnimu planu:

- Faze 1-2 jsou povazovane za implementovane pro IR, prepare-time fused plan a dispatch decision.
- Finalni ASM-only prepare-time generated kernel handle kontrakt se doplnuje az ve Fazi 2.5 pred
  Fazi 3, aby executable unit nevznikal nad nefinalnim prepare kontraktem.

Implementacni evidence:

- Pridan `Cpu1PreparedFusedElementwiseUnit` jako immutable prepared popis bez `Operation` objektu.
- Pridan `Cpu1FusedDispatchDecision` a `Cpu1DispatchPolicy.decideFusedElementwise(...)`; fused cost se pocita z `Operation.computationalCost()` source operaci sebranych behem prepare.
- Pridan `Cpu1FusedElementwisePreparer.prepareUnit(...)`, ktery validuje konkretni `node.operation().isFusable()`, buildi `Cpu1FusedExpressionPlan`, rozhoduje storage/layout/launch/dispatch a neuklada `Operation` do prepared unit.
- Phase 2 zamerne nepridava `Cpu1PreparedArtifact` route ani `BackendPrepareDispatcher` prepnuti, protoze executable/artifact integrace jsou Faze 3/5/6. Tim nevznika docasny neexecutable artifact ani docasna facade.
- Overeno: `./gradlew classes`.
- Overeno: `./gradlew test --tests backend.cpu1.Cpu1DispatchPolicyTest --tests backend.cpu1.Cpu1FusedElementwisePreparerTest`.
- Overeno: `git diff --check -- <touched Phase 2 files>`.
- Overeno: grep zakazaneho hard-coded fused-op helperu v `src/main/java`, `src/test/java` a tomto todo souboru bez nalezu.

### Task 2.1: Pridat `Cpu1PreparedFusedElementwiseUnit`

Stav: `[x]`

Novy soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1PreparedFusedElementwiseUnit.java
```

Kod:

```java
package backend.cpu1.prepare;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
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
    private final Cpu1FusedDispatchDecision dispatchDecision;
    private final Cpu1FusedCodegenRejectionReason codegenRejectionReason;
    private final Cpu1FusedCodegenKernel generatedKernel;
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
            Cpu1FusedDispatchDecision dispatchDecision,
            Cpu1FusedCodegenRejectionReason codegenRejectionReason,
            Cpu1FusedCodegenKernel generatedKernel,
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
        if (codegenRejectionReason == null) {
            throw new IllegalArgumentException("codegenRejectionReason cannot be null");
        }
        if (codegenRejectionReason == Cpu1FusedCodegenRejectionReason.NONE && generatedKernel == null) {
            throw new IllegalArgumentException("generatedKernel cannot be null when codegenRejectionReason is NONE");
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
        this.codegenRejectionReason = codegenRejectionReason;
        this.generatedKernel = generatedKernel;
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

    public Cpu1FusedDispatchDecision dispatchDecision() {
        return dispatchDecision;
    }

    public Cpu1FusedCodegenRejectionReason codegenRejectionReason() {
        return codegenRejectionReason;
    }

    public Cpu1FusedCodegenKernel generatedKernel() {
        return generatedKernel;
    }

    public boolean approximateExp() {
        return approximateExp;
    }

    public boolean approximateTanh() {
        return approximateTanh;
    }
}
```

Proc:

- Prepared unit je immutable popis fused vypoctu.
- Prepared unit drzi prepared generated kernel instanci/handle. Execute ji jen vola; neresi codegen,
  cache lookup, eligibility ani fallback.
- Exekuce bude bindovat runtime views az v `Cpu1FusedElementwiseExecutableUnit`.
- `launchConfig` je ulozen zvlast pro trace, protoze `Cpu1LaunchPolicy` je interface.
- Neposkytuje `Operation` helpery ani trait snapshoty; pokud trace/test potrebuje zjistit pritomnost
  konkretniho op typu, ma cist canonicalizovane `plan().nodes()`.

### Task 2.2: Pridat fused dispatch decision a rozsirit `Cpu1DispatchPolicy`

Stav: `[x]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/dispatch/Cpu1FusedDispatchDecision.java
src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java
```

Pridat record:

```java
package backend.cpu1.prepare.dispatch;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;

/**
 * Prepare-time dispatch result for a fused elementwise unit.
 *
 * <p>Unlike single-op dispatch this record intentionally has no kernel
 * {@code Operation.OpType}: a fused region has no representative operation
 * identity. The concrete fused IR and later ASM/codegen eligibility
 * decisions describe what can run.</p>
 */
public record Cpu1FusedDispatchDecision(
        Cpu1CostClass costClass,
        Cpu1VectorizationKind requestedVectorizationKind,
        Cpu1LaunchConfig launchConfig,
        Cpu1StorageKind storageKind,
        int scalarChunkSize,
        int vectorChunkSize,
        int plannedWorkers
) {
    public Cpu1FusedDispatchDecision {
        if (costClass == null) {
            throw new IllegalArgumentException("costClass cannot be null");
        }
        if (requestedVectorizationKind == null) {
            throw new IllegalArgumentException("requestedVectorizationKind cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        scalarChunkSize = Math.max(1, scalarChunkSize);
        vectorChunkSize = Math.max(1, vectorChunkSize);
        plannedWorkers = Math.max(1, plannedWorkers);
    }
}
```

Pridat public metodu:

```java
public Cpu1FusedDispatchDecision decideFusedElementwise(
        Cpu1FusedExpressionPlan plan,
        List<Operation> sourceOperations,
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
    if (sourceOperations == null || sourceOperations.isEmpty()) {
            throw new IllegalArgumentException("sourceOperations cannot be null or empty");
        }
    if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
    Cpu1CostClass costClass = fusedCostClass(sourceOperations);
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
    return new Cpu1FusedDispatchDecision(
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
private static Cpu1CostClass fusedCostClass(List<Operation> sourceOperations) {
    for (Operation operation : sourceOperations) {
        if (operationCostClass(operation) == Cpu1CostClass.EXPENSIVE_ELEMENTWISE) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
    }
    return Cpu1CostClass.CHEAP_ELEMENTWISE;
}

private static Cpu1CostClass operationCostClass(Operation operation) {
    if (operation == null) {
        return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
    }
    return switch (operation.computationalCost()) {
        case MEDIUM, EXPENSIVE -> Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        case TRIVIAL, CHEAP, UNKNOWN -> Cpu1CostClass.CHEAP_ELEMENTWISE;
    };
}
```

Nutne importy:

```java
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import java.util.List;
```

Proc:

- Fused chain ma cost podle nejdrazsiho nodu.
- Backend-neutral zdroj cost klasifikace je `Operation.computationalCost()` konkretni
  source operace. `Operation.OpType` se nepouziva jako cheap/transcendental registry
  ani jako representative fused op.
- Pocet source operaci nemusi byt stejny jako pocet canonical fused nodu, protoze builder smi
  jeden source node rozlozit na podporovanou sekvenci jako `POW(-2)` -> `MUL` -> `INV`.
- `Cpu1FusedElementwisePreparer` ma source operations sebrat z puvodnich `CompiledNode.operation()`
  pred canonicalizaci do `Cpu1FusedNodePlan`. `Cpu1FusedNodePlan` dal nesmi uchovavat `Operation`
  ani trait snapshot; drzi jen `opType`, refs, dtype a scalar parameter.
- Stejna pripravena dispatch decision se pouzije pro launch policy a ASM/codegen hot path.
- Execute nesmi drzet `Operation`, cist traits, pocitat thresholdy ani znovu rozhodovat generated-kernel eligibility.
- Bool/select/logical/control/layout omezeni nejsou cost. Nesmí menit fused cost class jen proto,
  ze aktualni ASM/codegen subset neumi dany vysledek nebo control trait; tyto limity patri
  do ASM/codegen eligibility a rejection reason ve Fazi 8.

Poznamka:

- `classifyElementwise(...)` uz nema udrzovat vlastni hard-coded transcendental list; ma mapovat
  `Operation.computationalCost()` na cpu1 policy bucket a zachovat existujici cpu1 vyjimky.
- Pokud bude implementace potrebovat testovat cost separovane, pridej package-private metodu primo v teto fazi.
- Stary CPU fused kod muze lokalne klasifikovat canonicalizovane `OpType`, pokud uz nema puvodni
  `Operation`; tato legacy vyjimka se nesmi povysit zpet na centralni `OpType` trait registry.

### Task 2.3: Pridat `Cpu1FusedElementwisePreparer`

Stav: `[x]`

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
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
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
        List<Operation> sourceOperations = sourceOperations(loweredUnit, context);

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
        Cpu1FusedDispatchDecision dispatchDecision = dispatchPolicy.decideFusedElementwise(
                plan,
                sourceOperations,
                computeType,
                outputNode.flatDataSize(),
                config
        );
        Cpu1LaunchConfig launchConfig = dispatchDecision.launchConfig();
        Cpu1LayoutKind layoutKind = layoutKind(plan, outputNode);
        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                computeType,
                layoutKind,
                dispatchDecision.storageKind(),
                Cpu1FusedCodegenLoopKind.select(plan, layoutKind, dispatchDecision),
                config
        );
        Cpu1FusedCodegenRejectionReason rejectionReason = codegenEligibility.reason(
                codegenPlan,
                sourceOperations
        );
        if (rejectionReason != Cpu1FusedCodegenRejectionReason.NONE) {
            throw new UnsupportedOperationException("cpu1 fused ASM codegen rejected "
                    + loweredUnit.unitId() + ": " + rejectionReason);
        }
        Cpu1FusedCodegenKernel generatedKernel = Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
        Cpu1PreparedFusedElementwiseUnit preparedUnit = new Cpu1PreparedFusedElementwiseUnit(
                loweredUnit.unitId(),
                loweredUnit.orderedNodeIds(),
                plan.inputs().stream().map(input -> input.nodeId()).toList(),
                outputNode.id(),
                outputNode.dataType(),
                outputNode.flatDataSize(),
                outputNode.shape(),
                plan,
                layoutKind,
                dispatchDecision.storageKind(),
                launchPolicy(launchConfig),
                launchConfig,
                dispatchDecision,
                rejectionReason,
                generatedKernel,
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
            if (node == null || node.operation() == null || !node.operation().isFusable()) {
                throw new UnsupportedOperationException("cpu1 fused unit contains non-fusable nodeId=" + nodeId);
            }
            requireSupportedDType(node.dataType(), "node " + nodeId + " output");
        }
    }

    private static List<Operation> sourceOperations(LoweredExecutionUnit loweredUnit, BackendPrepareContext context) {
        return loweredUnit.orderedNodeIds().stream()
                .map(context::compiledNode)
                .map(CompiledNode::operation)
                .toList();
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

    private static InputResidencyRequirement inputResidencyRequirement(Cpu1FusedDispatchDecision decision) {
        return decision.storageKind() == backend.cpu1.storage.Cpu1StorageKind.MEMORY_SEGMENT
                ? InputResidencyRequirement.none()
                : InputResidencyRequirement.cpuReadableAll();
    }
}
```

Proc:

- Preparer je misto, kde se fused lowered unit prelozi na cpu1 prepared unit.
- Dela validate pres `node.operation().isFusable()` na konkretnich source operacich.
- Neudrzuje vlastni cpu1 whitelist fusable op podle `Operation.OpType`; source of truth je
  metadata implementovane primo v jednotlivych `Operation`.
- ASM/codegen eligibility patri do prepare a pred vytvorenim `Cpu1PreparedArtifact` musi
  bud pripravit generated kernel, nebo region odmitnout s rejection reason.
- Sbira `sourceOperations` z puvodnich `CompiledNode.operation()` pred canonicalizaci, predava je
  pouze prepare-time dispatch policy a neuklada je do prepared unit.
- Execute uz nesmi cist `sourceOperations`, volat codegen factory, rozhodovat eligibility ani hledat fallback.
- Rozhoduje storage podle runtime CPU storage profile.
- Nepouziva stary `FusedOperationBuilder`.

Poznamka k `runtimeConfig.approximation().useFastExp()`:

- Pred implementaci overit skutecna jmena metod v `ApproximationConfig`.
- Pokud se jmenuji jinak, kod upravit podle aktualniho API.
- Smysl zustava: EXP/TANH aproximacni policy musi jit z runtime configu.

## Faze 2.5: ASM Prepare Contract Alignment

Stav: `[ ]`

Phase 1-2 uz implementovaly cpu1 fused IR, prepare-time fused plan a dispatch rozhodnuti.
Pred Fazi 3 ale musi probehnout cleanup/alignment, protoze finalni smer planu je
ASM-only prepare-time generated kernel handle. Executable unit nema dostat mezistupen, ktery by
za behu volal factory/cache, eligibility helper, fallback, interpreter nebo runtime evaluator.

Tato faze neni compatibility layer, facade, fallback ani docasny interpreter. Je to finalni
prepare-time kontrakt, ktery musi existovat pred implementaci executable unit. Faze 3 na tuto
fazi zavisi. Faze 8 pozdeji vyplni realne telo ASM emitteru a hot path pro podporovany subset,
ale Faze 2.5 definuje finalni contract typy, prepared generated kernel handle a prepare-time
chovani pri nepodporovanem codegen subsetu.

### Task 2.5.1: Pridat minimalni finalni codegen contract typy

Stav: `[ ]`

Nove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenPlan.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenLoopKind.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenClassSignature.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenRejectionReason.java
```

Pozadavky:

- `Cpu1FusedCodegenKernel` je finalni prepared kernel handle, ktery executable unit pouze vola.
- `Cpu1FusedCodegenPlan` je prepare-time vstup do factory/emitteru a obsahuje expression plan,
  storage kind, layout/access volby, loop kind, compute dtype, scalar binding metadata a class
  signature.
- `Cpu1FusedCodegenLoopKind` minimalne rozlisuje `CONTIGUOUS_VECTOR`, `CONTIGUOUS_SCALAR` a
  `STRIDED_SCALAR`.
- `Cpu1FusedCodegenClassSignature` je structural class/template identity bez graph node id, unit id
  a konkretnich scalar hodnot. Je to value object nad jedinym canonical stringem; pokud generated
  bytecode vola support helpers, support ABI a helper targety jsou soucasti tohoto stringu.
- `Cpu1FusedCodegenRejectionReason` obsahuje `NONE` a explicitni prepare-time duvody pro
  nepodporovany intrinsic/op, dtype, layout/access, storage kind, loop kind, scalar binding nebo
  chybejici ASM emitter implementaci.

### Task 2.5.2: Pridat finalni prepare-time factory/cache API skeleton

Stav: `[ ]`

Novy soubor:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java
```

Pozadavky:

- Public API je `Cpu1FusedCodegenKernelFactory.prepareKernel(Cpu1FusedCodegenPlan plan)`.
- Factory/cache se vola jen v prepare, nikdy v execute.
- Pokud ASM emitter jeste nema realnou implementaci pro dany supported-looking plan, prepare musi
  jasne odmitnout pres `Cpu1FusedCodegenRejectionReason`, ne spustit fallback, interpreter ani
  runtime evaluator.
- Skeleton smi docasne produkovat prepared handle jen pro minimalni testovaci supported subset,
  pokud to neobchazi ASM-only kontrakt. Nesmí pridat zadnou execute-time lookup cestu.

### Task 2.5.3: Pridat generated kernel handle do prepared unit

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1PreparedFusedElementwiseUnit.java
```

Pozadavky:

- `Cpu1PreparedFusedElementwiseUnit` uklada `Cpu1FusedCodegenKernel generatedKernel`.
- Constructor uklada `Cpu1FusedCodegenRejectionReason codegenRejectionReason`.
- Pokud je `codegenRejectionReason == Cpu1FusedCodegenRejectionReason.NONE`, constructor musi
  vyzadovat non-null `generatedKernel`.
- Pokud rejection reason neni `NONE`, prepared executable artifact se nema vytvorit pro cpu1 fused
  route; prepare ma region odmitnout pred executable phase.
- Execute ma cist jen prepared `generatedKernel` handle. Nesmí znovu pocitat eligibility ani hledat
  factory/cache.

### Task 2.5.4: Aktualizovat preparer na finalni prepare-time codegen route

Stav: `[ ]`

Soubor:

```text
src/main/java/backend/cpu1/prepare/Cpu1FusedElementwisePreparer.java
```

Pozadavky:

- Preparer buildi `Cpu1FusedCodegenPlan` po fused IR, dispatch, storage/layout a launch rozhodnuti.
- Preparer spocita ASM/codegen eligibility a explicitni `Cpu1FusedCodegenRejectionReason`.
- Preparer vola `Cpu1FusedCodegenKernelFactory.prepareKernel(plan)` pouze pri prepare.
- `Cpu1PreparedFusedElementwiseUnit` vznikne az po uspesnem `prepareKernel`.
- Nepodporovany intrinsic, dtype nebo layout/access se odmitne v prepare s jasnou rejection reason.
- Nepridavat fallback, interpreter, runtime evaluator ani docasnou non-ASM execution cestu.

### Task 2.5.5: Srovnat zavadejici dispatch komentare a nazvoslovi

Stav: `[ ]`

Soubory:

```text
src/main/java/backend/cpu1/prepare/dispatch/Cpu1FusedDispatchDecision.java
src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java
```

Pozadavky:

- Prejmenovat nebo upravit zavadejici komentare typu vector/codegen eligibility tak, aby mluvily
  o ASM/codegen tuning/eligibility.
- Dispatch decision zustava prepare-time cost/launch/vectorization tuning vysledek, ne execute-time
  generated-kernel eligibility autorita.
- Bool/select/logical/control/layout omezeni patri do ASM/codegen eligibility a rejection reason,
  ne do fused cost class.

### Task 2.5.6: Pridat focused alignment testy

Stav: `[ ]`

Testy:

```text
src/test/java/backend/cpu1/Cpu1FusedCodegenContractAlignmentTest.java
src/test/java/backend/cpu1/Cpu1FusedElementwisePreparerTest.java
```

Pokryt:

- Podporovany minimalni plan ulozi non-null `generatedKernel` handle do
  `Cpu1PreparedFusedElementwiseUnit`.
- Nepodporovany intrinsic/op, dtype a layout/access se odmitne v prepare pres konkretni
  `Cpu1FusedCodegenRejectionReason`.
- Executable phase uz nepotrebuje factory/cache lookup; prepared unit nese vse, co execute potrebuje
  k zavolani `generatedKernel`.
- Constructor `Cpu1PreparedFusedElementwiseUnit` odmitne `NONE` rejection reason s null
  `generatedKernel`.

Evidence po implementaci:

- `./gradlew classes`
- `./gradlew test --tests backend.cpu1.Cpu1FusedCodegenContractAlignmentTest --tests backend.cpu1.Cpu1FusedElementwisePreparerTest`
- `rg -n "vector/codegen" src/main/java/backend/cpu1 src/test/java/backend/cpu1 todo/116-cpu1-fused-elementwise-implementation-plan.md`
- `rg -n "fallback|interpreter|evaluator" src/main/java/backend/cpu1 src/test/java/backend/cpu1 todo/116-cpu1-fused-elementwise-implementation-plan.md`
- `git diff --check -- todo/116-cpu1-fused-elementwise-implementation-plan.md`

## Faze 3: Executable Unit V `backend.cpu1.exec`

Zavislost:

- Faze 3 zacina az po Fazi 2.5. Executable unit smi predpokladat, ze
  `Cpu1PreparedFusedElementwiseUnit` uz obsahuje finalni prepared `generatedKernel` handle a ze
  prepare odmitl nepodporovane ASM/codegen cases pred vytvorenim executable artifactu.

### Task 3.1: Pridat `Cpu1FusedKernelArgs`

Stav: `[x]`

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
- Fused potrebuje plan pro vice internal nodu, generated offset metadata a external input metadata.
- Internal fused mezivysledky patri do generated ASM locals; `Cpu1FusedKernelArgs` nesmi zavest
  runtime scratch arrays ani evaluator state.

### Task 3.2: Pridat `Cpu1FusedElementwiseExecutableUnit`

Stav: `[x]`

Novy soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java
```

Kod:

```java
package backend.cpu1.exec;
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
        preparedUnit.launchPolicy().launch(
                args.elementCount(),
                (start, end) -> preparedUnit.generatedKernel().computeRange(args, start, end)
        );

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

Implementovano spolu s nejmensi nutnou upravou launch interface z Faze 4.1, protoze puvodni
`Cpu1LaunchPolicy.launch(...)` bylo pevne navazane na `Cpu1ElementwiseRangeRunner` a `Cpu1KernelArgs`.

Proc:

- Runtime executable patri do `backend.cpu1.exec`.
- Vstupy/vystupy se binduji stejne jako u elementwise.
- Vystup native storage jde pres runtime slot cache, ne per-execute alokace mimo kontext.

## Faze 4: Codegen-First Fused Runner

### Task 4.1: Zobecnit range launch bez rozbiti elementwise

Stav: `[x]`

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

Upraveno `Cpu1LaunchPolicy.java` na genericky range launch:

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

Stav: `[x]`

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

### Task 4.3: Vynutit ASM-only fused route

Stav: `[x]`

Zasada:

- cpu1 fused route ma pouze generated ASM kernels.
- Pokud codegen eligibility neni `NONE`, prepare/dispatch musi jednotku odmitnout s konkretnim
  `Cpu1FusedCodegenRejectionReason`.
- Nepridavat zadnou jinou runtime execution cestu.
- Korektnost generated kernelu se overuje proti existujicim unfused/fused execution contract testum,
  ne proti novemu cpu1 node-switch executoru.

Implementovano:

- `Cpu1FusedElementwisePreparer` buildi `Cpu1FusedCodegenPlan`, cte prepare-time
  `rejectionReason()` a pri nepodporovanem planu vyhazuje explicitni rejection pres
  `Cpu1FusedCodegenKernelFactory.rejection(...)`.
- `Cpu1FusedCodegenKernelFactory.prepareKernel(...)` je prepare-time API. Po Fazi 8 generuje
  scalar ASM kernel pro podporovany `JAVA_ARRAY` F32/F64 subset; nepodporovane loop/storage/op/dtype
  kombinace zustavaji explicitni prepare-time rejection.
- `Cpu1PreparedFusedElementwiseUnit` vyzaduje non-null `generatedKernel`, pokud je
  `codegenRejectionReason == NONE`.
- `Cpu1FusedElementwiseExecutableUnit` neobsahuje fallback, interpreter, runtime evaluator,
  execute-time factory lookup ani eligibility rozhodovani.
- Overeno kontrakt testem `Cpu1FusedCodegenContractAlignmentTest`.

### Task 4.4: Overit codegen-first fused launch call

Stav: `[x]`

Po uprave launch policy z Tasku 4.1 ma kod v `Cpu1FusedElementwiseExecutableUnit.run(...)`
volat prepared generated kernel primo:

```java
Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(preparedUnit, inputs, output);
preparedUnit.launchPolicy().launch(
        args.elementCount(),
        (start, end) -> preparedUnit.generatedKernel().computeRange(args, start, end)
);
```

Proc:

- Fused executable pouziva stejne chunking a worker policy jako elementwise.
- Hot path nema inner-loop registry lookup ani inner-loop op switch.
- Execute-time dispatch je prime volani prepared generated kernel handle, ne codegen/cache lookup,
  eligibility, fallback ani node-list execution expression nodu uvnitr smycky.
- Executable zustava stabilni i po doplneni dalsich generated variant; meni se jen prepared
  generated kernel handle.

### Task 4.5: Overit pouziti codegen kontraktu z Faze 2.5

Stav: `[x]`

Existujici soubory z Faze 2.5:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenClassSignature.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenLoopKind.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenPlan.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenRejectionReason.java
```

Pozadavky:

- Phase 4 nepridava novy codegen kontrakt; pouziva finalni kontrakt pripraveny ve Fazi 2.5.
- `Cpu1FusedElementwiseExecutableUnit` vola pouze prepared `generatedKernel` handle.
- Factory/cache lookup zustava pouze v prepare.
- Pokud Phase 4 narazi na chybejici contract detail, doplnit ho do Faze 2.5 tasku nebo jejich
  implementace, ne jako execute-time adapter.

Implementovano:

- Nebyl pridan zadny novy codegen kontrakt.
- `Cpu1FusedCodegenKernel` je callable prepared handle s `computeRange(...)`.
- `Cpu1FusedElementwiseExecutableUnit` vola `preparedUnit.generatedKernel().computeRange(...)`
  primo pres obecny range launch.
- `Cpu1FusedCodegenKernelFactory` je v produkcnim `backend.cpu1` pouzivan pouze ve fused prepareru.
- `Cpu1FusedCodegenContractAlignmentTest` chrani, ze factory zustava mimo runtime executable
  cestu a ze accepted prepared unit nemuze existovat bez generated kernel handle.

## Faze 5: Artifact A Trace Integrace

### Task 5.1: Upravit `Cpu1PreparedArtifact`

Stav: `[x]`

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

Implementovano:

- `Cpu1PreparedArtifact` drzi `Cpu1PreparedFusedElementwiseUnit` jako first-class prepared unit.
- Fused constructor vytvari `Cpu1FusedElementwiseExecutableUnit`.
- `preparedFusedElementwiseUnit()` vraci fused prepared unit nebo jasne selze pro jiny typ artifactu.
- `traceContribution(...)` predava fused prepared unit do `Cpu1TraceContributor`.

### Task 5.2: Upravit `Cpu1TraceContributor`

Stav: `[x]`

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

- Route musi byt videt v trace.
- Trace musi rozlisit ASM/codegen route, storage kind, worker count a fused node count.
- Trace smi ukazovat vectorization jen jako vlastnost generated kernelu, ne jako samostatnou
  runtime route. cpu1 fused znamena generated ASM kernel.

Implementovano:

- `Cpu1TraceContributor.traceContribution(...)` prijima `Cpu1PreparedFusedElementwiseUnit`.
- Fused trace vraci kernel id `CPU1_FUSED_ELEMENTWISE`.
- Trace atributy obsahuji node/input count, storage/layout kind, output node id, element count,
  launch workers/chunk size, cost class, requested vectorization, approximate math flags,
  codegen rejection reason, class signature a generated class name.
- `DispatchTraceMetadata` ukazuje requested vectorization, vector width a launch config.
- `FusedTraceMetadata` ukazuje cpu1 fused dispatch family, generated class signature,
  backend `CPU1`, fused node/input count a `NONE` vector fallback reason.

## Faze 6: Prepare Dispatcher Integrace

### Task 6.1: Pridat field do `BackendPrepareDispatcher`

Stav: `[x]`

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

Stav: `[x]`

Soubor:

```text
src/main/java/config/runtime/FusedExecutionPolicy.java
```

Rozsirit record. Stary `allowBackendFallback` zustava zachovany, novy route flag je
samostatna runtime volba:

```java
package config.runtime;

/**
 * Runtime policy for fused elementwise execution.
 *
 * @param allowBackendFallback whether fallback execution is allowed when generated ASM cannot execute a fused region
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
- Default zustava stary CPU fused, dokud profil/test explicitne nezvoli cpu1.
- Pokud je cpu1 route zapnuta a ASM kernel nejde emitovat, prepare ma selhat nebo region odmitnout
  podle explicitni codegen rejection reason.

### Task 6.3: Zapojit route do `BackendPrepareDispatcher`

Stav: `[x]`

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
- Default zustava kompatibilni, protoze `useCpu1Elementwise=false`.
- cpu1 fused lze zapnout benchmarkem/profilem bez zmeny graph optimizeru.
- Pri `useCpu1Elementwise=true` dispatcher vola pouze cpu1 fused preparer; nepodporovany region
  hlasi prepare/codegen rejection reason.

### Task 6.4: Doplnit profile IO route knob

Stav: `[x]`

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
"      \"fusedAllowBackendFallback\": " + fused.allowBackendFallback() + ",\n" +
"      \"fusedUseCpu1Elementwise\": " + fused.useCpu1Elementwise() + "\n"
```

Presne umisteni musi respektovat soucasny JSON formatting v `ExecutionProfileIO`.

Proc:

- Tuning/profile system musi umet cpu1 fused route reprodukovat.
- Bez toho by benchmark mohl pouzit jinou route nez produkcni prepare.

Implementovano:

- `BackendPrepareDispatcher` vlastni `Cpu1FusedElementwisePreparer`.
- `prepareCpuFusedStep(...)` pri `runtimeConfig.fused().useCpu1Elementwise()==true`
  vola pouze cpu1 fused preparer.
- Pri `false` zustava puvodni `cpuPreparer.prepareLoweredFusedStep(...)`.
- Nebyl pridan zadny `try/catch` fallback.
- `Cpu1FusedElementwisePreparer.prepare(...)` bali prepared fused unit do `Cpu1PreparedArtifact`
  a vraci standardni `CompiledNodeExecutionMetadata`.
- `ExecutionProfileIO` cte a zapisuje `fusedUseCpu1Elementwise`.
- Testy pokryvaji default route-off, explicit route-on rejection a profile IO roundtrip/default.

## Faze 7: Test Coverage

### Task 7.1: Pridat IR builder testy

Stav: `[x]`

Novy soubor:

```text
src/test/java/backend/cpu1/fused/Cpu1FusedIrBuilderTest.java
```

Testy:

- `canonicalizesPowTwoToMul`
  - graph: `y = x.pow(2).relu()`
  - fused IR prvni node canonicalizuje `POW(2)` na `MUL`
  - oba `MUL` input refs ukazuji na stejny input ref
- `buildsBroadcastEffectiveStrides`
  - graph: `y = a([N,M]) + b([M])`
  - broadcast bias input ma effective strides `[0, 1]`
  - linearni input zustava linearni
- `capturesScalarParametersForSupportedScalarOps`
  - graph: `y = x.mul(0.25).clampMin(0.125)`
  - scalar parametry se prenesou do `Cpu1FusedScalarParameter`

Proc:

- Nejcastejsi bug bude spatne ref mapovani nebo broadcast stride.
- Testovat IR oddelene od execution zkrati debug.

### Task 7.2: Pridat cpu1 fused execution contract test

Stav: `[~]`

Aktualni stav:

- Plna execution parity sada je blokovana na realnem ASM emitteru z Faze 8.
- V teto fazi se nesmi pridat fake fallback, scalar interpreter ani runtime evaluator jen kvuli testum.
- Aktualni kontrakt bez ASM emitteru je pokryty testy:
  - `src/test/java/backend/cpu1/Cpu1FusedElementwisePreparerTest.java`
  - `src/test/java/backend/cpu1/Cpu1FusedCodegenContractAlignmentTest.java`
  - `src/test/java/backend/cpu1/BackendPrepareDispatcherCpu1FusedRouteTest.java`

Blokovany cilovy soubor po Fazi 8:

```text
src/test/java/backend/cpu1/Cpu1FusedElementwiseExecutionContractTest.java
```

Po ASM emitteru pridat parity testy:

- `fusedReluMulAddMatchesUnfusedF32Array`
- `fusedReluMulAddMatchesUnfusedF64Array`
- `fusedBroadcastWhereMatchesUnfusedF32Array`
- `fusedBf16MatchesWithinTolerance`
- `fusedNativeSegmentMatchesArrayPath`

Proc:

- Tyto testy musi overit skutecne execution pres generated ASM kernel.
- Bez ASM emitteru by slo testy napsat jen pres fake fallback/interpreter, coz je zakazane.

### Task 7.3: Pridat route test pro config knob

Stav: `[x]`

Test soubor:

```text
src/test/java/backend/cpu1/BackendPrepareDispatcherCpu1FusedRouteTest.java
```

Pokryto:

- default `useCpu1Elementwise=false` zustava na stare CPU fused route
- explicit `useCpu1Elementwise=true` jde do cpu1 fused prepareru
- podporovany scalar cpu1 fused region dostane generated kernel bez fallbacku
- test zapina `allowBackendFallback=true`, aby overil, ze cpu1 fused route nema skryty fallback

Proc:

- Route test je povinna cast kompletni migrace.
- Aktualni no-ASM stav musi byt viditelny jako explicitni rejection, ne jako tichy fallback.

### Task 7.4: Pridat nezavisle testy codegen support helperu

Stav: `[~]`

Aktualni stav:

- `Cpu1FusedGeneratedSupport` a `Cpu1FusedMathSupport` zatim nejsou ve zdrojich implementovane.
- Testy helperu se proto nepridavaji v teto fazi.
- Jakmile Faze 8 zavede support helper tridy pro ASM emitter, musi se pridat nezavisle testy nize.

Novy soubor:

```text
src/test/java/backend/cpu1/fused/Cpu1FusedGeneratedSupportTest.java
```

Pokryt:

- `Cpu1FusedGeneratedSupport.bf16ToFloat(...)` a `floatToBf16(...)` proti existujicimu BF16
  kontraktu v tensor/storage kodu.
- `boolFromByte(...)` a `boolToByte(...)` pro canonical bool reprezentaci.
- F32/F64 dtype konverze a NaN/Infinity chovani, pokud konverze muze byt emitovana jako helper call.
- `Cpu1FusedMathSupport` exact helpers `exp`, `log`, `tanh`, `pow` proti `Math.*` s toleranci
  podle dtype.
- Fast approximation helpers, pokud budou zapojene, proti dokumentovane toleranci a monotonicite
  tam, kde ji aproximace slibuje.

Proc:

- Support metody jsou hand-written Java volane staticky z generated bytecode, proto se maji testovat
  samostatne bez ASM tridy.
- Testy nesmi zavest runtime op dispatch; helper API ma byt primitive/dtype oriented.

## Faze 8: Fused ASM/Codegen Hot Path

### ASM local temporaries invariant

Stav: `[x]`

Implementovano:

- `Cpu1FusedElementwiseExecutableUnit.scratchBufferSpec()` zustava `none()`.
- Generated scalar ASM path drzi fused mezivysledky v JVM local slotech.
- Nebyly pridane zadne fused scratch arrays ani per-node temporary buffers.

Pozadavek:

- Nepridavat `fusedNumericScratch`, `fusedBoolScratch` ani podobne scratch arrays.
- Mezivysledky fused expression v generated kernelu maji byt JVM local slots nebo Vector API
  local variables emitovane ASM tridu.
- `Cpu1FusedElementwiseExecutableUnit.scratchBufferSpec()` ma zustat `Cpu1ScratchBufferSpec.none()`,
  pokud konkretni budouci generated algoritmus neprokaze jinou docasnou pametovou potrebu.

Proc:

- Scratch arrays byly potreba jen pro zrusenou scratch-array variantu.
- ASM kernel ma generovat konkretni smycku a mezivysledky drzet v locals, ne v heap poli.
- Tim se vyhneme per-chunk alokacim i zbytecnemu runtime buffer kontraktu.

### Task 8.1: Overit codegen-first generated-kernel route

Stav: `[x]`

Soubor:

```text
src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java
```

Pozadovany execute call:

```java
preparedUnit.launchPolicy().launch(
        args.elementCount(),
        (start, end) -> preparedUnit.generatedKernel().computeRange(args, start, end)
);
```

Proc:

- Execute stale vola jeden prepared generated runner.
- Prepare/codegen eligibility uz zarucila, ze prepared unit ma concrete generated kernel.
- Produkcni route je generated kernel.
- Codegen eligibility je pripravena oddelene od cost; runtime uz eligibility flags necte kvuli rozhodovani.
- Execute nesmi volat codegen factory, class cache, supports/eligibility helper, scalar interpreter,
  vector fallback, backend fallback ani runtime evaluator.
- Neexistuje jina runtime cesta uvnitr cpu1 fused path.

### Task 8.2: Napojit emitter na ASM/codegen plan kontrakt

Stav: `[~]`

Aktualni implementacni stav:

- `Cpu1FusedCodegenKernelFactory.prepareKernel(...)` uz pro accepted plan generuje realnou ASM
  tridu a vraci callable `Cpu1FusedCodegenKernel`.
- Emitter pouziva existujici `Cpu1FusedCodegenPlan`, `Cpu1FusedCodegenLoopKind` a
  `Cpu1FusedCodegenClassSignature`.
- Podporovane jsou `JAVA_ARRAY + FLOAT32/FLOAT64 + CONTIGUOUS_SCALAR/STRIDED_SCALAR`.
- `CONTIGUOUS_VECTOR` je v teto prvni implementaci explicitne prepare-time rejected pres
  `UNSUPPORTED_LOOP_KIND`.
- `MEMORY_SEGMENT` je v teto prvni implementaci explicitne prepare-time rejected pres
  `UNSUPPORTED_STORAGE_KIND`.

Existujici soubory z Faze 2.5:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenPlan.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenLoopKind.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenClassSignature.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenRejectionReason.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java
```

Proc:

- Codegen route uz ma vlastni explicitni plan z Faze 2.5.
- Faze 8 nesmi znovu definovat prepare-time kontrakt; ma vyplnit realny ASM emitter body,
  class loading/cache implementaci a supported subset nad existujicimi typy.
- Plan slouzi jako ASM emitter vstup a obsahuje structural class signature pro class/template reuse.
- `CONTIGUOUS_VECTOR` ma v cilovem stavu generovat vector hlavni smycku a generated scalar tail
  ve stejne tride. Aktualne je prepare-time rejected bez runtime fallbacku.
- `CONTIGUOUS_SCALAR` generuje scalar ASM loop pro contiguous data bez Vector API.
- `STRIDED_SCALAR` generuje scalar ASM loop s offset math pro strided/broadcast access.
- Strided vectorization muze zustat nepodporovana v prvni iteraci; strided data jako cele se
  nesmi automaticky odmitnout, pokud generated scalar ASM umi korektni offset math.
- Class signature je structural: canonical expression/layout/storage/dtype/loop signature bez
  graph node ids, unit ids a bez konkretni scalar values.
- Pokud generated bytecode vola support metody, canonical signature string obsahuje `supportAbi=...`
  a stabilni serazeny seznam helper targetu. Tyto targety jsou soucast template/cache identity,
  ne prepared scalar bindingu.

### Task 8.3: Pridat helper/intrinsic support vrstvu pro ASM emitovani

Stav: `[~]`

Aktualni implementacni stav:

- Pridane jsou pozadovane tridy `Cpu1FusedAsmMethodEmitter`, `Cpu1FusedAsmCallEmitter`,
  `Cpu1FusedAsmIntrinsicRegistry`, `Cpu1FusedGeneratedSupport` a `Cpu1FusedMathSupport`.
- `Cpu1FusedAsmIntrinsicRegistry` se pouziva pri prepare/codegen eligibility, ne v hot path.
- Support metody jsou primitive static Java methods bez `Operation`, node id nebo plan dispatch
  parametru.
- Support ABI a serazeny helper target list jsou serializovane do canonical class signature pro
  aktualne emitovane math helper calls (`relu`, `abs`, `min`, `max` pro F32/F64).

Nove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmMethodEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmCallEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmIntrinsicRegistry.java
src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedGeneratedSupport.java
src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport.java
```

Implementacni pozadavky:

- `Cpu1FusedAsmCallEmitter.emitInvokeStatic(...)` musi byt jedine misto, kde expression/loop
  emittery rucne skladaji `visitMethodInsn(INVOKESTATIC, ...)` pro support helpery.
- `Cpu1FusedAsmIntrinsicRegistry` mapuje op/dtype/access primitive na jednu z emit strategii:
  direct bytecode, static call do `Cpu1FusedGeneratedSupport`, static call do
  `Cpu1FusedMathSupport`, nebo prepare-time unsupported.
- Registry se pouziva pouze pri generovani bytecode; generated kernel nesmi registry volat pri
  `computeRange`.
- `Cpu1FusedGeneratedSupport` a `Cpu1FusedMathSupport` musi mit primitive static metody bez
  `Operation.OpType`, node id, planu nebo per-node dispatch parametru.
- `Cpu1FusedCodegenClassSignature` builder musi do canonical signature stringu pridat support ABI
  a helper targety pro kazdy static call, ktery generated trida emituje.

Direct emit povinne pokryva:

- simple arithmetic/comparison;
- `MIN`, `MAX`, `ABS`, `RELU`, `CLAMP_MIN`, `CLAMP_MAX`;
- scalar field load a jednoduchy F32/F64/BF16 scalar parameter binding;
- linear contiguous F32/F64 load/store, pokud nepotrebuje konverzni helper.

Static helper call povinne pokryva nebo explicitne rejectuje:

- `exp`, `log`, `tanh`, `erf`, obecny `pow`;
- fast approximations pro `exp`/`tanh` nebo dalsi pozdeji pridane approximations;
- BF16 load/store conversion a rounding policy;
- boolean load/store conversion;
- dtype conversion mezi podporovanymi generated dtype;
- offset helper pouze mimo hot inner loop, pokud je potreba pro setup.

Proc:

- Tezka matematika a bitove konverze zustanou citelne a samostatne testovatelne v Java helper
  metodach.
- Generated kernel zustava ASM-only: static helper call je soucast generated bytecode, ne runtime
  fallback nebo interpreter.
- Support ABI/dependency podpis brani cache reuse pres nekompatibilni helper targety.

### Task 8.4: Implementovat cpu1 ASM emitter pro contiguous a strided scalar subset

Stav: `[~]`

Aktualni implementacni stav:

- Pridane jsou `Cpu1FusedAsmClassEmitter`, `Cpu1FusedAsmExpressionEmitter`,
  `Cpu1FusedAsmLoopEmitter` a `Cpu1FusedGeneratedClassLoader`.
- Generated trida implementuje `Cpu1FusedElementwiseRangeRunner`.
- Constructor binduje scalar hodnoty do instancnich fields; scalar values nejsou soucasti
  structural class signature.
- `CONTIGUOUS_SCALAR` generuje scalar loop pro `JAVA_ARRAY` F32/F64.
- `STRIDED_SCALAR` generuje rank-specific offset math primo v loopu podle runtime
  `Cpu1TensorView` base/stride locals, vcetne zero-stride broadcast vstupu.
- Neni zavedena zadna skryta contiguous materializace.
- `CONTIGUOUS_VECTOR`, BF16, MEMORY_SEGMENT a transcendental intrinsics zustavaji prepare-time
  rejected.

Nove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmClassEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmExpressionEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmLoopEmitter.java
src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedGeneratedClassLoader.java
```

V ramci Faze 8 musi byt doplnena realna ASM generated implementace aspon pro tyto op pro
podporovane dtype/storage/access kombinace:

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

Loop kinds:

- `CONTIGUOUS_VECTOR`: contiguous inputs/output, Vector API bytecode pro hlavni smycku a generated scalar tail.
  Aktualne jeste neni implementovane a je prepare-time rejected.
- `CONTIGUOUS_SCALAR`: contiguous inputs/output, scalar ASM loop bez Vector API.
- `STRIDED_SCALAR`: strided nebo broadcast input/output access, scalar ASM loop s generated offset math.

Broadcast/strided pravidla:

- Broadcast a strided access se reprezentuji v generated offset math podle `Cpu1FusedInputPlan`
  a `Cpu1TensorView` strides/offsetu.
- Strided vectorization muze zustat nepodporovana v prvni iteraci; to znamena volbu
  `STRIDED_SCALAR`, ne automaticke odmítnuti celeho fused regionu.
- cpu1 fused runtime nesmi skryte materializovat contiguous kopii. Pokud se pozdeji zvoli
  materializace, musi vzniknout explicitni graph/lowering/memory-planning krok mimo fused runtime.

Scalar parametry v ASM/codegen hot path:

- Structural class/template signature nezahrnuje konkretni scalar hodnoty.
- F32/BF16 prepared generated kernel instance binduje F32 scalar hodnoty do fields nebo constructor state.
- F64 prepared generated kernel instance binduje F64 scalar hodnoty do fields nebo constructor state.
- Helper method targety a support ABI patri do canonical class/template signature stringu, protoze
  meni emitted bytecode targety; nejsou to scalar hodnoty a nepatri do instance-only bindingu.
- Pokud scalar op voli ruzny helper podle dtype nebo approximation policy, helper target a policy
  musi byt v signature. Konkretni scalar cislo stale zustava bindovane na prepared kernel instance.
- Scalar hodnoty maji nutit unikátní generated class jen tehdy, kdyz plan explicitne obhaji
  constant embedding pro konkretni hot path a popise trade-off cache cardinality vs vykon.

Povinna generated F32 contiguous smycka musi byt semanticky tvaru:

```text
for (int i = startInclusive; i < vectorUpper; i += speciesLength) {
    // generated loads for each external input
    // generated expression bytecode for this exact Cpu1FusedExpressionPlan
    // generated output store
}
for (int i = vectorUpper; i < endExclusive; i++) {
    // generated scalar tail for the same expression
}
```

Povinna generated strided/broadcast scalar smycka musi byt semanticky tvaru:

```text
for (int logical = startInclusive; logical < endExclusive; logical++) {
    // generated output offset math for logical index
    // generated input offset math, including zero-stride broadcast dimensions
    // generated scalar expression bytecode for this exact structural expression
    // generated output store
}
```

Proc:

- Complete migration musi koncit u concrete generated ASM smycek.
- ASM/generated route je nutna pro paritu se starym ASM fused.
- Tail smycka je take generovana; nesmi pro kazdy tail prvek volat runtime node-switch executor.
- Codegen rejection reason se urci pri prepare. Emitter nesmi znovu klasifikovat support podle `Operation`
  nebo menit cost rozhodnuti.

### Task 8.5: Dokoncit codegen kernel factory

Stav: `[x]`

Nove soubory:

```text
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenClassSignature.java
src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenLoopKind.java
```

Minimalni kontrakt:

```java
package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.kernels.fused.Cpu1FusedElementwiseRangeRunner;

public record Cpu1FusedCodegenKernel(
        Cpu1FusedCodegenClassSignature classSignature,
        String generatedClassName,
        Cpu1FusedElementwiseRangeRunner rangeRunner
) {
    public Cpu1FusedCodegenKernel {
        if (classSignature == null) {
            throw new IllegalArgumentException("classSignature cannot be null");
        }
        if (generatedClassName == null || generatedClassName.isBlank()) {
            throw new IllegalArgumentException("generatedClassName cannot be blank");
        }
        if (rangeRunner == null) {
            throw new IllegalArgumentException("rangeRunner cannot be null");
        }
    }

    public void computeRange(Cpu1FusedKernelArgs args, int startInclusive, int endExclusive) {
        rangeRunner.computeRange(args, startInclusive, endExclusive);
    }
}
```

Factory skeleton:

```java
package backend.cpu1.kernels.fused.codegen;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class Cpu1FusedCodegenKernelFactory {
    private static final ConcurrentMap<Cpu1FusedCodegenClassSignature, CachedTemplate> CACHE =
            new ConcurrentHashMap<>();

    private Cpu1FusedCodegenKernelFactory() {
    }

    public static Cpu1FusedCodegenKernel prepareKernel(Cpu1FusedCodegenPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        CachedTemplate template = templateFor(plan);
        return template.instantiate(plan);
    }

    private static CachedTemplate templateFor(Cpu1FusedCodegenPlan plan) {
        Cpu1FusedCodegenClassSignature signature = plan.classSignature();
        return CACHE.computeIfAbsent(
                signature,
                ignored -> backend.cpu1.kernels.fused.codegen.asm.Cpu1FusedAsmClassEmitter.emitTemplate(plan)
        );
    }

    public interface CachedTemplate {
        Cpu1FusedCodegenClassSignature signature();

        Cpu1FusedCodegenKernel instantiate(Cpu1FusedCodegenPlan plan);
    }
}
```

Class signature skeleton:

```java
package backend.cpu1.kernels.fused.codegen;

public record Cpu1FusedCodegenClassSignature(String canonicalSignature) {
    public Cpu1FusedCodegenClassSignature {
        if (canonicalSignature == null || canonicalSignature.isBlank()) {
            throw new IllegalArgumentException("canonicalSignature cannot be blank");
        }
    }
}
```

Proc:

- Codegen route ma vlastni jmeno a vlastni cache key.
- `codegen/` reprezentuje runtime cestu, ktera kernel vytvari/generuje a pote ho spousti bez node-list execution v hot path.
- `codegen/` reprezentuje jedinou cpu1 fused runtime cestu.
- Factory je prepare-time API. `Cpu1FusedElementwiseExecutableUnit` ji nesmi volat.
- `Cpu1FusedCodegenClassSignature` je maly value object nad canonical stringem, ktery je jediny
  zdroj pravdy pro cache identity.
- `canonicalSignature` je structural class/template signature: expression op/ref shape, dtype,
  storage kind, layout/access model, loop kind, support ABI a helper method targety.
  Neobsahuje graph node ids, unit ids ani scalar values.
- `instantiate(plan)` binduje scalar hodnoty do prepared kernel instance/fields. Constant embedding je
  pozdejsi optimalizace jen s explicitnim zduvodnenim.

### Task 8.6: Pridat codegen rejection reason do prepared unit a trace

Stav: `[~]`

Aktualni implementacni stav:

- `Cpu1FusedCodegenRejectionReason` existuje a obsahuje konkretni prepare-time duvody:
  `NONE`, `UNSUPPORTED_DTYPE`, `UNSUPPORTED_OPERATION`, `UNSUPPORTED_INTRINSIC`,
  `UNSUPPORTED_LAYOUT_OR_ACCESS`, `UNSUPPORTED_STORAGE_KIND`, `UNSUPPORTED_LOOP_KIND` a
  `UNSUPPORTED_SCALAR_BINDING`.
- `MISSING_ASM_EMITTER` byl po zavedeni scalar ASM emitteru odstranen jako neplatny mezistav.
- Accepted prepared fused unit smi vzniknout jen s `NONE`.
- Rejected plan konci v prepare pres `Cpu1FusedCodegenKernelFactory.rejection(...)`; execute podle
  rejection reason nerozhoduje.

Do `Cpu1PreparedFusedElementwiseUnit` mit prepare-time trace field:

```java
private final Cpu1FusedCodegenRejectionReason codegenRejectionReason;
```

Poznamka: prepared executable fused unit smi vzniknout jen s `NONE`. Nenulovy rejection reason je
duvod, proc prepare cpu1 fused region odmitne; execute podle tohoto fieldu nic nerozhoduje.

Do `Cpu1FusedElementwisePreparer` pridat prepare-time eligibility helper. Helper smi cist
`Operation.resultKind()` a `Operation.controlTrait()` ze `sourceOperations`, ale vraci jen
konkretni rejection reason a neuklada trait snapshot:

```java
private static Cpu1FusedCodegenRejectionReason codegenRejectionReason(
        Cpu1FusedCodegenPlan codegenPlan,
        List<Operation> sourceOperations,
        DataType computeType
) {
    if (codegenPlan.loopKind() != Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
            && codegenPlan.loopKind() != Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR
            && codegenPlan.loopKind() != Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
        return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LOOP_KIND;
    }
    if (codegenPlan.loopKind() == Cpu1FusedCodegenLoopKind.STRIDED_SCALAR
            && !supportsGeneratedOffsetMath(codegenPlan.expressionPlan())) {
        return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LAYOUT;
    }
    if (!supportsGeneratedDType(computeType)) {
        return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
    }
    for (int i = 0; i < codegenPlan.expressionPlan().nodes().size(); i++) {
        Operation operation = sourceOperations.get(i);
        if (operation.resultKind() != Operation.OpResultKind.NUMERIC
                && operation.resultKind() != Operation.OpResultKind.BOOLEAN) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_RESULT_KIND;
        }
        if (operation.controlTrait() == Operation.OpControlTrait.SELECT_MASK
                && codegenPlan.expressionPlan().nodes().get(i).opType() != Operation.OpType.WHERE) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_CONTROL_TRAIT;
        }
        if (operation.controlTrait() == Operation.OpControlTrait.BOOL_LOGIC
                && !supportsGeneratedBoolLogic(codegenPlan.expressionPlan().nodes().get(i))) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_CONTROL_TRAIT;
        }
        if (!supportsCodegenOp(codegenPlan.expressionPlan().nodes().get(i))) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_OP;
        }
        if (!supportsCodegenIntrinsic(codegenPlan.expressionPlan().nodes().get(i), computeType)) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_INTRINSIC;
        }
    }
    return Cpu1FusedCodegenRejectionReason.NONE;
}
```

Trace attrs:

```java
attrs.put("cpu1FusedCodegenRejectionReason", unit.codegenRejectionReason().name());
```

Proc:

- Stary CPU fused trace uz mel rejection reason; cpu1 musi reportovat codegen eligibility primo.
- cpu1 musi byt stejne vysvetlitelny.
- Rejection reason je eligibility vysledek, ne cost. Boolean/select/logical nody zustavaji
  levne nebo drahe podle `operation.computationalCost()`; ASM/codegen nepodpora je
  vysvetlena tady.
- `UNSUPPORTED_INTRINSIC` znamena, ze op je obecne fusable, ale aktualni direct-bytecode/helper
  mapping v `Cpu1FusedAsmIntrinsicRegistry` jej pro dany dtype/support policy neumi emitovat.
- BF16, BOOL, broadcast a strided access nejsou automaticke rejection reasons. Odmitaji se pouze
  tehdy, kdyz aktualni generated dtype/op/control/layout support konkretni pripad nepokryva.

## Faze 9: Profile IO A Tuning Knobs

### Task 9.1: Pouzit existujici fused thresholdy z `CpuKernelConfig`

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

### Task 9.2: Pridat fused cpu1 route do profile JSON

Stav: `[ ]`

Soubor:

```text
src/main/java/config/profile/ExecutionProfileIO.java
```

Viz Task 6.4. Tento task je zde duplicitne uvedeny jako checkpoint pro profile/tuning fazi a musi byt oznacen `[x]` pouze pokud:

- read podporuje `fusedUseCpu1Elementwise`;
- write serializuje `fusedUseCpu1Elementwise`;
- chybejici pole pouzije fallback profile value; standardni runtime defaults maji hodnotu `false`;
- existuje test pro backward compatible nacitani stareho profilu.

### Task 9.3: Pridat profile IO test

Stav: `[ ]`

Upravit existujici profile IO testy nebo pridat:

```text
src/test/java/config/profile/FusedExecutionPolicyProfileIOTest.java
```

Testy:

```java
@Test
void profileRoundTripsCpu1FusedRouteFlag() {
    RuntimeConfig base = RuntimeConfig.inferenceDefaults(DataType.FLOAT32);
    RuntimeConfig runtime = new RuntimeConfig(
            base.kernel(),
            base.approximation(),
            base.blas(),
            base.conv2d(),
            base.fused().withUseCpu1Elementwise(true),
            base.accelerator(),
            base.cpuStorageProfile(),
            base.nativeCpuFailurePolicy(),
            base.deviceTransferPolicy(),
            base.nativeCpuMemory(),
            base.bfloat16TrainingPolicy()
    );
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

## Faze 10: Parity Benchmarky Proti Staremu CPU Fused

### Task 10.1: Pridat benchmark/test harness pro old CPU fused vs cpu1 fused

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
case,dtype,storage,elements,oldCpuMedianMs,cpu1MedianMs,ratio,cpu1CodegenRejectionReason
```

Proc:

- Complete migration musi mit vykonovou evidenci.
- Pokud cpu1 route neni rychlejsi, trace musi rict proc.

### Task 10.2: Pridat correctness parity pro route on/off

Stav: `[ ]`

Test:

```java
@Test
void cpu1FusedRouteMatchesOldCpuFusedRoute() {
    Tensor yOld = graph.compile().prepare(runtimeWithCpu1Fused(config, false)).execute();
    Tensor yCpu1 = graph.compile().prepare(runtimeWithCpu1Fused(config, true)).execute();
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

### Task 10.3: Aktualizovat dokument podle benchmark vysledku

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

## Faze 11: Finalni Overeni

### Task 11.1: Kompilace

Stav: `[ ]`

Spustit:

```bash
./gradlew classes
```

### Task 11.2: Cilene testy cpu1 fused

Stav: `[ ]`

Spustit:

```bash
./gradlew test --tests backend.cpu1.Cpu1FusedElementwiseExecutionContractTest --tests backend.cpu1.fused.Cpu1FusedIrBuilderTest --tests backend.cpu1.fused.Cpu1FusedGeneratedSupportTest
```

### Task 11.3: Regression testy dotcenych oblasti

Stav: `[ ]`

Spustit:

```bash
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest --tests backend.cpu1.Cpu1MseLossExecutionContractTest --tests graph.compile.planning.region.DefaultRegionOptimizerTest --tests graph.compile.planning.partition.CpuNaturalExecutionRegionPlannerTest
```

### Task 11.4: Stare fused regression testy

Stav: `[ ]`

Spustit stare fused testy:

```bash
./gradlew test --tests FusedExecutionModesTest --tests OptimizerFuseTest
```

Proc:

- Route knob je soucast kompletni migrace, proto jsou stare fused regression testy povinne.
- Testy musi projit pro default starou route i pro explicitni cpu1 route, kde dava smysl.

### Task 11.5: Benchmark/test evidence

Stav: `[ ]`

Spustit nebo dolozit:

```bash
./gradlew test --tests debug.Cpu1FusedParityBenchmarkTest
```

Pokud debug benchmark neni vhodny pro CI, vysledky ulozit pouze do dokumentu nebo konzole, ne do `profiles/platform/*`, pokud uzivatel explicitne nechce aktualizovat kanonicke profily.

### Task 11.6: Diff hygiene

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

- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedAccessKind.java`
- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedScalarParameter.java`
- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedInputPlan.java`
- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedNodePlan.java`
- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedExpressionPlan.java`
- [x] `src/main/java/backend/cpu1/fused/ir/Cpu1FusedIrBuilder.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1FusedElementwisePreparer.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1PreparedFusedElementwiseUnit.java`
- [ ] `src/main/java/backend/cpu1/prepare/dispatch/Cpu1FusedDispatchDecision.java`
- [x] `src/main/java/backend/cpu1/exec/Cpu1FusedKernelArgs.java`
- [x] `src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java`
- [x] `src/main/java/backend/cpu1/launch/Cpu1RangeTask.java`
- [x] `src/main/java/backend/cpu1/kernels/fused/Cpu1FusedElementwiseRangeRunner.java`

Faze 2.5 codegen contract soubory:

- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernel.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenKernelFactory.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenClassSignature.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenLoopKind.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenPlan.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/Cpu1FusedCodegenRejectionReason.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedGeneratedSupport.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmClassEmitter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmMethodEmitter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmCallEmitter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmIntrinsicRegistry.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmExpressionEmitter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedAsmLoopEmitter.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/codegen/asm/Cpu1FusedGeneratedClassLoader.java`
- [ ] `src/main/java/backend/cpu1/kernels/fused/tuning/Cpu1FusedTuningClassifier.java`
- [ ] `src/test/java/backend/cpu1/fused/Cpu1FusedIrBuilderTest.java`
- [ ] `src/test/java/backend/cpu1/fused/Cpu1FusedGeneratedSupportTest.java`
- [x] `src/test/java/backend/cpu1/Cpu1FusedCodegenContractAlignmentTest.java`
- [ ] `src/test/java/backend/cpu1/Cpu1FusedElementwisePreparerTest.java`
- [ ] `src/test/java/backend/cpu1/Cpu1FusedElementwiseExecutionContractTest.java`
- [ ] `src/test/java/backend/prepare/Cpu1FusedPrepareRoutingTest.java`
- [ ] `src/test/java/config/profile/FusedExecutionPolicyProfileIOTest.java`
- [ ] `src/test/java/debug/Cpu1FusedParityBenchmarkTest.java`

### Upravy existujicich souboru

- [ ] `src/main/java/backend/cpu1/exec/Cpu1ScratchBuffer.java`
- [ ] `src/main/java/backend/cpu1/exec/Cpu1ScratchBufferSpec.java`
- [x] `src/main/java/backend/cpu1/launch/Cpu1LaunchPolicy.java`
- [x] `src/main/java/backend/cpu1/launch/Cpu1SingleThreadLaunch.java`
- [x] `src/main/java/backend/cpu1/launch/Cpu1ParallelLaunch.java`
- [ ] `src/main/java/backend/cpu1/launch/Cpu1RangeLauncher.java`
- [x] `src/main/java/backend/cpu1/exec/Cpu1ElementwiseExecutableUnit.java`
- [ ] `src/main/java/backend/cpu1/prepare/Cpu1PreparedElementwiseUnit.java`
- [ ] `src/main/java/backend/cpu1/prepare/dispatch/Cpu1DispatchPolicy.java`
- [x] `src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java`
- [x] `src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java`
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
- Complete migration vyzaduje cpu1-native ASM/codegen hot path ve Fazi 8.
- ASM emitter ma byt znovu napsany nad `Cpu1FusedExpressionPlan`, `Cpu1TensorView` a cpu1 storage/layout kontrakty.
- Neimportovat stary `backend.cpu.fused.asm` jako compatibility layer.

### Stary native segment helper

Neprenest.

Proc:

- cpu1 uz ma `Cpu1TensorView`, runtime native slots a `ExecutionContext.requireNativeOutputStorage(...)`.

## Zadny Mimo-Plan Stav

Tento plan nema zamerne zadne sekce mimo hlavni implementaci. Veci, ktere byly v predchozi verzi mimo hlavni implementaci, jsou soucasti povinnych fazi:

- ASM prepare contract alignment je Faze 2.5;
- ASM/codegen hot path je Faze 8 a obsahuje invariant pro fused temporaries;
- route knob a profile IO jsou Faze 6 a Faze 9;
- benchmark parity proti staremu CPU fused je Faze 10;
- finalni overeni je Faze 11.

Pokud implementace narazi na novou prekazku, dokument se ma aktualizovat jako novy task uvnitr techto fazi, ne jako nova samostatna sekce mimo plan.

## Finalni Akceptacni Kriteria

Implementace je hotova, kdyz plati:

- [ ] cpu1 ma vlastni fused IR, neimportuje `backend.cpu.fused.*`
- [ ] cpu1 fused preparer pripravuje `Cpu1PreparedFusedElementwiseUnit`
- [ ] Faze 2.5 definuje finalni ASM-only prepare-time codegen kontrakt a prepared
  `generatedKernel` handle pred executable unit
- [x] runtime executable je `backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnit`
- [ ] fused execution pouziva `Cpu1TensorView`, ne `TensorInternalAccess` jako primarni runtime kontrakt
- [x] prepare generuje nebo z cache ziska ASM class/template a uklada prepared generated kernel handle
  do `Cpu1PreparedFusedElementwiseUnit`
- [~] generated ASM smi volat jen stabilni static support helpers pres `INVOKESTATIC`; helpery jsou
  hand-written Java a testovane nezavisle
- [x] class/template canonical signature string zahrnuje support ABI/helper targety, pokud generated
  bytecode vola support metody
- [x] execute neprovadi codegen, cache lookup, eligibility, scalar interpreter, vector fallback,
  backend fallback ani runtime evaluator
- [x] JAVA_ARRAY podporovany F32/F64 scalar subset bezi pres ASM-generated concrete kernels
- [ ] MEMORY_SEGMENT podporovany subset bezi pres ASM-generated concrete kernels
- [~] podporovane loop kinds zahrnuji `CONTIGUOUS_SCALAR` a `STRIDED_SCALAR`; `CONTIGUOUS_VECTOR`
  zustava prepare-time rejected do vector emitter faze
- [x] broadcast/strided access je reprezentovan generated offset math, ne skrytou contiguous materializaci
- [ ] BF16, bool/logical, broadcast a strided cases jsou odmitnute jen pokud konkretni dtype/op/control/layout
  neni podporovan generated ASM codegenem
- [ ] parallel launch funguje pres stejnou cpu1 launch policy
- [x] fused temporaries nepouzivaji per-range alokace ani scratch arrays; generated ASM je drzi v locals
- [x] IR builder testy pokryvaji pow canonicalizaci, broadcast effective strides a scalar parametry
- [~] ASM/codegen fused hot path existuje pro podporovany contiguous scalar a strided scalar subset;
  contiguous vector zustava follow-up
- [x] nepodporovane codegen cases maji explicitni `Cpu1FusedCodegenRejectionReason`
- [x] cpu1 fused ma pouze ASM-generated concrete kernels a zadnou jinou runtime cestu
- [x] `FusedExecutionPolicy.useCpu1Elementwise` existuje a defaultuje na `false`
- [x] `ExecutionProfileIO` umi route flag nacist i zapsat
- [x] route-on/route-off no-ASM kontrakt je otestovany bez ticheho fallbacku
- [ ] route-on/route-off correctness parity proti staremu CPU fused je otestovana
- [ ] plne cpu1 fused execution parity testy jsou doplnene po realnem ASM emitteru
- [ ] codegen support helper tridy existuji a maji nezavisle testy
- [ ] benchmark parity evidence je doplnena v dokumentu
- [x] trace ukazuje `CPU1_FUSED_ELEMENTWISE`
- [x] trace ukazuje storage, node count, launch workers, vectorization a codegen rejection reason
- [ ] zadne lokalni profily/IDE soubory nejsou soucasti commitu
- [x] `./gradlew classes` projde
- [x] cilene cpu1 fused testy projdou
- [ ] stare fused regression testy projdou
- [ ] dokument neobsahuje zadne migracni kroky mimo tento plan
