# 119. General backend-neutral MATMUL_EPILOGUE IR plan

## Stav a sledovani

Status: `[ ] planned`

Legenda:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` hotovo a overeno
- `[deferred]` zamerne odlozeno mimo tento plan

Implementacni tracking:

- [ ] Faze 0: inventory, guard testy a ochrana stavajicich specializaci
- [ ] Faze 1: backend-neutral payload model pro `RegionSpecializationCandidate`
- [ ] Faze 2: `MatmulEpiloguePlan` a typed expression tree
- [ ] Faze 3: nahrada tri matmul enum specializaci jednim `MATMUL_EPILOGUE`
- [ ] Faze 4: generic detector/builder pro `MATMUL`, `MATMUL + bias`, `MATMUL + bias + relu`, `LINEAR`, `LINEAR + relu`
- [ ] Faze 5: cpu1 prepare route pres typed epilogue plan
- [ ] Faze 6: cpu1 prepared epilogue runtime kontrakt bez enum exploze
- [ ] Faze 7: trace, diagnostika a benchmark/report metadata
- [ ] Faze 8: test migrace a odstraneni obsolete kodu
- [ ] Faze 9: finalni overeni

Detailni task tracking:

- [ ] 0.1: zachytit aktualni inventory region specializaci a source odkazu na
  `MATMUL_RELU`, `MATMUL_ADD_BIAS`, `MATMUL_ADD_BIAS_RELU`
- [ ] 0.2: doplnit/overit guard testy pro soucasne tvary
  `MATMUL -> RELU`, `MATMUL -> ADD`, `MATMUL -> ADD -> RELU`, `LINEAR`,
  `LINEAR -> RELU`
- [ ] 0.3: doplnit negativni guard test pro publikovany nebo materializovany
  intermediate
- [ ] 1.1: pridat `RegionSpecializationPayload`
- [ ] 1.2: pridat `NoRegionSpecializationPayload`
- [ ] 1.3: rozsirit `RegionSpecializationCandidate` o typed payload bez
  `Object` a bez backend-specific typu
- [ ] 1.4: prepnout `MseLossSpecializationDetector` na prazdny payload
- [ ] 2.1: pridat `MatmulCoreKind`
- [ ] 2.2: pridat `MatmulCoreSpec`
- [ ] 2.3: pridat epilogue input/op enumy
- [ ] 2.4: pridat sealed `MatmulEpilogueExpression`
- [ ] 2.5: pridat `MatmulEpiloguePlan`
- [ ] 2.6: pridat `MatmulEpiloguePlanBuilder`
- [ ] 3.1: nahradit konkretni matmul specialization kindy za
  `MATMUL_EPILOGUE`
- [ ] 3.2: odstranit source switch vetve nad starymi kindy
- [ ] 3.3: zajistit, ze stare kindy nezustanou jako compatibility path
- [ ] 4.1: pridat `MatmulEpilogueSpecializationDetector`
- [ ] 4.2: pokryt `MATMUL -> RELU`
- [ ] 4.3: pokryt `MATMUL -> ADD(bias)`
- [ ] 4.4: pokryt `MATMUL -> ADD(bias) -> RELU`
- [ ] 4.5: pokryt `LINEAR` s biasem jako `MatmulCoreKind.LINEAR`
- [ ] 4.6: pokryt `LINEAR -> RELU`
- [ ] 4.7: odstranit obsolete konkretni detektory
- [ ] 5.1: prepnout `RegionSpecializationPlanner` na jeden generic detector
- [ ] 5.2: prepnout `DefaultRegionSpecializationCapability` na payload-aware
  `MATMUL_EPILOGUE`
- [ ] 5.3: prepnout `BackendPrepareDispatcher` na `prepareCpu1MatmulEpilogue`
- [ ] 5.4: predavat `MatmulEpiloguePlan` a core `CompiledNode` do cpu1
  prepareru
- [ ] 6.1: pridat `Cpu1PreparedMatmulEpilogue`
- [ ] 6.2: pridat `Cpu1MatmulEpilogueClassifier`
- [ ] 6.3: pridat `Cpu1MatmulPreparer.prepareEpilogue`
- [ ] 6.4: upravit `Cpu1PreparedMatmulUnit` tak, aby nesl prepared epilogue
  kontrakt
- [ ] 6.5: odstranit stare cpu1 helpery pro konkretni graph kindy, pokud uz
  nebudou pouzite
- [ ] 7.1: pridat region trace summary pro `MATMUL_EPILOGUE`
- [ ] 7.2: pridat cpu1 trace metadata pro core, expression a bias node
- [ ] 7.3: zachovat `cpu1MatmulPostOp` jen jako cpu1 executable detail
- [ ] 8.1: aktualizovat region/planning testy na payload assertions
- [ ] 8.2: aktualizovat cpu1 matmul/linear execution contract testy
- [ ] 8.3: pridat unsupported-payload prepare rejection test
- [ ] 8.4: pridat source hygiene kontrolu pro odstranene graph-level kindy
- [ ] 9.1: spustit cilene Gradle testy
- [ ] 9.2: spustit `./gradlew classes`
- [ ] 9.3: spustit `git diff --check`
- [ ] 9.4: zapsat finalni stav implementace do tracking sekce

Tento dokument je pouze plan. Neimplementuje zdrojove zmeny.

## Cil

Zavest obecny backend-neutral IR pro matmul epilogue specializace:

```text
RegionSpecializationKind.MATMUL_EPILOGUE
  payload: MatmulEpiloguePlan
    core: MatmulCoreSpec
    expression: MatmulEpilogueExpression
```

Cilem je nahradit soucasnou enum explozi:

- `MATMUL_RELU`
- `MATMUL_ADD_BIAS`
- `MATMUL_ADD_BIAS_RELU`

jednou specializaci `MATMUL_EPILOGUE`, ktera nese typed payload. `LINEAR` s biasem
nesmi byt zvlastni specializacni family; ma byt jeden konkretni `MatmulCoreSpec`
scenario uvnitr stejneho epilogue planu.

`MSE_LOSS` zustava samostatna specializace. Tento plan se ho dotyka jen tam, kde
je potreba udrzet `RegionSpecializationCandidate` kompatibilni s payload modelem.

## Non-goals

- Neimplementovat kod v ramci tohoto dokumentu.
- Nemenit verejne `Tensor` API.
- Nedelat backend residency soucasti public `Tensor` API.
- Nerozsirovat cpu1 hot path o skryte fallbacky.
- Nevytvaret prechodne facade/adapter vrstvy pro stare enum kindy.
- Nedrzet soubezne stare `MATMUL_RELU`, `MATMUL_ADD_BIAS`, `MATMUL_ADD_BIAS_RELU`
  jako compatibility path po dokonceni migrace.
- Nezavadet obecny fused expression IR pro libovolne operace. Scope je matmul
  core plus epilogue vyraz nad vystupem matmul a externimi epilogue vstupy.
- Nezmenit MSE lowering/preparer, krome nutneho doplneni prazdneho payloadu nebo
  payload helperu v `RegionSpecializationCandidate`.
- Nezavadet Metal/CUDA vykonavani v prvni implementacni vlne. IR ma byt
  backend-neutral, ale prvni executable route je cpu1.

## Soucasny stav

Aktualni lokalni kod uz ma rozpracovane konkretni matmul specializace:

- `RegionSpecializationKind` obsahuje `MSE_LOSS`, `MATMUL_RELU`,
  `MATMUL_ADD_BIAS`, `MATMUL_ADD_BIAS_RELU`.
- `RegionSpecializationCandidate` nese jen `kind`, `orderedNodeIds`,
  `inputValueRefs`, `outputValueRef`, `anchorNodeId`, `summary`; nema typed
  payload.
- `RegionSpecializationPlanner` sbira kandidaty v poradi:
  `MseLossSpecializationDetector`, `MatmulBiasReluSpecializationDetector`,
  `MatmulBiasSpecializationDetector`, `MatmulReluSpecializationDetector`.
- `DefaultRegionSpecializationCapability` akceptuje kazdy konkretni matmul enum
  samostatnym `switch` case.
- `CpuNaturalExecutionRegionPlanner` uz umi vytvorit presne regiony pro:
  - `MATMUL -> RELU`
  - `MATMUL -> ADD(bias)`
  - `MATMUL -> ADD(bias) -> RELU`
  - `LINEAR` s biasem
  - `LINEAR -> RELU`
- `BackendPrepareDispatcher.prepareCpuSpecializedStep` preklada konkretni kindy
  na `Cpu1MatmulPostOp`.
- `Cpu1MatmulPreparer` uz umi:
  - prosty `MATMUL`
  - `MATMUL -> RELU`
  - `MATMUL -> ADD_BIAS`
  - `MATMUL -> ADD_BIAS_RELU`
  - `LINEAR` s biasem jako efektivni `ADD_BIAS`
  - `LINEAR -> RELU` jako efektivni `ADD_BIAS_RELU`
- `Cpu1PreparedMatmulUnit` uklada `Cpu1MatmulPostOp`, bias node id a bias
  broadcast strides.
- `Cpu1MatmulPostOp` je runtime enum s hodnotami `NONE`, `RELU`, `ADD_BIAS`,
  `ADD_BIAS_RELU`.

Soucasne chovani je funkcni, ale typy se budou dal mnozit pro kazdou dalsi
epilogue kombinaci. Priklady dalsich variant, ktere by nemely pridavat nove
`RegionSpecializationKind`:

- `MATMUL + ADD residual`
- `MATMUL + ADD bias + GELU`
- `MATMUL + SCALE`
- `MATMUL + BIAS + CLAMP`
- `LINEAR + bias + activation`

## Cilova architektura

### Specializacni vrstva

Cil:

```text
graph.compile.planning.region.specialization
  RegionSpecializationKind
    MSE_LOSS
    MATMUL_EPILOGUE

  RegionSpecializationCandidate
    kind
    payload
    orderedNodeIds
    inputValueRefs
    outputValueRef
    anchorNodeId
    summary

  MatmulEpiloguePlan
  MatmulCoreSpec
  MatmulCoreKind
  MatmulEpilogueExpression
  MatmulEpilogueValue
  MatmulEpilogueInputRole
  MatmulEpilogueUnaryOp
  MatmulEpilogueBinaryOp
  MatmulEpiloguePlanBuilder
  MatmulEpilogueSpecializationDetector
```

`RegionSpecializationKind` ma rikat "jaka rodina specializace", ne konkretni
post-op kombinaci. Konkretni kombinace patri do payloadu.

### Backend prepare vrstva

Cil:

```text
backend.prepare.BackendPrepareDispatcher
  case MSE_LOSS -> cpu1MseLossPreparer.prepare(...)
  case MATMUL_EPILOGUE -> cpu1MatmulPreparer.prepareEpilogue(...)
```

Dispatcher nema odvozovat semantiku z poctu nodu nebo enum kindu. Ma vyzvednout
`MatmulEpiloguePlan` z kandidata a predat jej cpu1 prepareru.

### cpu1 prepare/runtime vrstva

Prvni implementacni vlna muze uvnitr cpu1 stale prelozit obecny epilogue plan na
omezeny executable subset:

```text
MatmulEpiloguePlan
  -> Cpu1PreparedMatmulEpilogue
  -> Cpu1PreparedMatmulUnit
```

`Cpu1MatmulPostOp` muze byt docasne vnitrni cpu1 executable classification, ale
nesmi zustat graph-level specializacni vocabulary. Pokud se zachova, musi byt
privatni detail cpu1 matmul kernels, ne `RegionSpecializationKind` nahrada.

Lepsi cilovy stav pro cpu1 je:

```text
Cpu1PreparedMatmulUnit
  MatmulCoreSpec coreSpec
  Cpu1PreparedMatmulEpilogue epilogue

Cpu1PreparedMatmulEpilogue
  expression
  input bindings
  broadcast strides
  executable classification
```

## Package placement

Doporucene nove soubory:

```text
src/main/java/graph/compile/planning/region/specialization/RegionSpecializationPayload.java
src/main/java/graph/compile/planning/region/specialization/NoRegionSpecializationPayload.java
src/main/java/graph/compile/planning/region/specialization/MatmulCoreKind.java
src/main/java/graph/compile/planning/region/specialization/MatmulCoreSpec.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpiloguePlan.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpilogueExpression.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpilogueInputRole.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpilogueUnaryOp.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpilogueBinaryOp.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpiloguePlanBuilder.java
src/main/java/graph/compile/planning/region/specialization/MatmulEpilogueSpecializationDetector.java
src/main/java/backend/cpu1/prepare/Cpu1PreparedMatmulEpilogue.java
```

Soubory k uprave:

```text
src/main/java/graph/compile/planning/region/specialization/RegionSpecializationKind.java
src/main/java/graph/compile/planning/region/specialization/RegionSpecializationCandidate.java
src/main/java/graph/compile/planning/region/specialization/RegionSpecializationPlanner.java
src/main/java/graph/compile/planning/region/specialization/DefaultRegionSpecializationCapability.java
src/main/java/graph/compile/planning/partition/CpuNaturalExecutionRegionPlanner.java
src/main/java/backend/prepare/BackendPrepareDispatcher.java
src/main/java/backend/cpu1/prepare/Cpu1MatmulPreparer.java
src/main/java/backend/cpu1/prepare/Cpu1PreparedMatmulUnit.java
src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java
```

Soubory k odstraneni po migraci:

```text
src/main/java/graph/compile/planning/region/specialization/MatmulReluSpecializationDetector.java
src/main/java/graph/compile/planning/region/specialization/MatmulBiasSpecializationDetector.java
src/main/java/graph/compile/planning/region/specialization/MatmulBiasReluSpecializationDetector.java
```

Odstraneni je soucast planu, ne optional cleanup. Pokud zustanou, budou
zdroj duplicitni logiky a technickeho dluhu.

## Navrh datoveho modelu

### `RegionSpecializationKind`

```java
package graph.compile.planning.region.specialization;

/**
 * Graph-level region specialization families.
 */
public enum RegionSpecializationKind {
    MSE_LOSS,
    MATMUL_EPILOGUE
}
```

### `RegionSpecializationPayload`

```java
package graph.compile.planning.region.specialization;

/**
 * Marker for typed specialization payloads carried by graph-level candidates.
 */
public interface RegionSpecializationPayload {
}
```

### `NoRegionSpecializationPayload`

```java
package graph.compile.planning.region.specialization;

/**
 * Empty payload for specialization families that do not need typed metadata.
 */
public enum NoRegionSpecializationPayload implements RegionSpecializationPayload {
    INSTANCE
}
```

### `RegionSpecializationCandidate`

```java
package graph.compile.planning.region.specialization;

import graph.compile.planning.value.GraphValueRef;

import java.util.List;

/**
 * Backend-neutral graph-level specialization candidate.
 *
 * @param kind specialization family
 * @param payload typed specialization payload
 * @param orderedNodeIds nodes covered by the candidate in graph order
 * @param inputValueRefs values consumed from outside the candidate
 * @param outputValueRef candidate output value
 * @param anchorNodeId node that anchors the specialized unit
 * @param summary short diagnostic summary
 */
public record RegionSpecializationCandidate(
        RegionSpecializationKind kind,
        RegionSpecializationPayload payload,
        List<Integer> orderedNodeIds,
        List<GraphValueRef> inputValueRefs,
        GraphValueRef outputValueRef,
        int anchorNodeId,
        String summary
) {
    public RegionSpecializationCandidate {
        if (kind == null) {
            throw new IllegalArgumentException("kind cannot be null");
        }
        payload = payload == null ? NoRegionSpecializationPayload.INSTANCE : payload;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        inputValueRefs = List.copyOf(inputValueRefs == null ? List.of() : inputValueRefs);
        if (outputValueRef == null) {
            throw new IllegalArgumentException("outputValueRef cannot be null");
        }
        if (anchorNodeId < 0) {
            throw new IllegalArgumentException("anchorNodeId must be >= 0");
        }
        summary = summary == null ? "" : summary;
    }

    public static RegionSpecializationCandidate withoutPayload(
            RegionSpecializationKind kind,
            List<Integer> orderedNodeIds,
            List<GraphValueRef> inputValueRefs,
            GraphValueRef outputValueRef,
            int anchorNodeId,
            String summary
    ) {
        return new RegionSpecializationCandidate(
                kind,
                NoRegionSpecializationPayload.INSTANCE,
                orderedNodeIds,
                inputValueRefs,
                outputValueRef,
                anchorNodeId,
                summary
        );
    }

    public <T extends RegionSpecializationPayload> T requirePayload(Class<T> type) {
        if (!type.isInstance(payload)) {
            throw new IllegalStateException("Specialization " + kind
                    + " requires payload " + type.getSimpleName()
                    + ", got " + payload.getClass().getSimpleName());
        }
        return type.cast(payload);
    }
}
```

Poznamka: pokud lokalni styl nechce generic helper na recordu, helper muze byt
staticka metoda v `RegionSpecializationPayloads`. Dulezite je, aby dispatcher a
capability nemusely delat nebezpecne raw casty.

### `MatmulCoreKind`

```java
package graph.compile.planning.region.specialization;

/**
 * Source graph shape used to compute the matmul core.
 */
public enum MatmulCoreKind {
    MATMUL,
    LINEAR
}
```

### `MatmulCoreSpec`

```java
package graph.compile.planning.region.specialization;

import graph.compile.planning.value.GraphValueRef;

import java.util.List;

/**
 * Backend-neutral description of the matmul-like core before epilogue.
 *
 * <p>For MATMUL, left/right are the two matmul inputs. For LINEAR, left/right
 * are input/weight and optional bias is represented as a normal epilogue input,
 * not as a separate specialization kind.</p>
 */
public record MatmulCoreSpec(
        MatmulCoreKind kind,
        int coreNodeId,
        GraphValueRef left,
        GraphValueRef right,
        List<Integer> coreNodeIds
) {
    public MatmulCoreSpec {
        if (kind == null) {
            throw new IllegalArgumentException("kind cannot be null");
        }
        if (coreNodeId < 0) {
            throw new IllegalArgumentException("coreNodeId must be >= 0");
        }
        if (left == null) {
            throw new IllegalArgumentException("left cannot be null");
        }
        if (right == null) {
            throw new IllegalArgumentException("right cannot be null");
        }
        coreNodeIds = List.copyOf(coreNodeIds == null ? List.of(coreNodeId) : coreNodeIds);
        if (coreNodeIds.isEmpty() || coreNodeIds.getFirst() != coreNodeId) {
            throw new IllegalArgumentException("coreNodeIds must start with coreNodeId");
        }
    }

    public static MatmulCoreSpec matmul(int matmulNodeId, int leftNodeId, int rightNodeId) {
        return new MatmulCoreSpec(
                MatmulCoreKind.MATMUL,
                matmulNodeId,
                GraphValueRef.node(leftNodeId),
                GraphValueRef.node(rightNodeId),
                List.of(matmulNodeId)
        );
    }

    public static MatmulCoreSpec linear(int linearNodeId, int inputNodeId, int weightNodeId) {
        return new MatmulCoreSpec(
                MatmulCoreKind.LINEAR,
                linearNodeId,
                GraphValueRef.node(inputNodeId),
                GraphValueRef.node(weightNodeId),
                List.of(linearNodeId)
        );
    }
}
```

### `MatmulEpilogueInputRole`

```java
package graph.compile.planning.region.specialization;

/**
 * Role of a value referenced by the matmul epilogue expression.
 */
public enum MatmulEpilogueInputRole {
    MATMUL_OUTPUT,
    BIAS,
    RESIDUAL,
    SCALAR,
    OTHER
}
```

### `MatmulEpilogueUnaryOp`

```java
package graph.compile.planning.region.specialization;

/**
 * Unary operation supported in matmul epilogue expression trees.
 */
public enum MatmulEpilogueUnaryOp {
    RELU,
    NEG,
    ABS,
    TANH,
    SIGMOID,
    GELU,
    CLAMP_MIN,
    CLAMP_MAX
}
```

Prvni executable subset ma podporovat jen `RELU`. Ostatni hodnoty mohou byt
pripraveny jako IR vocabulary az ve chvili, kdy existuje jasny plan testu.
Pokud implementace nechce pridavat zatim nepodporovane hodnoty, zacit pouze
`RELU` je prijatelne. Dokument je uvadi jako cilovou expanzi, ne jako povinny
prvni runtime scope.

### `MatmulEpilogueBinaryOp`

```java
package graph.compile.planning.region.specialization;

/**
 * Binary operation supported in matmul epilogue expression trees.
 */
public enum MatmulEpilogueBinaryOp {
    ADD,
    SUB,
    MUL,
    MIN,
    MAX
}
```

Prvni executable subset ma podporovat jen `ADD` pro bias/residual-like vstup.

### `MatmulEpilogueExpression`

```java
package graph.compile.planning.region.specialization;

import graph.compile.planning.value.GraphValueRef;

import java.util.List;

/**
 * Small backend-neutral expression tree for a matmul epilogue.
 */
public sealed interface MatmulEpilogueExpression extends RegionSpecializationPayload
        permits MatmulEpilogueExpression.MatmulValue,
        MatmulEpilogueExpression.InputValue,
        MatmulEpilogueExpression.Unary,
        MatmulEpilogueExpression.Binary {

    List<GraphValueRef> externalInputs();

    default boolean isIdentityMatmulOutput() {
        return this instanceof MatmulValue;
    }

    record MatmulValue() implements MatmulEpilogueExpression {
        @Override
        public List<GraphValueRef> externalInputs() {
            return List.of();
        }
    }

    record InputValue(
            GraphValueRef ref,
            MatmulEpilogueInputRole role
    ) implements MatmulEpilogueExpression {
        public InputValue {
            if (ref == null) {
                throw new IllegalArgumentException("ref cannot be null");
            }
            role = role == null ? MatmulEpilogueInputRole.OTHER : role;
        }

        @Override
        public List<GraphValueRef> externalInputs() {
            return List.of(ref);
        }
    }

    record Unary(
            MatmulEpilogueUnaryOp op,
            MatmulEpilogueExpression input
    ) implements MatmulEpilogueExpression {
        public Unary {
            if (op == null) {
                throw new IllegalArgumentException("op cannot be null");
            }
            if (input == null) {
                throw new IllegalArgumentException("input cannot be null");
            }
        }

        @Override
        public List<GraphValueRef> externalInputs() {
            return input.externalInputs();
        }
    }

    record Binary(
            MatmulEpilogueBinaryOp op,
            MatmulEpilogueExpression left,
            MatmulEpilogueExpression right
    ) implements MatmulEpilogueExpression {
        public Binary {
            if (op == null) {
                throw new IllegalArgumentException("op cannot be null");
            }
            if (left == null) {
                throw new IllegalArgumentException("left cannot be null");
            }
            if (right == null) {
                throw new IllegalArgumentException("right cannot be null");
            }
        }

        @Override
        public List<GraphValueRef> externalInputs() {
            java.util.LinkedHashSet<GraphValueRef> out = new java.util.LinkedHashSet<>();
            out.addAll(left.externalInputs());
            out.addAll(right.externalInputs());
            return List.copyOf(out);
        }
    }

    static MatmulEpilogueExpression matmulOutput() {
        return new MatmulValue();
    }

    static MatmulEpilogueExpression input(GraphValueRef ref, MatmulEpilogueInputRole role) {
        return new InputValue(ref, role);
    }

    static MatmulEpilogueExpression relu(MatmulEpilogueExpression input) {
        return new Unary(MatmulEpilogueUnaryOp.RELU, input);
    }

    static MatmulEpilogueExpression add(
            MatmulEpilogueExpression left,
            MatmulEpilogueExpression right
    ) {
        return new Binary(MatmulEpilogueBinaryOp.ADD, left, right);
    }
}
```

Pokud projektovy JDK nechce `sealed`, pouzit obycejny interface a nested
records bez `permits`. Podle soucasneho kodu projekt uz pouziva moderni Java
features (`record`, `List.getFirst`, switch expressions), proto je sealed
varianta rozumna, ale neni nutna.

### `MatmulEpiloguePlan`

```java
package graph.compile.planning.region.specialization;

import graph.compile.planning.value.GraphValueRef;

import java.util.LinkedHashSet;
import java.util.List;

/**
 * Backend-neutral plan for a matmul-like core and its epilogue expression.
 */
public record MatmulEpiloguePlan(
        MatmulCoreSpec core,
        MatmulEpilogueExpression expression,
        List<Integer> epilogueNodeIds,
        List<GraphValueRef> inputValueRefs,
        GraphValueRef outputValueRef,
        String summary
) implements RegionSpecializationPayload {
    public MatmulEpiloguePlan {
        if (core == null) {
            throw new IllegalArgumentException("core cannot be null");
        }
        if (expression == null) {
            throw new IllegalArgumentException("expression cannot be null");
        }
        epilogueNodeIds = List.copyOf(epilogueNodeIds == null ? List.of() : epilogueNodeIds);
        inputValueRefs = List.copyOf(inputValueRefs == null
                ? defaultInputs(core, expression)
                : inputValueRefs);
        if (outputValueRef == null) {
            throw new IllegalArgumentException("outputValueRef cannot be null");
        }
        summary = summary == null ? "" : summary;
    }

    public List<Integer> orderedNodeIds() {
        LinkedHashSet<Integer> out = new LinkedHashSet<>();
        out.addAll(core.coreNodeIds());
        out.addAll(epilogueNodeIds);
        return List.copyOf(out);
    }

    public int anchorNodeId() {
        return outputValueRef.nodeId();
    }

    public boolean hasEpilogueNodes() {
        return !epilogueNodeIds.isEmpty();
    }

    private static List<GraphValueRef> defaultInputs(
            MatmulCoreSpec core,
            MatmulEpilogueExpression expression
    ) {
        LinkedHashSet<GraphValueRef> out = new LinkedHashSet<>();
        out.add(core.left());
        out.add(core.right());
        out.addAll(expression.externalInputs());
        return List.copyOf(out);
    }
}
```

Invariant: `inputValueRefs` musi obsahovat core vstupy jako prvni dve hodnoty.
Epilogue externi vstupy potom nasleduji v poradi prvniho vyskytu v expression
tree. Tato stabilita je dulezita pro prepared input node ids a trace.

## Generic detector/builder

Detektor ma nahradit tri stavajici matmul detector tridy. Ma byt uzky, explicitni
a presny, stejne jako dnesni detektory.

Podporovany prvni scope:

| Graph shape | Core | Expression | Output |
|---|---|---|---|
| `MATMUL -> RELU` | `MATMUL` | `relu(matmul)` | relu node |
| `MATMUL -> ADD(bias)` | `MATMUL` | `add(matmul, bias)` | add node |
| `MATMUL -> ADD(bias) -> RELU` | `MATMUL` | `relu(add(matmul, bias))` | relu node |
| `LINEAR` with bias | `LINEAR` | `add(matmul, bias)` | linear node |
| `LINEAR -> RELU` with bias | `LINEAR` | `relu(add(matmul, bias))` | relu node |

Prvni scope muze volitelne podporovat `MATMUL` bez epilogue jako identity
specializaci, ale neni nutne pro splneni migrace. Pokud se prida, musi se
jasne odlisit od bezneho jednotkoveho matmul kernelu a nesmi snizit OpenBLAS
route coverage.

### `MatmulEpiloguePlanBuilder`

```java
package graph.compile.planning.region.specialization;

import graph.CompiledNode;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;
import operations.linalg.linear;

import java.util.List;

/**
 * Builds typed MATMUL_EPILOGUE plans from exact graph patterns.
 */
final class MatmulEpiloguePlanBuilder {
    private MatmulEpiloguePlanBuilder() {
    }

    static MatmulEpiloguePlan matmulRelu(CompiledNode matmul, CompiledNode relu) {
        requireOp(matmul, Operation.OpType.MATMUL, "matmul");
        requireOp(relu, Operation.OpType.RELU, "relu");
        requireInputCount(matmul, 2, "matmul");
        if (relu.inputIds().size() != 1 || relu.inputIds().getFirst() != matmul.id()) {
            throw new UnsupportedOperationException("RELU must consume MATMUL output.");
        }
        MatmulCoreSpec core = MatmulCoreSpec.matmul(
                matmul.id(),
                matmul.inputIds().get(0),
                matmul.inputIds().get(1)
        );
        MatmulEpilogueExpression expression = MatmulEpilogueExpression.relu(
                MatmulEpilogueExpression.matmulOutput()
        );
        return new MatmulEpiloguePlan(
                core,
                expression,
                List.of(relu.id()),
                null,
                GraphValueRef.node(relu.id()),
                "matmul-epilogue:core=MATMUL,expr=relu(matmul),matmul="
                        + matmul.id() + ",relu=" + relu.id()
        );
    }

    static MatmulEpiloguePlan matmulAddBias(CompiledNode matmul, CompiledNode add) {
        requireOp(matmul, Operation.OpType.MATMUL, "matmul");
        requireOp(add, Operation.OpType.ADD, "add");
        requireInputCount(matmul, 2, "matmul");
        requireInputCount(add, 2, "add");
        int biasNodeId = biasNodeId(add, matmul.id());
        MatmulCoreSpec core = MatmulCoreSpec.matmul(
                matmul.id(),
                matmul.inputIds().get(0),
                matmul.inputIds().get(1)
        );
        MatmulEpilogueExpression expression = MatmulEpilogueExpression.add(
                MatmulEpilogueExpression.matmulOutput(),
                MatmulEpilogueExpression.input(GraphValueRef.node(biasNodeId), MatmulEpilogueInputRole.BIAS)
        );
        return new MatmulEpiloguePlan(
                core,
                expression,
                List.of(add.id()),
                null,
                GraphValueRef.node(add.id()),
                "matmul-epilogue:core=MATMUL,expr=add(matmul,bias),matmul="
                        + matmul.id() + ",add=" + add.id() + ",bias=" + biasNodeId
        );
    }

    static MatmulEpiloguePlan matmulAddBiasRelu(
            CompiledNode matmul,
            CompiledNode add,
            CompiledNode relu
    ) {
        MatmulEpiloguePlan addPlan = matmulAddBias(matmul, add);
        requireOp(relu, Operation.OpType.RELU, "relu");
        if (relu.inputIds().size() != 1 || relu.inputIds().getFirst() != add.id()) {
            throw new UnsupportedOperationException("RELU must consume ADD output.");
        }
        return new MatmulEpiloguePlan(
                addPlan.core(),
                MatmulEpilogueExpression.relu(addPlan.expression()),
                List.of(add.id(), relu.id()),
                addPlan.inputValueRefs(),
                GraphValueRef.node(relu.id()),
                "matmul-epilogue:core=MATMUL,expr=relu(add(matmul,bias)),matmul="
                        + matmul.id() + ",add=" + add.id() + ",relu=" + relu.id()
        );
    }

    static MatmulEpiloguePlan linearBias(CompiledNode linearNode) {
        requireOp(linearNode, Operation.OpType.LINEAR, "linear");
        if (!(linearNode.operation() instanceof linear linearOp) || !linearOp.hasBias()) {
            throw new UnsupportedOperationException("LINEAR epilogue requires bias metadata.");
        }
        requireInputCount(linearNode, 3, "linear");
        MatmulCoreSpec core = MatmulCoreSpec.linear(
                linearNode.id(),
                linearNode.inputIds().get(0),
                linearNode.inputIds().get(1)
        );
        int biasNodeId = linearNode.inputIds().get(2);
        MatmulEpilogueExpression expression = MatmulEpilogueExpression.add(
                MatmulEpilogueExpression.matmulOutput(),
                MatmulEpilogueExpression.input(GraphValueRef.node(biasNodeId), MatmulEpilogueInputRole.BIAS)
        );
        return new MatmulEpiloguePlan(
                core,
                expression,
                List.of(),
                null,
                GraphValueRef.node(linearNode.id()),
                "matmul-epilogue:core=LINEAR,expr=add(matmul,bias),linear="
                        + linearNode.id() + ",bias=" + biasNodeId
        );
    }

    static MatmulEpiloguePlan linearBiasRelu(CompiledNode linearNode, CompiledNode relu) {
        MatmulEpiloguePlan linearPlan = linearBias(linearNode);
        requireOp(relu, Operation.OpType.RELU, "relu");
        if (relu.inputIds().size() != 1 || relu.inputIds().getFirst() != linearNode.id()) {
            throw new UnsupportedOperationException("RELU must consume LINEAR output.");
        }
        return new MatmulEpiloguePlan(
                linearPlan.core(),
                MatmulEpilogueExpression.relu(linearPlan.expression()),
                List.of(relu.id()),
                linearPlan.inputValueRefs(),
                GraphValueRef.node(relu.id()),
                "matmul-epilogue:core=LINEAR,expr=relu(add(matmul,bias)),linear="
                        + linearNode.id() + ",relu=" + relu.id()
        );
    }

    private static int biasNodeId(CompiledNode add, int matmulNodeId) {
        int first = add.inputIds().get(0);
        int second = add.inputIds().get(1);
        if (first == matmulNodeId && second != matmulNodeId) {
            return second;
        }
        if (second == matmulNodeId && first != matmulNodeId) {
            return first;
        }
        throw new UnsupportedOperationException("ADD must consume MATMUL output exactly once.");
    }

    private static void requireOp(CompiledNode node, Operation.OpType opType, String label) {
        if (node == null || node.operation() == null || node.operation().opType() != opType) {
            throw new UnsupportedOperationException(label + " must be " + opType);
        }
    }

    private static void requireInputCount(CompiledNode node, int expected, String label) {
        if (node.inputIds().size() != expected) {
            throw new UnsupportedOperationException(label + " expects " + expected
                    + " inputs, got " + node.inputIds().size());
        }
    }
}
```

### `MatmulEpilogueSpecializationDetector`

```java
package graph.compile.planning.region.specialization;

import graph.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

import java.util.List;

/**
 * Finds exact backend-neutral MATMUL_EPILOGUE specialization candidates.
 */
final class MatmulEpilogueSpecializationDetector {
    private MatmulEpilogueSpecializationDetector() {
    }

    static List<RegionSpecializationCandidate> findCandidates(
            Partition partition,
            RegionOptimizationContext context
    ) {
        if (partition == null || context == null) {
            return List.of();
        }
        MatmulEpiloguePlan plan = switch (partition.orderedNodeIds().size()) {
            case 1 -> singleNodePlan(partition, context);
            case 2 -> twoNodePlan(partition, context);
            case 3 -> threeNodePlan(partition, context);
            default -> null;
        };
        if (plan == null || !candidateOutputIsPartitionOutput(plan, partition)) {
            return List.of();
        }
        if (publishesIntermediate(plan, partition)) {
            return List.of();
        }
        return List.of(new RegionSpecializationCandidate(
                RegionSpecializationKind.MATMUL_EPILOGUE,
                plan,
                plan.orderedNodeIds(),
                plan.inputValueRefs(),
                plan.outputValueRef(),
                plan.anchorNodeId(),
                plan.summary()
        ));
    }

    private static MatmulEpiloguePlan singleNodePlan(
            Partition partition,
            RegionOptimizationContext context
    ) {
        CompiledNode node = node(partition, context, 0);
        if (opType(node) == Operation.OpType.LINEAR) {
            return MatmulEpiloguePlanBuilder.linearBias(node);
        }
        return null;
    }

    private static MatmulEpiloguePlan twoNodePlan(
            Partition partition,
            RegionOptimizationContext context
    ) {
        CompiledNode first = node(partition, context, 0);
        CompiledNode second = node(partition, context, 1);
        Operation.OpType firstOp = opType(first);
        Operation.OpType secondOp = opType(second);
        if (firstOp == Operation.OpType.MATMUL && secondOp == Operation.OpType.RELU) {
            return MatmulEpiloguePlanBuilder.matmulRelu(first, second);
        }
        if (firstOp == Operation.OpType.MATMUL && secondOp == Operation.OpType.ADD) {
            return MatmulEpiloguePlanBuilder.matmulAddBias(first, second);
        }
        if (firstOp == Operation.OpType.LINEAR && secondOp == Operation.OpType.RELU) {
            return MatmulEpiloguePlanBuilder.linearBiasRelu(first, second);
        }
        return null;
    }

    private static MatmulEpiloguePlan threeNodePlan(
            Partition partition,
            RegionOptimizationContext context
    ) {
        CompiledNode first = node(partition, context, 0);
        CompiledNode second = node(partition, context, 1);
        CompiledNode third = node(partition, context, 2);
        if (opType(first) == Operation.OpType.MATMUL
                && opType(second) == Operation.OpType.ADD
                && opType(third) == Operation.OpType.RELU) {
            return MatmulEpiloguePlanBuilder.matmulAddBiasRelu(first, second, third);
        }
        return null;
    }

    private static boolean candidateOutputIsPartitionOutput(
            MatmulEpiloguePlan plan,
            Partition partition
    ) {
        return partition.outputValueRefs().contains(plan.outputValueRef());
    }

    private static boolean publishesIntermediate(
            MatmulEpiloguePlan plan,
            Partition partition
    ) {
        GraphValueRef output = plan.outputValueRef();
        for (int nodeId : plan.orderedNodeIds()) {
            GraphValueRef ref = GraphValueRef.node(nodeId);
            if (!ref.equals(output)
                    && (partition.outputValueRefs().contains(ref)
                    || partition.requiredMaterializedValueRefs().contains(ref))) {
                return true;
            }
        }
        return false;
    }

    private static CompiledNode node(
            Partition partition,
            RegionOptimizationContext context,
            int offset
    ) {
        return context.compiledNode(partition.orderedNodeIds().get(offset));
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? null : node.operation().opType();
    }
}
```

## Region planner a capability migrace

### `RegionSpecializationPlanner`

Planovana zmena:

```java
private static List<RegionSpecializationCandidate> specializationCandidates(
        Partition partition,
        RegionOptimizationContext context
) {
    ArrayList<RegionSpecializationCandidate> out = new ArrayList<>();
    out.addAll(MseLossSpecializationDetector.findCandidates(partition, context));
    out.addAll(MatmulEpilogueSpecializationDetector.findCandidates(partition, context));
    return List.copyOf(out);
}
```

Trace udalosti maji zustat zalozene na `candidate.kind().name()`. Pro epilogue
detaily pridat payload summary, ne nove enum kindy:

```text
specialization-candidate-found:kind=MATMUL_EPILOGUE,nodes=[2,3,4],output=node:4,summary=...
specialized-primitive:MATMUL_EPILOGUE
specialized-primitive-summary:matmul-epilogue:core=MATMUL,expr=relu(add(matmul,bias)),...
```

### `DefaultRegionSpecializationCapability`

```java
package graph.compile.planning.region.specialization;

import graph.compile.planning.partition.PartitionTarget;

/**
 * Default backend specialization acceptance hook used by region optimization.
 */
public final class DefaultRegionSpecializationCapability implements RegionSpecializationCapability {
    @Override
    public RegionSpecializationDecision evaluate(
            PartitionTarget target,
            RegionSpecializationCandidate candidate
    ) {
        if (candidate == null) {
            return RegionSpecializationDecision.reject("candidate-null");
        }
        if (target != PartitionTarget.CPU) {
            return RegionSpecializationDecision.reject("backend-specialization-unsupported:" + safeTarget(target));
        }
        return switch (candidate.kind()) {
            case MSE_LOSS -> RegionSpecializationDecision.accept("cpu1-mse-loss-executable");
            case MATMUL_EPILOGUE -> evaluateMatmulEpilogue(candidate);
        };
    }

    private static RegionSpecializationDecision evaluateMatmulEpilogue(
            RegionSpecializationCandidate candidate
    ) {
        MatmulEpiloguePlan plan = candidate.requirePayload(MatmulEpiloguePlan.class);
        if (!Cpu1MatmulEpilogueSupport.isSupported(plan)) {
            return RegionSpecializationDecision.reject("cpu1-matmul-epilogue-unsupported:"
                    + Cpu1MatmulEpilogueSupport.rejectionReason(plan));
        }
        return RegionSpecializationDecision.accept("cpu1-matmul-epilogue-executable");
    }

    private static String safeTarget(PartitionTarget target) {
        return target == null ? "NONE" : target.name();
    }
}
```

`Cpu1MatmulEpilogueSupport` by nemel byt v graph package, protoze by to tahalo
cpu1 dependency do planneru. Prakticka varianta je dat maly backend-neutral
subset checker do specialization package:

```text
MatmulEpilogueExecutableSubset
```

a pojmenovat rejection reason neutralne:

```text
matmul-epilogue-expression-not-in-initial-executable-subset
```

Prvni implementace muze akceptovat vse, co detektor sam umi vytvorit. Jakmile se
expression vocabulary rozsiri mimo `ADD`/`RELU`, capability musi explicitne
kontrolovat supported subset.

## CpuNaturalExecutionRegionPlanner

`CpuNaturalExecutionRegionPlanner` dnes vybira presne epilogue regiony pomoci
`exactMatmulEpilogueNodeIds`. Tato cast muze zustat vecne stejna, ale dulezite
je prejmenovat komentar/reasony a nezavazet ji k enum kindum.

Doporucene upravy:

- `exactMatmulEpilogueNodeIds` muze zustat, protoze vraci jen node ids.
- `cpu-natural-exact-matmul-epilogue` reason uz odpovida cilovemu modelu.
- `exactLinearEpilogueNodeIds` ma zustat, ale komentar musi rikat, ze `LINEAR`
  je syntakticky core scenario pro `MATMUL_EPILOGUE`.
- Pokud se casem prida generic residual epilogue, rozsireni patri sem i do
  `MatmulEpilogueSpecializationDetector`, ale ne do `RegionSpecializationKind`.

## BackendPrepareDispatcher migrace

Stavajici switch:

```java
return switch (candidate.kind()) {
    case MSE_LOSS -> cpu1MseLossPreparer.prepare(outputNode, loweredUnit, context);
    case MATMUL_RELU -> prepareCpu1MatmulRelu(outputNode, candidate, context);
    case MATMUL_ADD_BIAS -> prepareCpu1MatmulBiasEpilogue(...);
    case MATMUL_ADD_BIAS_RELU -> prepareCpu1MatmulBiasEpilogue(...);
};
```

Cilovy switch:

```java
public CompiledNodeExecutionMetadata prepareCpuSpecializedStep(
        CompiledNode outputNode,
        LoweredExecutionUnit loweredUnit,
        BackendPrepareContext context
) {
    Objects.requireNonNull(outputNode, "outputNode cannot be null");
    Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
    Objects.requireNonNull(context, "context cannot be null");
    RegionSpecializationCandidate candidate = requireSpecializationCandidate(loweredUnit);
    return switch (candidate.kind()) {
        case MSE_LOSS -> cpu1MseLossPreparer.prepare(outputNode, loweredUnit, context);
        case MATMUL_EPILOGUE -> prepareCpu1MatmulEpilogue(outputNode, candidate, context);
    };
}

private CompiledNodeExecutionMetadata prepareCpu1MatmulEpilogue(
        CompiledNode outputNode,
        RegionSpecializationCandidate candidate,
        BackendPrepareContext context
) {
    MatmulEpiloguePlan plan = candidate.requirePayload(MatmulEpiloguePlan.class);
    validateMatmulEpilogueCandidate(outputNode, candidate, plan, context);
    List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
            .map(GraphValueRef::nodeId)
            .toList();
    Cpu1PreparedArtifact artifact = cpu1MatmulPreparer.prepareEpilogue(
            plan,
            outputNode,
            context.descriptorIndex(),
            Cpu1PrepareConfig.automatic(runtimeConfig, Runtime.getRuntime().availableProcessors())
    );
    return new CompiledNodeExecutionMetadata(
            ComputeBackend.CPU,
            null,
            inputNodeIds,
            artifact,
            InputResidencyRequirement.cpuReadableAll(),
            OutputResidencyEffect.cpuCurrentPreserveNative()
    );
}

private static void validateMatmulEpilogueCandidate(
        CompiledNode outputNode,
        RegionSpecializationCandidate candidate,
        MatmulEpiloguePlan plan,
        BackendPrepareContext context
) {
    if (candidate.kind() != RegionSpecializationKind.MATMUL_EPILOGUE) {
        throw new UnsupportedOperationException("cpu1 MATMUL_EPILOGUE preparer does not support "
                + candidate.kind());
    }
    if (candidate.outputValueRef().nodeId() != outputNode.id()) {
        throw new IllegalStateException("MATMUL_EPILOGUE specialization output node mismatch. candidate="
                + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
    }
    if (!candidate.orderedNodeIds().equals(plan.orderedNodeIds())) {
        throw new UnsupportedOperationException("MATMUL_EPILOGUE candidate nodes do not match plan nodes.");
    }
    if (!candidate.inputValueRefs().equals(plan.inputValueRefs())) {
        throw new UnsupportedOperationException("MATMUL_EPILOGUE candidate inputs do not match plan inputs.");
    }
    for (int nodeId : candidate.orderedNodeIds()) {
        if (context.compiledNode(nodeId) == null) {
            throw new UnsupportedOperationException("MATMUL_EPILOGUE references unknown node " + nodeId);
        }
    }
}
```

Po teto zmene odstranit `prepareCpu1MatmulRelu`,
`prepareCpu1MatmulBiasEpilogue`, `validateMatmulReluCandidate`,
`validateMatmulBiasEpilogueCandidate`, `validateLinearBiasEpilogueCandidate`.
Nenechavat je jako compatibility helpers.

## cpu1 prepare/runtime design

### `Cpu1PreparedMatmulEpilogue`

```java
package backend.cpu1.prepare;

import graph.compile.planning.region.specialization.MatmulEpilogueExpression;
import graph.compile.planning.value.GraphValueRef;

import java.util.List;

/**
 * Prepared cpu1 binding for a matmul epilogue expression.
 */
public final class Cpu1PreparedMatmulEpilogue {
    private final MatmulEpilogueExpression expression;
    private final Cpu1MatmulPostOp executablePostOp;
    private final List<GraphValueRef> externalInputs;
    private final int biasNodeId;
    private final int biasRowStride;
    private final int biasColStride;
    private final int[] biasBatchOffsets;

    public Cpu1PreparedMatmulEpilogue(
            MatmulEpilogueExpression expression,
            Cpu1MatmulPostOp executablePostOp,
            List<GraphValueRef> externalInputs,
            int biasNodeId,
            int biasRowStride,
            int biasColStride,
            int[] biasBatchOffsets
    ) {
        if (expression == null) {
            throw new IllegalArgumentException("expression cannot be null");
        }
        if (executablePostOp == null) {
            throw new IllegalArgumentException("executablePostOp cannot be null");
        }
        this.expression = expression;
        this.executablePostOp = executablePostOp;
        this.externalInputs = List.copyOf(externalInputs == null ? List.of() : externalInputs);
        if (executablePostOp.requiresBias() && biasNodeId < 0) {
            throw new IllegalArgumentException("bias post-op requires biasNodeId");
        }
        if (!executablePostOp.requiresBias() && biasNodeId >= 0) {
            throw new IllegalArgumentException("non-bias post-op does not accept biasNodeId");
        }
        this.biasNodeId = biasNodeId;
        this.biasRowStride = biasRowStride;
        this.biasColStride = biasColStride;
        this.biasBatchOffsets = biasBatchOffsets == null ? new int[0] : biasBatchOffsets.clone();
    }

    public static Cpu1PreparedMatmulEpilogue none(MatmulEpilogueExpression expression, int batchCount) {
        return new Cpu1PreparedMatmulEpilogue(
                expression,
                Cpu1MatmulPostOp.NONE,
                List.of(),
                -1,
                0,
                0,
                new int[batchCount]
        );
    }

    public MatmulEpilogueExpression expression() {
        return expression;
    }

    public Cpu1MatmulPostOp executablePostOp() {
        return executablePostOp;
    }

    public List<GraphValueRef> externalInputs() {
        return externalInputs;
    }

    public boolean hasBias() {
        return executablePostOp.requiresBias();
    }

    public int biasNodeId() {
        if (!hasBias()) {
            throw new IllegalStateException("This matmul epilogue does not have a bias input.");
        }
        return biasNodeId;
    }

    public int biasRowStride() {
        return biasRowStride;
    }

    public int biasColStride() {
        return biasColStride;
    }

    public int biasBatchOffset(int batch) {
        return biasBatchOffsets[batch];
    }
}
```

### `Cpu1MatmulEpilogueClassifier`

Prvni vlna muze classify obecny expression tree na existujici executable enum.
Tento helper patri do cpu1 prepare, ne do graph specializace.

```java
package backend.cpu1.prepare;

import graph.compile.planning.region.specialization.MatmulEpilogueBinaryOp;
import graph.compile.planning.region.specialization.MatmulEpilogueExpression;
import graph.compile.planning.region.specialization.MatmulEpilogueInputRole;
import graph.compile.planning.region.specialization.MatmulEpilogueUnaryOp;
import graph.compile.planning.value.GraphValueRef;

final class Cpu1MatmulEpilogueClassifier {
    private Cpu1MatmulEpilogueClassifier() {
    }

    static Cpu1MatmulPostOp classify(MatmulEpilogueExpression expression) {
        if (expression instanceof MatmulEpilogueExpression.MatmulValue) {
            return Cpu1MatmulPostOp.NONE;
        }
        if (isReluOfMatmul(expression)) {
            return Cpu1MatmulPostOp.RELU;
        }
        if (biasInput(expression) != null) {
            return Cpu1MatmulPostOp.ADD_BIAS;
        }
        if (expression instanceof MatmulEpilogueExpression.Unary unary
                && unary.op() == MatmulEpilogueUnaryOp.RELU
                && biasInput(unary.input()) != null) {
            return Cpu1MatmulPostOp.ADD_BIAS_RELU;
        }
        throw new UnsupportedOperationException("cpu1 does not support MATMUL_EPILOGUE expression " + expression);
    }

    static GraphValueRef biasInput(MatmulEpilogueExpression expression) {
        if (!(expression instanceof MatmulEpilogueExpression.Binary binary)
                || binary.op() != MatmulEpilogueBinaryOp.ADD) {
            return null;
        }
        if (binary.left() instanceof MatmulEpilogueExpression.MatmulValue
                && binary.right() instanceof MatmulEpilogueExpression.InputValue input
                && input.role() == MatmulEpilogueInputRole.BIAS) {
            return input.ref();
        }
        if (binary.right() instanceof MatmulEpilogueExpression.MatmulValue
                && binary.left() instanceof MatmulEpilogueExpression.InputValue input
                && input.role() == MatmulEpilogueInputRole.BIAS) {
            return input.ref();
        }
        return null;
    }

    private static boolean isReluOfMatmul(MatmulEpilogueExpression expression) {
        return expression instanceof MatmulEpilogueExpression.Unary unary
                && unary.op() == MatmulEpilogueUnaryOp.RELU
                && unary.input() instanceof MatmulEpilogueExpression.MatmulValue;
    }
}
```

Pozor na poradi v `classify`: `ADD_BIAS_RELU` musi byt testovano pred prostym
`ADD_BIAS`, jinak by `relu(add(...))` nespadlo do bias branch. Spravna verze:

```java
static Cpu1MatmulPostOp classify(MatmulEpilogueExpression expression) {
    if (expression instanceof MatmulEpilogueExpression.MatmulValue) {
        return Cpu1MatmulPostOp.NONE;
    }
    if (isReluOfMatmul(expression)) {
        return Cpu1MatmulPostOp.RELU;
    }
    if (expression instanceof MatmulEpilogueExpression.Unary unary
            && unary.op() == MatmulEpilogueUnaryOp.RELU
            && biasInput(unary.input()) != null) {
        return Cpu1MatmulPostOp.ADD_BIAS_RELU;
    }
    if (biasInput(expression) != null) {
        return Cpu1MatmulPostOp.ADD_BIAS;
    }
    throw new UnsupportedOperationException("cpu1 does not support MATMUL_EPILOGUE expression " + expression);
}
```

### `Cpu1MatmulPreparer.prepareEpilogue`

```java
package backend.cpu1.prepare;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.region.specialization.MatmulCoreKind;
import graph.compile.planning.region.specialization.MatmulEpiloguePlan;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;

public final class Cpu1MatmulPreparer {
    public Cpu1PreparedArtifact prepareEpilogue(
            MatmulEpiloguePlan plan,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (outputNode == null) {
            throw new IllegalArgumentException("outputNode cannot be null");
        }
        CompiledNode coreNode = descriptorIndexNode(plan.core().coreNodeId(), outputNode, descriptorIndex);
        Cpu1MatmulPostOp postOp = Cpu1MatmulEpilogueClassifier.classify(plan.expression());
        int biasNodeId = biasNodeId(plan, postOp);
        CompiledNode addLikeNode = addLikeContractNode(plan, outputNode, coreNode);
        return prepare(
                coreNode,
                outputNode,
                descriptorIndex,
                config,
                postOp,
                biasNodeId,
                addLikeNode
        );
    }

    private static int biasNodeId(MatmulEpiloguePlan plan, Cpu1MatmulPostOp postOp) {
        if (!postOp.requiresBias()) {
            return -1;
        }
        GraphValueRef bias = Cpu1MatmulEpilogueClassifier.biasInput(
                postOp == Cpu1MatmulPostOp.ADD_BIAS_RELU
                        ? ((graph.compile.planning.region.specialization.MatmulEpilogueExpression.Unary)
                        plan.expression()).input()
                        : plan.expression()
        );
        if (bias == null || bias.nodeId() < 0) {
            throw new UnsupportedOperationException("cpu1 MATMUL_EPILOGUE requires graph node bias input.");
        }
        return bias.nodeId();
    }

    private static CompiledNode addLikeContractNode(
            MatmulEpiloguePlan plan,
            CompiledNode outputNode,
            CompiledNode coreNode
    ) {
        if (plan.core().kind() == MatmulCoreKind.LINEAR) {
            return coreNode;
        }
        if (plan.epilogueNodeIds().isEmpty()) {
            return outputNode;
        }
        // In the initial subset the first epilogue node is ADD for bias plans.
        // For MATMUL -> RELU this value is ignored because postOp.requiresBias() is false.
        return outputNode;
    }

    private static CompiledNode descriptorIndexNode(
            int nodeId,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        // Sketch only: actual implementation should use BackendPrepareContext
        // or pass the compiled-node lookup into prepareEpilogue. Do not infer
        // CompiledNode from descriptor data.
        throw new UnsupportedOperationException("prepareEpilogue needs a compiled-node lookup");
    }
}
```

Dulezita korekce pro realnou implementaci: `CompiledTensorDescriptorIndex` neumi
vratit `CompiledNode`. Proto `prepareEpilogue` ma prijmout bud `CompiledNode coreNode`,
nebo `BackendPrepareDispatcher` ma predat `context.compiledNode(plan.core().coreNodeId())`.
Doporucena realna signatura:

```java
public Cpu1PreparedArtifact prepareEpilogue(
        MatmulEpiloguePlan plan,
        CompiledNode coreNode,
        CompiledNode outputNode,
        CompiledTensorDescriptorIndex descriptorIndex,
        Cpu1PrepareConfig config
)
```

Tato signatura je cistejsi a nevyzaduje zadny novy lookup abstraction.

### Realna `prepareEpilogue` signatura

```java
public Cpu1PreparedArtifact prepareEpilogue(
        MatmulEpiloguePlan plan,
        CompiledNode coreNode,
        CompiledNode outputNode,
        CompiledTensorDescriptorIndex descriptorIndex,
        Cpu1PrepareConfig config
) {
    if (plan == null) {
        throw new IllegalArgumentException("plan cannot be null");
    }
    if (coreNode == null) {
        throw new IllegalArgumentException("coreNode cannot be null");
    }
    if (coreNode.id() != plan.core().coreNodeId()) {
        throw new UnsupportedOperationException("MATMUL_EPILOGUE core node mismatch. plan="
                + plan.core().coreNodeId() + ", node=" + coreNode.id());
    }
    Cpu1MatmulPostOp postOp = Cpu1MatmulEpilogueClassifier.classify(plan.expression());
    int biasNodeId = biasNodeId(plan, postOp);
    return prepare(
            coreNode,
            outputNode,
            descriptorIndex,
            config,
            postOp,
            biasNodeId,
            coreNode.operation().opType() == Operation.OpType.LINEAR ? coreNode : outputNode
    );
}
```

V existujicim `prepare(...)` helperu je treba zachovat validace:

- dtype core vstupu a outputu musi sedet
- dense contiguous input/output contract zustava
- bias descriptor musi byt dense contiguous bez offsetu
- `LINEAR` s biasem ma efektivni bias z tretiho inputu
- route `OPENBLAS_ARRAY_COPYING` nesmi potichu akceptovat bias/relu, pokud to
  neumime spustit

## Trace a metadata

Trace nesmi zustat jen `cpu1MatmulPostOp=ADD_BIAS_RELU`. Pridat:

```text
cpu1MatmulPostOp=ADD_BIAS_RELU
cpu1MatmulEpilogueKind=MATMUL_EPILOGUE
cpu1MatmulEpilogueCore=MATMUL|LINEAR
cpu1MatmulEpilogueExpression=relu(add(matmul,bias))
cpu1MatmulEpilogueBiasNode=<id>
```

`cpu1MatmulPostOp` muze zustat kvuli kompatibilite benchmark/report assertions
uvnitr cpu1. Graph/region trace ma pouzivat `MATMUL_EPILOGUE`.

## Migracni kroky

1. Pridat payload infra:
   - `RegionSpecializationPayload`
   - `NoRegionSpecializationPayload`
   - novy `payload` field do `RegionSpecializationCandidate`
   - `withoutPayload(...)` helper pro `MSE_LOSS`
   - `requirePayload(...)`

2. Zmenit `MseLossSpecializationDetector`:
   - pouzit `RegionSpecializationCandidate.withoutPayload(...)`
   - nezmenit MSE summary, nodes ani input/output refs

3. Pridat matmul epilogue IR:
   - `MatmulCoreKind`
   - `MatmulCoreSpec`
   - `MatmulEpilogueInputRole`
   - `MatmulEpilogueUnaryOp`
   - `MatmulEpilogueBinaryOp`
   - `MatmulEpilogueExpression`
   - `MatmulEpiloguePlan`
   - `MatmulEpiloguePlanBuilder`

4. Pridat `MatmulEpilogueSpecializationDetector`:
   - pokryt presne soucasne tvary
   - zachovat intermediate publication/materialization rejection
   - zachovat output must be partition output invariant
   - zahrnout `LINEAR` jako core kind, ne jako zvlastni specializaci

5. Upravit `RegionSpecializationKind`:
   - pridat `MATMUL_EPILOGUE`
   - odstranit `MATMUL_RELU`, `MATMUL_ADD_BIAS`, `MATMUL_ADD_BIAS_RELU`
     ve stejnem commitu, kde jsou prepsane switch cases

6. Upravit `RegionSpecializationPlanner`:
   - nahradit tri stare detektory jednim generic detektorem
   - trace ponechat generic podle kind + payload summary

7. Upravit `DefaultRegionSpecializationCapability`:
   - `MSE_LOSS` zustava
   - `MATMUL_EPILOGUE` akceptovat jen pro podporovany executable subset
   - rejection reason musi byt stabilni a citelny

8. Upravit `BackendPrepareDispatcher`:
   - nahradit tri matmul cases jednim `MATMUL_EPILOGUE`
   - vyzvednout `MatmulEpiloguePlan`
   - predat plan a core node do `Cpu1MatmulPreparer.prepareEpilogue`
   - odstranit stare validate/prepare helpers pro konkretni kindy

9. Upravit `Cpu1MatmulPreparer`:
   - pridat `prepareEpilogue(...)`
   - pridat classifier expression -> executable subset
   - zachovat stavajici `prepare`, `prepareLinearEpilogue`,
     `prepareMatmulBiasEpilogue` jen pokud jsou porad pouzite primymi testy nebo
     public internal API; pokud nejsou potreba, odstranit je
   - neprekladat graph-level kindy na `Cpu1MatmulPostOp`

10. Upravit `Cpu1PreparedMatmulUnit`:
    - idealne pridat `Cpu1PreparedMatmulEpilogue epilogue`
    - zachovat accessors `postOp()`, `hasBias()`, `biasNodeId()` jen pokud je
      pouzivaji existujici testy/trace; mohou delegovat do epilogue
    - pokud accessors zustanou, dokumentovat je jako cpu1 executable metadata,
      ne graph specialization API

11. Upravit trace:
    - graph/region assertions zmenit na `MATMUL_EPILOGUE`
    - cpu1 trace rozsirit o core/expression metadata
    - zachovat `cpu1MatmulPostOp` pro prepared route evidence

12. Odstranit obsolete tridy:
    - `MatmulReluSpecializationDetector`
    - `MatmulBiasSpecializationDetector`
    - `MatmulBiasReluSpecializationDetector`

13. Aktualizovat testy:
    - planning/region testy maji overovat `MATMUL_EPILOGUE` a payload obsah
    - execution testy maji overovat vystup + prepared epilogue metadata
    - negative testy maji overit intermediate materialization/publication rejection

14. Spustit cilene overeni a opravit regresi bez rozsireni scope.

## Test plan

### Region/planning testy

Upravit/pridat v:

```text
src/test/java/graph/compile/planning/region/DefaultRegionOptimizerTest.java
src/test/java/graph/compile/planning/partition/CpuNaturalExecutionRegionPlannerTest.java
```

Priklady assertions:

```java
assertEquals(RegionSpecializationKind.MATMUL_EPILOGUE, unit.specialization().kind());
MatmulEpiloguePlan plan = unit.specialization().requirePayload(MatmulEpiloguePlan.class);
assertEquals(MatmulCoreKind.MATMUL, plan.core().kind());
assertEquals(List.of(matmulNodeId, addNodeId, reluNodeId), plan.orderedNodeIds());
assertEquals(List.of(GraphValueRef.node(leftId), GraphValueRef.node(rightId), GraphValueRef.node(biasId)),
        plan.inputValueRefs());
assertTrue(plan.expression() instanceof MatmulEpilogueExpression.Unary);
```

Scenare:

- `MATMUL -> RELU` emits `MATMUL_EPILOGUE`, expression `relu(matmul)`.
- `MATMUL -> ADD(bias)` emits `MATMUL_EPILOGUE`, expression
  `add(matmul,bias)`.
- `MATMUL -> ADD(bias) -> RELU` emits `MATMUL_EPILOGUE`, expression
  `relu(add(matmul,bias))`.
- `LINEAR` with bias emits `MATMUL_EPILOGUE`, core `LINEAR`, expression
  `add(matmul,bias)`.
- `LINEAR -> RELU` emits `MATMUL_EPILOGUE`, core `LINEAR`, expression
  `relu(add(matmul,bias))`.
- Required materialized intermediate rejects specialization.
- Partition output containing intermediate rejects specialization.
- MSE tests stale pass with `MSE_LOSS`.

### Prepare/execute testy

Upravit/pridat v:

```text
src/test/java/backend/cpu1/Cpu1MatmulExecutionContractTest.java
src/test/java/backend/cpu1/Cpu1LinearExecutionContractTest.java
src/test/java/backend/cpu1/BackendPrepareDispatcherCpu1FusedRouteTest.java
```

Scenare:

- Prepared graph `matmul.relu()` ma `Cpu1PreparedArtifact` s matmul executable
  a specialization kind `MATMUL_EPILOGUE`.
- Prepared graph `matmul.add(bias)` zachova vystup a bias strides.
- Prepared graph `matmul.add(bias).relu()` zachova vystup a trace
  `cpu1MatmulEpilogueExpression=relu(add(matmul,bias))`.
- Prepared graph `linear(weight,bias)` ma core `LINEAR` a cpu1 executable
  post-op `ADD_BIAS`.
- Prepared graph `linear(weight,bias).relu()` ma core `LINEAR` a cpu1
  executable post-op `ADD_BIAS_RELU`.
- Unsupported expression v payloadu je prepare-time rejected s jasnou hlaskou,
  bez fallbacku.

### Source hygiene testy

Pridat/aktualizovat test nebo aspon `rg` kontrolu v review:

```bash
rg "MATMUL_RELU|MATMUL_ADD_BIAS|MATMUL_ADD_BIAS_RELU" src/main/java src/test/java
```

Po migraci nesmi zbyt zadny source odkaz na stare graph-level kindy. Pokud
zustane text v historickem todo dokumentu, nevadi; kontrola ma byt scopeovana
na `src`.

## Rizika a mitigace

### Riziko: payload zacne byt netypovana taska

Mitigace:

- povolit jen `RegionSpecializationPayload`
- pridat `candidate.requirePayload(MatmulEpiloguePlan.class)`
- nepouzivat `Object`
- nedavat do payloadu backend runtime objekty

### Riziko: graph package zacne zaviset na cpu1

Mitigace:

- graph specializace obsahuje jen backend-neutral IR
- cpu1 support/classifier zustava v `backend.cpu1.prepare`
- capability bud akceptuje detektorem omezeny subset, nebo pouziva neutralni
  subset checker bez cpu1 imports

### Riziko: `LINEAR` semantika bude dvakrat modelovana

Mitigace:

- `LINEAR` je jen `MatmulCoreKind.LINEAR`
- bias je normalni epilogue input
- neni zadny `LINEAR_EPILOGUE` kind
- testy musi overit `plan.core().kind() == LINEAR`

### Riziko: `Cpu1MatmulPostOp` zustane druha enum exploze

Mitigace:

- v prvni vlne muze zustat jen jako executable subset classifier
- graph/region/lowering nesmi na `Cpu1MatmulPostOp` zaviset
- dalsi epilogue kombinace maji rozsirovat expression evaluator/codegen,
  ne `RegionSpecializationKind`

### Riziko: detektor bude prilis obecny a prijme nebezpecne grafy

Mitigace:

- prvni detektor zustane exact-shape only
- stale vyzaduje sole internal chain implicitne pres partition order a input ids
- stale odmita intermediate outputs/materialization
- testy pro multi-output/intermediate required materialized

### Riziko: candidate input order regresuje prepare

Mitigace:

- `MatmulEpiloguePlan.defaultInputs` drzi left/right jako prvni dve hodnoty
- epilogue vstupy jsou stable insertion order z expression
- testy assertuji presny `inputValueRefs`

### Riziko: trace/reporting ztrati konkretni epilogue signal

Mitigace:

- region trace: `kind=MATMUL_EPILOGUE` + `summary=...expr=...`
- cpu1 trace: `cpu1MatmulEpilogueCore`, `cpu1MatmulEpilogueExpression`
- ponechat `cpu1MatmulPostOp` jako executable detail

## Verifikacni prikazy

Pro doc-only zmenu tohoto planu:

```bash
git diff --check
```

Pro budouci implementaci:

```bash
./gradlew classes
./gradlew test --tests graph.compile.planning.region.DefaultRegionOptimizerTest
./gradlew test --tests graph.compile.planning.partition.CpuNaturalExecutionRegionPlannerTest
./gradlew test --tests backend.cpu1.Cpu1MatmulExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1LinearExecutionContractTest
./gradlew test --tests backend.cpu1.BackendPrepareDispatcherCpu1FusedRouteTest
```

Volitelne po stabilizaci:

```bash
./gradlew test --tests backend.cpu1.Cpu1CpuParityInventoryTest
rg "MATMUL_RELU|MATMUL_ADD_BIAS|MATMUL_ADD_BIAS_RELU" src/main/java src/test/java
```

`./gradlew test` nespoustet automaticky jako prvni overeni teto migrace, protoze
default suite muze zahrnovat debug benchmark testy. Pouzit cilene filtry.
