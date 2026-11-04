# [Compile](../../glossary.md#compile)-configuration inputs

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Compile configuration separates which [graph](../../glossary.md#graph) is requested, which [backend](../../glossary.md#backend) [owners](../../glossary.md#term-owner) are acceptable,
and whether optional exact rewrites are allowed. A **[compile mode](../../glossary.md#compile-mode--compilemode)** describes graph [scope](../../glossary.md#term-scope), such
as forward results plus [derivatives](../../glossary.md#term-derivative); it is not itself a [training session](../../glossary.md#training-session--trainingsession). An [optimizer](../../glossary.md#term-optimizer) still
needs a separate training [owner](../../glossary.md#term-owner) to apply those derivatives to [parameters](../../glossary.md#term-parameter).

## Requests are data, not backend actions

An integration caller constructs these values and passes them to `AdvancedEngine.compile`; the
ordinary Engine supplies its own fixed values. Engine forwards them to Compiler, which calls
Planning for each surviving operation occurrence. Constructing a CUDA identity below therefore
does not open CUDA, register a provider or prove an executable CUDA route exists. Immutable
configuration may be retained safely without owning native resources. Compare
[explicit Engine composition](../engine/composition.md#current-explicit-cpu-and-metal-composition)
for the separate call that transfers opened integrations, and [eligibility and partitions](planning.md)
for how hard constraints differ from soft preferences.

<a id="current-compile-configuration-inputs"></a>
## Current [compile](../../glossary.md#compile)-configuration inputs

`io.github.pho001.synaptik.config.compile.BackendIntent` is current declarative compile
configuration. It holds exactly one `Optional<BackendRequirement>` named `hardRequirement`:

```java
// A CUDA requirement is data for Planning, not registration or a device-open call.
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda =
        BackendIntent.requiring(
                new BackendIdRequirement(new BackendId("cuda")));
```

The first value has no hard eligibility [target](../../glossary.md#term-target). The second retains the exact supplied requirement
reference for [Planning](../../glossary.md#term-planning). Neither construction evaluates availability or capability, ranks
candidates, chooses [ownership](../../glossary.md#term-ownership-meanings), discovers a service, or invokes a [compiler](../../glossary.md#term-compiler). In particular,
`unconstrained()` does not mean “automatic [backend](../../glossary.md#backend) selection succeeded”; it records only the
absence of a hard target.

The [canonical](../../glossary.md#term-canonical-form) constructor rejects a null optional with `NullPointerException("hardRequirement")`.
The `requiring` factory rejects a null requirement with `NullPointerException("requirement")`.
Current internal planning evaluates the hard target before its cost-free baseline [owner](../../glossary.md#term-owner) selection,
and the complete compiler pipeline supplies this intent once per final [graph](../../glossary.md#graph) [node](../../glossary.md#node). General score
calculation, planning cost profiles, `CompileConfig`, and [Engine](../../glossary.md#term-engine)-owned failure translation remain
planned. The separate current [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig) value
described below supplies only soft preference input; it does not change these hard-intent
semantics.

`io.github.pho001.synaptik.config.compile.CompileMode` is also current. It contains exactly three
requested graph scopes in declaration order:

```java
// TRAINING_STEP selects graph scope; it does not apply an optimizer update.
import io.github.pho001.synaptik.config.compile.CompileMode;

CompileMode inferenceScope = CompileMode.FORWARD_ONLY;
CompileMode gradientScope = CompileMode.FORWARD_AND_BACKWARD;
CompileMode trainingStepDirection = CompileMode.TRAINING_STEP;
```

`FORWARD_ONLY` requests [forward graph](../../glossary.md#forward-graph) construction and requested forward publications.
Package-private `GraphCompiler` currently interprets it as a forward-only graph-stage or complete
[artifact](../../glossary.md#term-artifact) request and requires the functional request to be absent. `FORWARD_AND_BACKWARD`
requests current package-private one/two-stage pre-[capture](../../glossary.md#term-capture-meanings) [autograd](../../glossary.md#autograd) plus combined
forward/[backward graph](../../glossary.md#backward-graph)-stage work.
`TRAINING_STEP` currently performs the same internal graph and artifact construction, but the enum value
does not itself introduce an [optimizer](../../glossary.md#term-optimizer), optimizer-update graph, session, [publication](../../glossary.md#term-publication) delivery,
schedule, or execution behavior. `GraphCompilationPort` and `AdvancedEngine.compile(...)` accept
all three values subject to the existing mode/request [validation](../../glossary.md#term-validation). The advanced Engine adds no
training-step execution, backward convenience, or guarantee that the resulting artifact can pass
its [CPU](../../glossary.md#term-cpu)-only advanced [prepare](../../glossary.md#prepare) restriction.

`io.github.pho001.synaptik.config.compile.GraphOptimizationConfig` is another current value:

```java
// Disabling optional rewrites still preserves mandatory validation and canonicalization.
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;

GraphOptimizationConfig noOptionalOptimization =
        GraphOptimizationConfig.disabled();
GraphOptimizationConfig standardOptimization =
        GraphOptimizationConfig.standard();
```

The first value requests that the compiler skip optional semantics-preserving optimization.
It cannot disable [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), ordering, [inference](../../glossary.md#term-inference), validation, mandatory canonical
representation, mode-required autograd, publication [binding](../../glossary.md#term-binding), planning, preparation, or execution.
The second permits the compiler's standard optional semantics-preserving pipeline without freezing
a pass list, pass order, internal graph [shape](../../glossary.md#shape), or implementation strategy. It permits no
approximate mathematics, changed numerical semantics, backend-specific [fusion](../../glossary.md#term-fusion), preparation, or
execution behavior. Direct construction retains either primitive [boolean](../../glossary.md#data-type--datatype) value, and both factories
return fresh values. The package-private compiler transformation boundary consumes this
permission through the public [integration port](../../glossary.md#term-integration-port); no `CompileConfig` aggregate or ordinary-user
compile [facade](../../glossary.md#term-facade) does.

`io.github.pho001.synaptik.config.compile.PartitionScoringConfig` is the fourth current value. It
holds exactly one `Optional<DeviceClass>` named `preferredDeviceClass`:

```java
// A soft class preference cannot override hard eligibility or promise accelerator ownership.
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

The first value contains an empty optional and means only that no explicit coarse [device](../../glossary.md#term-device)-class
preference was supplied. It selects no aggregate default or fallback and promises neither equal
candidate scores nor successful ownership selection. The second contains the exact supplied enum
reference as a soft preference that planning may apply only after hard eligibility. It does
not remove another eligible candidate, weaken a hard [`BackendRequirement`](../../glossary.md#backend-requirement--backendrequirement), or guarantee that an
[accelerator](../../glossary.md#term-accelerator) candidate wins.

Direct construction retains the exact non-null `Optional<DeviceClass>` reference and rejects null
with `NullPointerException("preferredDeviceClass")`. `preferring(null)` rejects null with
`NullPointerException("deviceClass")`; both factories return fresh records. The value performs no
candidate enumeration or [evaluation](../../glossary.md#term-evaluation), score calculation or comparison, profile [lookup](../../glossary.md#term-lookup), ownership
or device selection, route or [kernel](../../glossary.md#kernel) selection, [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), preparation, [runtime](../../glossary.md#term-runtime) work, or
execution. Current package-private planning consumes only its optional class preference: the first
eligible class match wins, or the first eligible backend wins when no preference or match exists.
The complete compiler pipeline supplies this value to the current per-[occurrence](../../glossary.md#term-occurrence) Planning
collaboration. Backend-neutral cost classification, planning cost inputs, public numeric
scoring evaluation, and the aggregate remain planned.


---

[Next topic](planning.md)
