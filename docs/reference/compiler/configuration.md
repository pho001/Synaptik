# Compile-configuration inputs

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-compile-configuration-inputs"></a>
## Current compile-configuration inputs

`io.github.pho001.synaptik.config.compile.BackendIntent` is current declarative compile
configuration. It holds exactly one `Optional<BackendRequirement>` named `hardRequirement`:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda =
        BackendIntent.requiring(
                new BackendIdRequirement(new BackendId("cuda")));
```

The first value has no hard eligibility target. The second retains the exact supplied requirement
reference for Planning. Neither construction evaluates availability or capability, ranks
candidates, chooses ownership, discovers a service, or invokes a compiler. In particular,
`unconstrained()` does not mean “automatic backend selection succeeded”; it records only the
absence of a hard target.

The canonical constructor rejects a null optional with `NullPointerException("hardRequirement")`.
The `requiring` factory rejects a null requirement with `NullPointerException("requirement")`.
Current internal planning evaluates the hard target before its cost-free baseline owner selection,
and the complete compiler pipeline supplies this intent once per final graph node. General score
calculation, planning cost profiles, `CompileConfig`, and Engine-owned failure translation remain
planned. The separate current `PartitionScoringConfig` value
described below supplies only soft preference input; it does not change these hard-intent
semantics.

`io.github.pho001.synaptik.config.compile.CompileMode` is also current. It contains exactly three
requested graph scopes in declaration order:

```java
import io.github.pho001.synaptik.config.compile.CompileMode;

CompileMode inferenceScope = CompileMode.FORWARD_ONLY;
CompileMode gradientScope = CompileMode.FORWARD_AND_BACKWARD;
CompileMode trainingStepDirection = CompileMode.TRAINING_STEP;
```

`FORWARD_ONLY` requests forward graph construction and requested forward publications.
Package-private `GraphCompiler` currently interprets it as a forward-only graph-stage or complete
artifact request and requires the functional request to be absent. `FORWARD_AND_BACKWARD`
requests current package-private one/two-stage pre-capture autograd plus combined
forward/backward graph-stage work.
`TRAINING_STEP` currently performs the same internal graph and artifact construction, but the enum value
does not itself introduce an optimizer, optimizer-update graph, session, publication delivery,
schedule, or execution behavior. `GraphCompilationPort` and `AdvancedEngine.compile(...)` accept
all three values subject to the existing mode/request validation. The advanced Engine adds no
training-step execution, backward convenience, or guarantee that the resulting artifact can pass
its CPU-only advanced prepare restriction.

`io.github.pho001.synaptik.config.compile.GraphOptimizationConfig` is another current value:

```java
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;

GraphOptimizationConfig noOptionalOptimization =
        GraphOptimizationConfig.disabled();
GraphOptimizationConfig standardOptimization =
        GraphOptimizationConfig.standard();
```

The first value requests that the compiler skip optional semantics-preserving optimization.
It cannot disable graph capture, ordering, inference, validation, mandatory canonical
representation, mode-required autograd, publication binding, planning, preparation, or execution.
The second permits the compiler's standard optional semantics-preserving pipeline without freezing
a pass list, pass order, internal graph shape, or implementation strategy. It permits no
approximate mathematics, changed numerical semantics, backend-specific fusion, preparation, or
execution behavior. Direct construction retains either primitive boolean value, and both factories
return fresh values. The package-private compiler transformation boundary consumes this
permission through the public integration port; no `CompileConfig` aggregate or ordinary-user
compile facade does.

`io.github.pho001.synaptik.config.compile.PartitionScoringConfig` is the fourth current value. It
holds exactly one `Optional<DeviceClass>` named `preferredDeviceClass`:

```java
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

The first value contains an empty optional and means only that no explicit coarse device-class
preference was supplied. It selects no aggregate default or fallback and promises neither equal
candidate scores nor successful ownership selection. The second contains the exact supplied enum
reference as a soft preference that planning may apply only after hard eligibility. It does
not remove another eligible candidate, weaken a hard `BackendRequirement`, or guarantee that an
accelerator candidate wins.

Direct construction retains the exact non-null `Optional<DeviceClass>` reference and rejects null
with `NullPointerException("preferredDeviceClass")`. `preferring(null)` rejects null with
`NullPointerException("deviceClass")`; both factories return fresh records. The value performs no
candidate enumeration or evaluation, score calculation or comparison, profile lookup, ownership
or device selection, route or kernel selection, compilation, preparation, runtime work, or
execution. Current package-private planning consumes only its optional class preference: the first
eligible class match wins, or the first eligible backend wins when no preference or match exists.
The complete compiler pipeline supplies this value to the current per-occurrence Planning
collaboration. Backend-neutral cost classification, planning cost inputs, public numeric
scoring evaluation, and the aggregate remain planned.


---

[Previous topic](geometry-qualification.md) · [Next topic](planning.md)
