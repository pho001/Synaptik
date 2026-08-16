# Task 0018: Typed Functional Model Topology

## Status

Complete

## Goal

Add the first public model-composition boundary above individual NN modules. A caller defines one
typed `Model<I, O>` with a functional forward body and registers every owned child through a
short-lived topology object:

```java
var model = Model.define(topology -> {
    Linear hidden = topology.addModule(
            "hidden",
            new Linear(32, 64, true, dataType, random));
    Linear output = topology.addModule(
            "output",
            new Linear(64, 10, true, dataType, random));

    return (Tensor input) -> output.forward(hidden.forward(input).relu());
});
```

This task establishes typed composition, stable names, and construction atomicity only. Current
layers remain eagerly initialized, so the example still supplies their current constructor
dimensions. Deferred input-feature inference is the immediately following Draft capability and
must not be simulated by an incomplete parameter or hidden first-forward mutation in this task.

Mental model:

```text
definition callback
  -> collect exact name/module pairs without ownership mutation
  -> obtain one typed forward function
  -> validate the complete topology
  -> atomically attach all children in declaration order
  -> seal the topology
  -> immutable model structure with ordinary mutable Module bindings/mode
```

The generic types describe the Java input and output boundary. A simple model may infer
`Model<Tensor, Tensor>` through `var`; a structured model may use caller-owned records without NN
introducing tuples or text-specific types. The caller does not define backward behavior:
compiler-owned automatic differentiation and future Training orchestration remain unchanged.

## Scope

- Add public abstract `Model<I, O> extends Module` in the existing `nn.module` package.
- Add public functional `ModelDefinition<I, O>` whose sole method receives one `Topology` and
  returns one `ModelForward<I, O>`.
- Add public functional `ModelForward<I, O>` whose sole method maps `I` to `O`.
- Add final public `Topology` with exactly one public
  `<M extends Module> M addModule(String name, M module)` method and no public constructor.
- Add `Model.define(ModelDefinition<I, O>)` as the sole factory. Java target typing must accept an
  explicitly typed inner lambda such as `(Tensor input) -> ...` and infer the resulting generic
  model when the caller assigns it to `var`.
- Collect topology entries in encounter order without changing child ownership during the
  definition callback. After the callback returns a non-null forward function, validate the
  complete snapshot and install every child under its exact supplied name in one existing Module
  namespace.
- Add one package-private atomic named-child registration primitive to `Module`. It must preflight
  nulls, local-name grammar and collisions, repeated module identity, self/ancestor cycles, and
  existing parent ownership for the complete snapshot before installing any child or parent link.
  Its exact source-level shape is
  `final void registerNamedChildren(Map<String, ? extends Module> children)`; it copies the
  encounter-ordered snapshot and exposes no new public/protected API.
- Seal the topology before `define` returns or propagates a callback/validation failure. A captured
  topology rejects every later `addModule` call without changing either the model or candidate.
- If the definition callback throws, returns null, returns a null forward function, or complete
  topology validation fails, no candidate module becomes owned. Otherwise the model permanently
  owns the exact modules under stable names and inherited recursive state paths.
- Forward rejects a null input before invoking the body, invokes the retained body exactly once,
  rejects a null result, and returns the exact non-null result reference. It performs no wrapper,
  reflection, map lookup, module-name lookup, graph capture, compilation, or execution.
- Preserve ordinary prefix effects of the caller's forward body. If it throws or returns null,
  any Tensor expressions or module-local effects it already created remain; Model catches no
  body exception and performs no rollback.
- Allow an empty topology. It is valid when the returned forward body is valid.
- Preserve inherited `train()`/`eval()`, recursive state discovery, state dictionaries, strict
  load, and parameter replacement through the stable named child tree.
- Add focused API-shape, inference compilation, topology lifecycle/atomicity, exact ownership,
  forward reference/failure, structured input/output, mode, and state-path tests.
- Finalize affected Javadocs, package documentation, Training API, glossary, and planning evidence
  in a separate documentation-focused clean context before completion.

## Out of scope

- Lazy or deferred parameters, uninitialized Tensor placeholders, input-feature inference,
  first-forward initialization, `bind`, `build`, input specifications, dynamic parameter Shapes,
  retained random sources, or changes to current eager initializer/layer constructors.
- Changing `Parameter`, `Buffer`, `StateDictionary`, `StateEntry`, or state load/export behavior.
- A universal `forward` method on `Module`, changing `UnaryTensorModule`, or replacing
  `Sequential`.
- A mutable topology, late module addition, removal, rename, replacement, shared children,
  reflection/field discovery, annotation scanning, map-based dispatch, service locator, or builder.
- NN-owned tuple, dictionary, text, token, sequence-batch, valid-length, padding-mask, data-loader,
  tokenizer, or predictor types.
- Default recurrent states, recurrent mask/length migration, recurrent scan/control flow, or
  changes to cells and sequence containers.
- User-defined backward methods, Tensor gradient state, compiler autograd changes, optimizer or
  Training behavior, execution, runtime, prepare, Engine, or backend work.
- A new Gradle project, dependency, architecture rule, ADR, architecture test, data/text extension,
  global-roadmap implementation-order exception, CPU change, or unrelated refactoring.
- Detailed task specifications for NN 0019 or any Data/Text task.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)

## Architecture constraints

- `extensions/nn` owns stateful module composition and continues to depend only on
  `modules/model`.
- Model generic parameters are Java composition types only. They do not add Tensor semantics,
  graph values, runtime bindings, serialization schema, or backend behavior.
- `Module` remains the general state/tree/mode owner without a universal forward signature.
  `Model<I, O>` is one truthful typed subclass; existing modules do not become models
  automatically.
- Model child ownership is exclusive and permanent. Stable topology names use the existing Module
  local-name grammar and produce the existing dot-separated recursive state paths.
- Construction must preflight the complete topology before ownership installation. A failed
  functional definition must not strand caller-visible modules under an unreachable partial
  model.
- Tensor identity, descriptors, provenance, and expression construction remain Model-module
  concerns. The NN model wrapper passes exact Java references and adds no Tensor operation.
- Compiler owns automatic differentiation. The new `Model` has no backward, gradient, compile,
  prepare, run, optimizer, or session method.
- If implementation requires an architecture/dependency change, a lazy state contract, a public
  Module mutation primitive, or another package, stop and report the conflict.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.nn.module` — already owns module structure and composition; it gains
  the typed model/topology boundary and one package-private atomic registration primitive.

No package is added. Type placement:

- `io.github.pho001.synaptik.nn.module.Model` — public typed root Module and definition factory.
- `io.github.pho001.synaptik.nn.module.Topology` — definition-scoped named child collector.
- `io.github.pho001.synaptik.nn.module.ModelDefinition` — factory callback contract.
- `io.github.pho001.synaptik.nn.module.ModelForward` — retained typed forward function.
- `io.github.pho001.synaptik.nn.module.Module` — sole owner able to validate and install private
  parent/child links atomically.
- `io.github.pho001.synaptik.nn.module.ModelTest` — same-package lifecycle and structure tests.

## Public API

```java
public abstract class Model<I, O> extends Module {
    protected Model()

    public abstract O forward(I input)

    public static <I, O> Model<I, O> define(ModelDefinition<I, O> definition)
}

@FunctionalInterface
public interface ModelDefinition<I, O> {
    ModelForward<I, O> define(Topology topology);
}

@FunctionalInterface
public interface ModelForward<I, O> {
    O forward(I input);
}

public final class Topology {
    public <M extends Module> M addModule(String name, M module)
}
```

The protected Model constructor permits an advanced caller to define an ordinary subclass and use
the inherited protected `child(...)` contract. The functional factory returns a private final
implementation that retains the exact `ModelForward`; it adds no public subtype, accessor, or
forward-function exposure.

`Topology` has no public/protected constructor, collection accessor, lookup, iterator, size,
remove, or mutation method other than `addModule`. `addModule` validates null name/module and
local-name grammar/collision against entries already collected, rejects a repeated exact module
identity, retains the exact available candidate, and returns it for strongly typed local use. It
does not attach the candidate before final commit.

`Model.define` rejects null definition, creates one collector, invokes the definition once, seals
the collector in a `finally` boundary, rejects a null returned forward function, atomically commits
the snapshot, and returns the exact typed model. Callback failure is propagated unchanged after
sealing and without ownership installation.

The factory implementation's `forward` rejects null input with message `input`, invokes the exact
retained function once, and rejects null output with message `model output`. The abstract base does
not impose those checks on arbitrary caller subclasses beyond its documented non-null contract.

## Topology and failure semantics

Topology collection and Module ownership are separate phases:

1. `define` creates one independent open collector; no model or parent link exists yet.
2. Each `addModule` validates only facts knowable without changing parent links and appends one
   exact name/module pair.
3. The callback returns one exact forward function or fails.
4. The collector seals in all cases. A captured reference can never mutate the result afterward.
5. On success, the factory constructs its private model implementation with the exact forward
   function and the encounter-ordered snapshot.
6. Module preflights the complete named snapshot against that model's empty namespace, candidate
   identities, cycles, and parent ownership.
7. Only after full preflight, Module installs entries and parent links in encounter order; the
   factory then publishes the completed model.

The implementation must not call arbitrary module forward methods during definition or topology
validation. It must not attempt to inspect which registered locals the returned lambda captures;
unused registered modules remain valid owned children and included state.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java` (new)
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelDefinition.java` (new)
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelForward.java` (new)
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Topology.java` (new)
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/package-info.java`

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java` (new)

Expected documentation and planning files:

- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/extensions/nn/master-plan.md`
- this task specification

## Maximum scope

This task may create or modify at most the eleven paths listed above. If implementation needs a
layer edit, `Parameter`/state-dictionary change, another test owner, a new package, a twelfth path,
or any build/architecture/dependency file, stop and propose a separate task.

## Acceptance criteria

- The exact public API exists with no extra facade, builder, lookup, mutation, context, state,
  compile, training, or execution surface.
- A Java external-package compilation test proves `var model = Model.define(...)` infers a simple
  Tensor input/output model from the explicitly typed inner forward lambda and supports different
  caller-defined record input/output types.
- Topology preserves exact declaration order, names, module identities, and typed return values.
  The resulting model's children, recursive parameter/buffer discovery, and state dictionary use
  stable descriptive paths such as `hidden.weight` and `output.bias`.
- Empty topology succeeds. Null definition, null callback result, null/invalid/duplicate names,
  null module, repeated identity, owned module, and defensive cycle failures follow documented
  categories and install no child or parent link.
- A definition callback that throws after several `addModule` calls leaves every candidate
  reusable by another owner and propagates the exact failure.
- A topology captured by caller code is sealed after every success or failure path; a late call
  fails without changing the completed model or candidate ownership.
- Functional forward validates null input, calls the exact body once, returns the exact output,
  rejects null output, propagates body exceptions unchanged, calls no hidden module, and preserves
  already-created body prefix effects.
- Model train/eval propagation, parameter replacement, state export, and strict load work through
  descriptive child paths without a special case.
- Model has no backward method. No existing Module/layer/Sequential/recurrent API or behavior
  changes beyond the package-private atomic registration support needed by this capability.
- Public and package Javadocs document generic typing, ownership, sealing, construction atomicity,
  non-null and failure behavior, forward prefix effects, state/mode inheritance, threading, and
  explicit deferred-binding/training/execution exclusions with complete tags.
- The Training API and glossary distinguish Model topology from Tensor graph structure and label
  deferred input-dimension inference as planned rather than implemented.
- A separate clean-context documentation pass finalizes all affected documentation/Javadocs and
  records glossary impact before completion.

## Tests / validation

Implementation context:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModelTest
./gradlew :extensions:nn:test
git diff --check
```

The separate documentation context reuses stable Java-test evidence unless it changes executable
behavior and runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also checks generated pages, exact public/protected surfaces, one external Java compilation
example for Tensor and record boundaries, Model-only production imports, Markdown links/anchors,
balanced fences, terminal newlines, trailing whitespace, and the exact path set. Repository-wide,
architecture, conformance, integration, numerical, compiler, and backend suites remain deferred
because this task changes no dependency or architecture boundary.

## Dependencies

- NN 0001–0017 are Complete.
- Module tree ownership, state dictionaries, and typed unary composition are stable.
- Model Tensor expression construction and compiler-owned autograd boundaries are stable.
- No dependency on the Draft Data/Text extensions, lazy binding, CPU, Engine, runtime, prepare,
  or Training implementation.

## Follow-up tasks

- NN 0019: define the deferred parameter binding/build lifecycle and implement lazy Linear input
  width without incomplete or partially published state.
- NN 0020: extend the proven lifecycle to initialized Embedding and lazy recurrent input weights,
  plus inferred zero initial recurrent states.
- Data/Text planning: add canonical valid-length metadata and tokenizer/batching boundaries only
  after their architecture/module decision.
- NN 0021–0022: establish genuine runtime recurrent scan/input binding before adding the
  Data-owned valid-length recurrent API; do not present current host-static specialization or a
  dense mask as skipped runtime work.

## Architecture impact

Expected impact: None. The existing contract assigns stateful module composition to
`extensions/nn`. Stop if implementation needs another module or dependency.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not use GSD. Do not commit or push unless the user
explicitly requests it for the completed change.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/roadmap.md, docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0018-typed-functional-model-topology.md in full. Read the final
Module, Sequential, state-dictionary source/tests/Javadocs and directly affected API/glossary
documentation.

Implement task 0018 exactly as specified. Stop on architecture, package, public-API, scope, or
atomic-ownership uncertainty rather than inventing a broader design. Run the focused and final NN
validation once after executable code stabilizes.

Then hand the final diff and exact Java-test evidence to a separate documentation-focused clean
agent/thread. That pass must follow docs/developer-guide/documentation-rules.md, independently
finalize affected Javadocs, Training API, glossary and planning evidence, run final Javadoc and
documentation checks, and update this task completion record before it may become Complete.
```

## Local decisions

- Select `addModule`, not `layer` or `addLayer`: every registered child is a Module, including a
  nested Model or `Sequential`, and stable ownership is the operation being performed.
- Keep Model generics at the Java composition boundary. Tensor-only callers normally use `var`;
  structured callers define domain records without NN-owned tuples.
- Collect before attaching so arbitrary definition callback failure cannot strand modules under
  an unreachable partial model.
- Keep eager constructors unchanged. Adding typed composition first makes the later lazy lifecycle
  independently reviewable and prevents uninitialized state from being smuggled into this task.
- Keep backward outside Model. The returned forward body constructs ordinary Tensor expressions,
  which the compiler differentiates through its existing functional request boundary.

## Known limitations

- Current layers still require their current eager parameter Shapes and initialization inputs.
- The factory implementation requires non-null input and output; it does not model optional Java
  values.
- Model structure is immutable after definition, while existing parameter/buffer bindings and mode
  retain their documented mutable, caller-coordinated behavior.
- No data/text batching, valid-length input, recurrent default state, compile reuse, or execution
  capability is added.

## Validation evidence

Clean implementation context `/root/nn_0018_typed_model_implementation` read the required
architecture, planning, completed NN ownership/composition/state tasks, final NN source/tests,
documentation rules, and selected General/API-Javadoc/Planning profiles. It found no architecture,
package, dependency, public-surface, or atomic-ownership blocker and preserved the concurrent
Data/Text/Vision/Checkpoint, Training, Engine, roadmap, and CPU planning work.

The first focused command
`./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModelTest` passed after
the initial executable draft with one suite and ten tests. The implementation then strengthened
the same focused test to prove that a sealed late-registration candidate and a candidate collected
before a null definition result both remain reusable. The stabilized rerun of the same command
passed one suite and ten tests with zero failures, errors, or skips.

After executable Java and tests stabilized, the sole authoritative
`./gradlew :extensions:nn:test` command passed 30 suites and 207 tests with zero failures, errors,
or skips. No executable Java or test changed afterward. The ten Model tests cover the exact API
surface, external-package Java type inference for Tensor and caller-record boundaries, ordered
descriptive ownership/state paths, empty topology, all sealing paths, collection and complete
ownership failures, reusable candidates, package-private atomic cycle/collision/ownership
preflight, exact forward references and calls, null/throw prefix effects, inherited mode,
replacement, state export, and strict load.

Preliminary implementation validation also passed:

- `./gradlew :extensions:nn:javadoc` completed successfully before the documentation handoff;
  the separate documentation context must rerun it after final Javadoc edits.
- `javap -public` confirmed exactly the planned public Model, ModelDefinition, ModelForward, and
  Topology surfaces. Private inspection confirmed package-private final
  `registerNamedChildren(Map)` and no Module or Model backward method.
- Production import and forbidden-mechanism scans found only JDK collections and the existing
  Model dependency, with no Training, compiler, runtime, prepare, Engine, backend, reflection,
  serialization, lookup-dispatch, or service-registry implementation.
- Preliminary `git diff --check` passed. Final combined whitespace and exact eleven-path checks
  remain part of the documentation pass because its Training API, glossary, and planning edits
  are still outstanding.

Repository-wide, architecture, conformance, integration, numerical, compiler, runtime, Engine,
and backend suites remain deliberately deferred under the task validation tier because the
implementation changes only the existing NN module and no build or dependency boundary.

Independent clean documentation context `/root/nn_0018_docs` read the required architecture,
module-boundary, dependency, training-graph, ADR, documentation-profile, planning, prior NN task,
final implementation/test, Module/Sequential/state, generated-Javadoc, Training API, and glossary
contracts in full. It found no executable, public-API, ownership-atomicity, architecture, or scope
defect and changed no executable Java or test. It therefore reused the stable focused and
authoritative NN test evidence above.

The documentation pass finalized all affected public/package Javadocs plus the Training API and
glossary distinction between the NN Model ownership tree and modules/model Tensor or compiled
graph structure. `./gradlew :extensions:nn:javadoc` then passed, and rendered Model,
ModelDefinition, ModelForward, Topology, Module, and package pages were inspected for the final
generic typing, ownership/sealing, atomic failure, exact-reference/prefix-effect, inherited
state/mode, threading, and no-lazy/no-training/no-execution contracts. Final `javap` confirmed the
exact four-type public surface, package-private final named-child registration, and no backward
method. A standalone external-package `javac` probe compiled both the documented eager Linear
Tensor model with `var` inference and a caller-record input/output model.

Production import and forbidden-mechanism scans passed with only JDK collections plus the existing
modules/model dependency. Local Markdown files and the new Training API anchor resolved, fences
were balanced, the final scope was exactly the eleven authorized paths, every modified file had a
terminal newline and no trailing whitespace, and `git diff --check` passed. No Compile API,
Tensor API, architecture contract/explanation/test, conformance/integration suite, Gradle file,
Data/Text/Vision/Checkpoint plan, Training/Engine plan, roadmap, CPU file, or other module required
a task-owned change.

## Implementation notes

- Added the four exact public composition types in `nn.module`. `Model.define` invokes one
  definition, seals its Topology in a `finally` boundary, rejects a null forward function, and
  publishes a private functional Model only after complete named-child ownership validation.
- Added package-private final `Module.registerNamedChildren(Map<String, ? extends Module>)`. It
  copies encounter order, validates all nulls, names/collisions, repeated identities, cycles, and
  existing parent ownership before changing any child entry or parent link, then installs the
  exact snapshot in order.
- Functional forward rejects null input, calls the retained function exactly once, rejects null
  output, returns the exact result reference, and catches no body failure. It performs no module
  lookup, reflection, graph work, initialization, compilation, training, or execution.
- Added one focused same-package suite with an in-memory external-package Java compilation probe.
  Existing Module, Sequential, layer, recurrent, state-dictionary, build, and dependency behavior
  otherwise remains unchanged.
- Drafted affected public/package Javadocs for independent review. The clean documentation context
  then finalized them plus Training API, glossary, planning evidence, and documentation validation
  without rerunning the stable Java tests because it changed no executable behavior.

## Completion summary

- Completed changes: Implemented typed functional Model definition, sealed named Topology
  collection, complete validate-before-install named child ownership, exact functional forward
  behavior, descriptive inherited state/mode behavior, and focused coverage.
- Files changed or created: `Module.java`, `Model.java`, `ModelDefinition.java`,
  `ModelForward.java`, `Topology.java`, module `package-info.java`, `ModelTest.java`, Training API,
  glossary, the NN master plan, and this task record: exactly the eleven authorized paths.
- Tests and validation: Focused Model one-suite/ten-test pass and authoritative NN 30-suite/207-
  test pass, both with zero failures, errors, or skips; final Javadoc/rendered-page, `javap`,
  external-use, import/mechanism, Markdown, scope, and whitespace checks passed.
- Documentation-agent review: Independent clean context `/root/nn_0018_docs` found no executable,
  API, architecture, or scope defect and finalized the authorized documentation.
- Documentation impact: Public/package Javadocs, Training API, glossary, and planning evidence now
  explain the typed boundary, atomic ownership, sealing, inherited lifecycle, failure effects, and
  planned boundaries without confusing NN topology with a Tensor or compiled graph.
- Javadoc review: Final generation and rendered-page inspection passed after the documentation
  edits.
- Glossary impact: Added the functional Model and Model-topology term, cross-linked it to the
  Training API, and corrected the current recurrent-container inventory.
- Unresolved issues: No executable or architecture issue found.
- Follow-up required: None for task 0018. NN 0019 remains Draft pending its explicit deferred-state
  lifecycle decisions.

Status: Complete
