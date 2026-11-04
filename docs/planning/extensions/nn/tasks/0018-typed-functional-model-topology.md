# Task 0018: Typed Functional [Model](../../../../glossary.md#term-model) [Topology](../../../../glossary.md#term-topology)

## Status

Complete

## Goal

Add the first public [model](../../../../glossary.md#term-model)-composition boundary above individual [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [modules](../../../../glossary.md#term-module). A caller defines one
typed `Model<I, O>` with a functional forward body and registers every owned child through a
short-lived [topology](../../../../glossary.md#term-topology) object:

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

This task establishes typed composition, stable names, and construction [atomicity](../../../../glossary.md#term-atomicity) only. Current
layers remain eagerly initialized, so the example still supplies their current constructor
[dimensions](../../../../glossary.md#dimension). Deferred input-[feature](../../../../glossary.md#term-feature) [inference](../../../../glossary.md#term-inference) is the immediately following Draft capability and
must not be simulated by an incomplete [parameter](../../../../glossary.md#term-parameter) or hidden first-forward mutation in this task.

[Mental model](../../../../glossary.md#term-mental-model):

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
[compiler](../../../../glossary.md#term-compiler)-owned [automatic differentiation](../../../../glossary.md#autograd) and future Training orchestration remain unchanged.

## [Scope](../../../../glossary.md#term-scope)

- Add public abstract `Model<I, O> extends Module` in the existing `nn.module` package.
- Add public functional `ModelDefinition<I, O>` whose sole method receives one [`Topology`](../../../../glossary.md#term-topology) and
  returns one `ModelForward<I, O>`.
- Add public functional `ModelForward<I, O>` whose sole method maps `I` to `O`.
- Add final public `Topology` with exactly one public
  `<M extends Module> M addModule(String name, M module)` method and no public constructor.
- Add `Model.define(ModelDefinition<I, O>)` as the sole factory. Java target typing must accept an
  explicitly typed inner lambda such as <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`(Tensor input) -> ...` and infer the resulting generic
  model when the caller assigns it to `var`.
- Collect topology entries in encounter order without changing child ownership during the
  definition callback. After the callback returns a non-null forward function, validate the
  complete snapshot and install every child under its exact supplied name in one existing Module
  namespace.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [callback glossary](../../../../glossary.md#term-callback); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Add one package-private [atomic](../../../../glossary.md#resource-transaction-and-rollback) named-child registration primitive to [`Module`](../../../../glossary.md#term-module). It must preflight
  nulls, local-name grammar and collisions, repeated module identity, self/ancestor cycles, and
  existing parent ownership for the complete snapshot before installing any child or parent link.
  Its exact source-level shape is<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings); [shape glossary](../../../../glossary.md#shape))</span>
  `final void registerNamedChildren(Map<String, ? extends Module> children)`; it copies the
  encounter-ordered snapshot and exposes no new public/protected API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Seal the topology before `define` returns or propagates a callback/validation failure. A captured
  topology rejects every later <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`addModule` call without changing either the [model](../../../../glossary.md#term-model) or candidate.
- If the definition callback throws, returns null, returns a null forward function, or complete
  topology validation fails, no candidate module becomes owned. Otherwise the model permanently
  owns the exact modules under stable names and inherited recursive state paths.
- Forward rejects a null input before invoking the body, invokes the retained body exactly once,
  rejects a null result, and returns the exact non-null result reference. It performs no wrapper,
  reflection, map lookup, module-name lookup, graph capture, compilation, or execution.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Preserve ordinary prefix effects of the caller's forward body. If it throws or returns null,
  any Tensor expressions or module-local effects it already created remain; Model catches no
  body exception and performs no rollback.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Allow an empty topology. It is valid when the returned forward body is valid.
- Preserve inherited `train()`/`eval()`, recursive state discovery, state dictionaries, strict
  load, and parameter replacement through the stable named child tree.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Add focused API-shape, inference compilation, topology lifecycle/atomicity, exact ownership,
  forward reference/failure, structured input/output, mode, and state-path tests.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [lifecycle glossary](../../../../glossary.md#lifecycle); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- Finalize affected Javadocs, package documentation, Training API, glossary, and planning evidence
  in a separate documentation-focused clean context before completion.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Lazy or deferred parameters, uninitialized Tensor placeholders, input-feature inference,
  first-forward initialization, <span data-glossary-navigation="true"> ([Lazy glossary](../../../../glossary.md#term-lazy-evaluation); [Tensor glossary](../../../../glossary.md#tensor); [feature glossary](../../../../glossary.md#term-feature); [inference glossary](../../../../glossary.md#term-inference))</span>`bind`, `build`, input specifications, dynamic parameter Shapes,
  retained random sources, or changes to current eager initializer/layer constructors.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Changing [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `StateDictionary`, `StateEntry`, or state load/export behavior.
- A universal `forward` method on [`Module`](../../../../glossary.md#term-module), changing `UnaryTensorModule`, or replacing
  `Sequential`.
- A mutable topology, late module addition, removal, rename, replacement, shared children,
  reflection/field discovery, annotation scanning, map-based dispatch, service locator, or builder.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [topology glossary](../../../../glossary.md#term-topology); [dispatch glossary](../../../../glossary.md#term-dispatch); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- NN-owned tuple, dictionary, text, token, sequence-batch, valid-length, padding-mask, data-loader,
  tokenizer, or predictor types.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding); [mask glossary](../../../../glossary.md#term-boolean-mask); [tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer))</span>
- Default recurrent states, recurrent mask/length migration, recurrent scan/control flow, or
  changes to cells and sequence containers.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- User-defined backward methods, Tensor gradient state, compiler autograd changes, optimizer or
  Training behavior, execution, runtime, prepare, Engine, or backend work.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- A new Gradle project, dependency, architecture rule, ADR, architecture test, data/text extension,
  global-roadmap implementation-order exception, CPU change, or unrelated refactoring.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed task specifications for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0019 or any Data/Text task.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary))</span>

## Architecture constraints

- `extensions/nn` owns stateful [module](../../../../glossary.md#term-module) composition and continues to depend only on
  `modules/model`.
- Model generic parameters are Java composition types only. They do not add Tensor semantics,
  graph values, runtime bindings, serialization schema, or backend behavior.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [bindings glossary](../../../../glossary.md#term-binding); [serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema); [backend glossary](../../../../glossary.md#backend))</span>
- `Module` remains the general state/tree/mode [owner](../../../../glossary.md#term-owner) without a universal forward [signature](../../../../glossary.md#term-signature).
  `Model<I, O>` is one truthful typed subclass; existing modules do not become models
  automatically.
- Model child ownership is exclusive and permanent. Stable topology names use the existing Module
  local-name grammar and produce the existing dot-separated recursive state paths.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [topology glossary](../../../../glossary.md#term-topology))</span>
- Construction must preflight the complete topology before ownership installation. A failed
  functional definition must not strand caller-visible modules under an unreachable partial
  model.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Tensor identity, descriptors, provenance, and expression construction remain Model-module
  concerns. The NN model wrapper passes exact Java references and adds no Tensor operation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [NN model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [operation glossary](../../../../glossary.md#operation))</span>
- [Compiler](../../../../glossary.md#term-compiler) owns [automatic differentiation](../../../../glossary.md#autograd). The new [`Model`](../../../../glossary.md#term-model) has no backward, gradient, compile,
  prepare, run, optimizer, or session method.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- If implementation requires an architecture/dependency change, a lazy state contract, a public
  Module mutation primitive, or another package, stop and report the conflict.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.nn.module` — already owns module structure and composition; it gains
  the typed model/topology boundary and one package-private atomic registration primitive.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [topology glossary](../../../../glossary.md#term-topology); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

No package is added. Type placement:

- `io.github.pho001.synaptik.nn.module.Model` — public typed root [Module](../../../../glossary.md#term-module) and definition factory.
- `io.github.pho001.synaptik.nn.module.Topology` — definition-scoped named child collector.
- `io.github.pho001.synaptik.nn.module.ModelDefinition` — factory [callback](../../../../glossary.md#term-callback) contract.
- `io.github.pho001.synaptik.nn.module.ModelForward` — retained typed forward function.
- `io.github.pho001.synaptik.nn.module.Module` — sole owner able to validate and install private
  parent/child links atomically.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.nn.module.ModelTest` — same-package [lifecycle](../../../../glossary.md#lifecycle) and structure tests.

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

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

The protected [Model](../../../../glossary.md#term-model) constructor permits an advanced caller to define an ordinary subclass and use
the inherited protected `child(...)` contract. The functional factory returns a private final
implementation that retains the exact `ModelForward`; it adds no public subtype, accessor, or
forward-function exposure.

[`Topology`](../../../../glossary.md#term-topology) has no public/protected constructor, collection accessor, [lookup](../../../../glossary.md#term-lookup), iterator, size,
remove, or mutation method other than `addModule`. `addModule` validates null name/[module](../../../../glossary.md#term-module) and
local-name grammar/collision against entries already collected, rejects a repeated exact module
[identity](../../../../glossary.md#term-identity-meanings), retains the exact available candidate, and returns it for strongly typed local use. It
does not attach the candidate before final [commit](../../../../glossary.md#term-commit).

`Model.define` rejects null definition, creates one collector, invokes the definition once, seals
the collector in a `finally` boundary, rejects a null returned forward function, atomically [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)
the [snapshot](../../../../glossary.md#term-snapshot), and returns the exact typed model. [Callback](../../../../glossary.md#term-callback) failure is propagated unchanged after
sealing and without [ownership](../../../../glossary.md#term-ownership-meanings) installation.

The factory implementation's `forward` rejects null input with message `input`, invokes the exact
retained function once, and rejects null output with message `model output`. The abstract base does
not impose those checks on arbitrary caller subclasses beyond its documented non-null contract.

## [Topology](../../../../glossary.md#term-topology) and failure semantics

Topology collection and [Module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) are separate phases:

1. `define` creates one independent open collector; no [model](../../../../glossary.md#term-model) or parent link exists yet.
2. Each `addModule` validates only facts knowable without changing parent links and appends one
   exact name/module pair.
3. The [callback](../../../../glossary.md#term-callback) returns one exact forward function or fails.
4. The collector seals in all cases. A captured reference can never mutate the result afterward.
5. On success, the factory constructs its private model implementation with the exact forward
   function and the encounter-ordered snapshot.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
6. Module preflights the complete named snapshot against that model's empty namespace, candidate
   identities, cycles, and parent ownership.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
7. Only after full preflight, Module installs entries and parent links in encounter order; the
   factory then publishes the completed model.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>

The implementation must not call arbitrary module forward methods during definition or topology
[validation](../../../../glossary.md#term-validation). It must not attempt to inspect which registered locals the returned lambda [captures](../../../../glossary.md#term-capture-meanings);
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

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/extensions/nn/master-plan.md`
- this task specification

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the eleven paths listed above. If implementation needs a
layer edit, [`Parameter`](../../../../glossary.md#term-parameter)/state-dictionary change, another test [owner](../../../../glossary.md#term-owner), a new package, a twelfth path,
or any build/architecture/[dependency](../../../../glossary.md#term-dependency) file, stop and propose a separate task.

## Acceptance criteria

- The exact public API exists with no extra facade, builder, lookup, mutation, context, state,
  compile, training, or execution surface.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [lookup glossary](../../../../glossary.md#term-lookup); [compile glossary](../../../../glossary.md#compile))</span>
- A Java external-package [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) test proves `var model = Model.define(...)` infers a simple
  Tensor input/output model from the explicitly typed inner forward lambda and supports different
  caller-defined record input/output types.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- Topology preserves exact declaration order, names, module identities, and typed return values.
  The resulting model's children, recursive parameter/buffer discovery, and state dictionary use
  stable descriptive paths such as <span data-glossary-navigation="true"> ([Topology glossary](../../../../glossary.md#term-topology); [module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer))</span>`hidden.weight` and `output.bias`.
- Empty topology succeeds. Null definition, null callback result, null/invalid/duplicate names,
  null module, repeated identity, owned module, and defensive cycle failures follow documented
  categories and install no child or parent link.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- A definition [callback](../../../../glossary.md#term-callback) that throws after several `addModule` calls leaves every candidate
  reusable by another owner and propagates the exact failure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- A topology captured by caller code is sealed after every success or failure path; a late call
  fails without changing the completed model or candidate ownership.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Functional forward validates null input, calls the exact body once, returns the exact output,
  rejects null output, propagates body exceptions unchanged, calls no hidden module, and preserves
  already-created body prefix effects.
- Model train/eval propagation, parameter replacement, state export, and strict load work through
  descriptive child paths without a special case.
- Model has no backward method. No existing Module/layer/Sequential/recurrent API or behavior
  changes beyond the package-private atomic registration support needed by this capability.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Public and package Javadocs document generic typing, ownership, sealing, construction atomicity,
  non-null and failure behavior, forward prefix effects, state/mode inheritance, threading, and
  explicit deferred-binding/training/execution exclusions with complete tags.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity); [binding glossary](../../../../glossary.md#term-binding))</span>
- The Training API and glossary distinguish Model topology from Tensor graph structure and label
  deferred input-dimension inference as planned rather than implemented.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dimension glossary](../../../../glossary.md#dimension); [inference glossary](../../../../glossary.md#term-inference))</span>
- A separate clean-context documentation pass finalizes all affected documentation/Javadocs and
  records glossary impact before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

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

It also checks generated pages, exact public/protected surfaces, one external Java [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)
example for [Tensor](../../../../glossary.md#tensor) and record boundaries, [Model](../../../../glossary.md#term-model)-only production imports, Markdown links/anchors,
balanced fences, terminal newlines, trailing whitespace, and the exact path set. Repository-wide,
architecture, [conformance](../../../../glossary.md#term-conformance), integration, numerical, [compiler](../../../../glossary.md#term-compiler), and [backend](../../../../glossary.md#backend) suites remain deferred
because this task changes no [dependency](../../../../glossary.md#term-dependency) or architecture boundary.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0017 are Complete.
- [Module](../../../../glossary.md#term-module) tree [ownership](../../../../glossary.md#term-ownership-meanings), state dictionaries, and typed [unary](../../../../glossary.md#term-unary) composition are stable.
- [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) expression construction and [compiler](../../../../glossary.md#term-compiler)-owned [autograd](../../../../glossary.md#autograd) boundaries are stable.
- No dependency on the Draft Data/Text extensions, lazy binding, CPU, Engine, runtime, prepare,
  or Training implementation.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [binding glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>

## Follow-up tasks

- NN 0019: define the deferred parameter binding/build lifecycle and implement lazy Linear input
  width without incomplete or partially published state.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- NN 0020: extend the proven lifecycle to initialized Embedding and lazy recurrent input weights,
  plus inferred zero initial recurrent states.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [weights glossary](../../../../glossary.md#term-weight))</span>
- Data/Text planning: add canonical valid-length metadata and tokenizer/batching boundaries only
  after their architecture/module decision.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [canonical glossary](../../../../glossary.md#term-canonical-form); [metadata glossary](../../../../glossary.md#term-metadata); [tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [module glossary](../../../../glossary.md#term-module))</span>
- NN 0021–0022: establish genuine runtime recurrent scan/input binding before adding the
  Data-owned valid-length recurrent API; do not present current host-static specialization or a
  dense mask as skipped runtime work.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [host glossary](../../../../glossary.md#term-host); [specialization glossary](../../../../glossary.md#term-specialization); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>

## Architecture impact

Expected impact: None. The existing contract assigns stateful [module](../../../../glossary.md#term-module) composition to
`extensions/nn`. Stop if implementation needs another module or [dependency](../../../../glossary.md#term-dependency).

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
  nested Model or <span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>`Sequential`, and stable [ownership](../../../../glossary.md#term-ownership-meanings) is the [operation](../../../../glossary.md#operation) being performed.
- Keep [Model](../../../../glossary.md#term-model) generics at the Java composition boundary. [Tensor](../../../../glossary.md#tensor)-only callers normally use `var`;
  structured callers define domain records without NN-owned tuples.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Collect before attaching so arbitrary definition callback failure cannot strand modules under
  an unreachable partial model.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback))</span>
- Keep eager constructors unchanged. Adding typed composition first makes the later lazy lifecycle
  independently reviewable and prevents uninitialized state from being smuggled into this task.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Keep backward outside Model. The returned forward body constructs ordinary Tensor expressions,
  which the compiler differentiates through its existing functional request boundary.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>

## Known limitations

- Current layers still require their current [eager](../../../../glossary.md#term-eager-evaluation) [parameter](../../../../glossary.md#term-parameter) [Shapes](../../../../glossary.md#shape) and initialization inputs.
- The factory implementation requires non-null input and output; it does not model optional Java
  values.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Model structure is immutable after definition, while existing parameter/buffer bindings and mode
  retain their documented mutable, caller-coordinated behavior.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer); [bindings glossary](../../../../glossary.md#term-binding); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- No data/text batching, valid-length input, recurrent default state, compile reuse, or execution
  capability is added.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Clean implementation context `/root/nn_0018_typed_model_implementation` read the required
architecture, [planning](../../../../glossary.md#term-planning), completed [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings)/composition/state tasks, final NN source/tests,
documentation rules, and selected General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning profiles. It found no architecture,
package, [dependency](../../../../glossary.md#term-dependency), public-surface, or [atomic](../../../../glossary.md#resource-transaction-and-rollback)-ownership blocker and preserved the concurrent
Data/Text/Vision/[Checkpoint](../../../../glossary.md#term-checkpoint), Training, [Engine](../../../../glossary.md#term-engine), roadmap, and [CPU](../../../../glossary.md#term-cpu) planning work.

The first focused command
`./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModelTest` passed after
the initial executable draft with one suite and ten tests. The implementation then strengthened
the same focused test to prove that a [sealed](../../../../glossary.md#term-sealed) late-registration candidate and a candidate collected
before a null definition result both remain reusable. The stabilized rerun of the same command
passed one suite and ten tests with zero failures, errors, or skips.

After executable Java and tests stabilized, the sole authoritative
`./gradlew :extensions:nn:test` command passed 30 suites and 207 tests with zero failures, errors,
or skips. No executable Java or test changed afterward. The ten [Model](../../../../glossary.md#term-model) tests cover the exact API
surface, external-package Java type [inference](../../../../glossary.md#term-inference) for [Tensor](../../../../glossary.md#tensor) and caller-record boundaries, ordered
descriptive ownership/state paths, empty [topology](../../../../glossary.md#term-topology), all sealing paths, collection and complete
ownership failures, reusable candidates, package-private atomic cycle/collision/ownership
[preflight](../../../../glossary.md#term-preflight), exact forward references and calls, null/throw prefix effects, inherited mode,
replacement, state export, and strict load.

Preliminary implementation validation also passed:

- `./gradlew :extensions:nn:javadoc` completed successfully before the documentation handoff;
  the separate documentation context must rerun it after final Javadoc edits.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- `javap -public` confirmed exactly the planned public Model, ModelDefinition, ModelForward, and
  Topology surfaces. Private inspection confirmed package-private final
  `registerNamedChildren(Map)` and no [Module](../../../../glossary.md#term-module) or Model backward method.
- Production import and forbidden-mechanism scans found only JDK collections and the existing
  Model dependency, with no Training, compiler, runtime, prepare, Engine, backend, reflection,
  serialization, lookup-dispatch, or service-registry implementation.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [serialization glossary](../../../../glossary.md#term-serialization); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry))</span>
- Preliminary `git diff --check` passed. Final combined whitespace and exact eleven-path checks
  remain part of the documentation pass because its Training API, glossary, and planning edits
  are still outstanding.

Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), integration, numerical, [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), Engine,
and [backend](../../../../glossary.md#backend) suites remain deliberately deferred under the task validation tier because the
implementation changes only the existing NN module and no build or dependency boundary.

Independent clean documentation context `/root/nn_0018_docs` read the required architecture,
module-boundary, dependency, training-[graph](../../../../glossary.md#graph), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), documentation-profile, planning, prior NN task,
final implementation/test, Module/Sequential/state, generated-Javadoc, Training API, and glossary
contracts in full. It found no executable, public-API, ownership-[atomicity](../../../../glossary.md#term-atomicity), architecture, or [scope](../../../../glossary.md#term-scope)
defect and changed no executable Java or test. It therefore reused the stable focused and
authoritative NN test evidence above.

The documentation pass finalized all affected public/package Javadocs plus the Training API and
glossary distinction between the [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) ownership tree and modules/model Tensor or [compiled
graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) structure. `./gradlew :extensions:nn:javadoc` then passed, and rendered Model,
ModelDefinition, ModelForward, Topology, Module, and package pages were inspected for the final
generic typing, ownership/sealing, atomic failure, exact-reference/prefix-effect, inherited
state/mode, threading, and no-[lazy](../../../../glossary.md#term-lazy-evaluation)/no-training/no-execution contracts. Final `javap` confirmed the
exact four-type public surface, package-private final named-child registration, and no backward
method. A standalone external-package `javac` probe compiled both the documented [eager](../../../../glossary.md#term-eager-evaluation) [Linear](../../../../glossary.md#term-linear-meanings)
Tensor model with `var` inference and a caller-record input/output model.

Production import and forbidden-mechanism scans passed with only [JDK](../../../../glossary.md#term-java-development-kit-jdk) collections plus the existing
modules/model dependency. Local Markdown files and the new Training API anchor resolved, fences
were balanced, the final scope was exactly the eleven authorized paths, every modified file had a
terminal newline and no trailing whitespace, and `git diff --check` passed. No [Compile](../../../../glossary.md#compile) API,
Tensor API, [architecture contract](../../../../glossary.md#architecture-contract)/explanation/test, conformance/integration suite, [Gradle](../../../../glossary.md#term-gradle) file,
Data/Text/Vision/Checkpoint plan, Training/Engine plan, roadmap, CPU file, or other module required
a task-owned change.

## Implementation notes

- Added the four exact public composition types in `nn.module`. `Model.define` invokes one
  definition, seals its Topology in a <span data-glossary-navigation="true"> ([Topology glossary](../../../../glossary.md#term-topology))</span>`finally` boundary, rejects a null forward function, and
  publishes a private functional Model only after complete named-child ownership validation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added package-private final `Module.registerNamedChildren(Map<String, ? extends Module>)`. It
  copies encounter order, validates all nulls, names/collisions, repeated identities, cycles, and
  existing parent ownership before changing any child entry or parent link, then installs the
  exact snapshot in order.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Functional forward rejects null input, calls the retained function exactly once, rejects null
  output, returns the exact result reference, and catches no body failure. It performs no module
  lookup, reflection, graph work, initialization, compilation, training, or execution.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [lookup glossary](../../../../glossary.md#term-lookup); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Added one focused same-package suite with an in-memory external-package Java compilation probe.
  Existing Module, Sequential, layer, recurrent, state-dictionary, build, and dependency behavior
  otherwise remains unchanged.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Drafted affected public/package Javadocs for independent review. The clean documentation context
  then finalized them plus Training API, glossary, planning evidence, and documentation validation
  without rerunning the stable Java tests because it changed no executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented typed functional Model definition, sealed named Topology
  collection, complete validate-before-install named child ownership, exact functional forward
  behavior, descriptive inherited state/mode behavior, and focused coverage.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [sealed glossary](../../../../glossary.md#term-sealed); [Topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Files changed or created: `Module.java`, `Model.java`, `ModelDefinition.java`,
  `ModelForward.java`, `Topology.java`, [module](../../../../glossary.md#term-module) `package-info.java`, `ModelTest.java`, Training API,
  glossary, the NN master plan, and this task record: exactly the eleven authorized paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and validation: Focused Model one-suite/ten-test pass and authoritative NN 30-suite/207-
  test pass, both with zero failures, errors, or skips; final Javadoc/rendered-page, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`,
  external-use, import/mechanism, Markdown, scope, and whitespace checks passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Independent clean context `/root/nn_0018_docs` found no executable,
  API, architecture, or scope defect and finalized the authorized documentation.
- Documentation impact: Public/package Javadocs, Training API, glossary, and planning evidence now
  explain the typed boundary, atomic ownership, sealing, inherited lifecycle, failure effects, and
  planned boundaries without confusing NN topology with a Tensor or compiled graph.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [lifecycle glossary](../../../../glossary.md#lifecycle); [Tensor glossary](../../../../glossary.md#tensor); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- Javadoc review: Final generation and rendered-page inspection passed after the documentation
  edits.
- Glossary impact: Added the functional Model and Model-topology term, cross-linked it to the
  Training API, and corrected the current recurrent-container inventory.
- Unresolved issues: No executable or architecture issue found.
- Follow-up required: None for task 0018. NN 0019 remains Draft pending its explicit deferred-state
  lifecycle decisions.

Status: Complete
