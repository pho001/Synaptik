# Task 0006B8: Bounded functional autograd contract and documentation reconciliation

## Status

Ready

## Change class

Class C — this documentation-only repair changes the authoritative Compiler-owned automatic-
differentiation contract and its user-visible explanation. It changes no Java behavior, API,
dependency, module ownership, or execution path, but the normative boundary and workflow require
a clean documentation implementation context and an independent clean review context.

## Goal

Make the authoritative autograd contract, focused architecture explanation, user guide, and
targeted glossary definition describe the already implemented bounded functional reverse-mode
behavior exactly, while preserving the narrower ordinary `Engine.backward(...)` convenience.

## Scope

- Replace the obsolete future-only higher-derivative paragraph in the Compiler autograd contract
  with the current bounded one/two-stage lifecycle.
- Reconcile `training-graph.md` with current compile consumption, publication, derivative-order
  metadata, and one combined capture.
- Rewrite the stale current-status and workflow portions of `user-guide/autograd.md` around the
  supported public ordinary and advanced entry points, with a current example and limitations.
- Correct only the stale `Pre-capture Tensor-expression autograd` glossary definition; keep related entries aligned.
- Synchronize this task, the Compiler master plan, and the roadmap when execution completes.

The exact current boundary is:

- a functional request contains exactly one or two stages with non-empty ordered outputs and
  output-aligned optional seeds; target lists are non-empty, ordered, identity-unique, and drawn
  from the complete original forward inventory;
- an absent seed is an exact typed one only for an eligible scalar output; a present seed must
  match its output's exact Shape and floating type and must not request gradients;
- `DisconnectedPolicy.ERROR` rejects a disconnected target, while `ZERO` returns an ordinary
  exact typed zero expression that several target roles may share;
- one stage requires stage-one `createGraph == false`; two stages require stage-one
  `createGraph == true` and stage-two `createGraph == false`;
- stage one selects forward Tensor outputs, while stage two may select only generated first-stage
  gradients through `FirstStageGradientReference(targetIndex)`;
- `createGraph` retains formulas only for the immediate second stage; it does not create an
  arbitrary persistent derivative chain;
- derivative metadata uses only orders zero, one, and two, while gradient publication bindings
  use only orders one and two; bindings are ordered by derivative order then stage-local target
  index, retain target identity and the final gradient value, and remain target-distinct when
  graph-output values are shared;
- there is no third stage, derivative order greater than two, arbitrary nesting, persistent or
  runtime tape, mutable Tensor gradient state, or `Tensor.backward()`;
- ordinary `Engine.backward(...)` always constructs one first-order stage with an absent scalar
  seed, `createGraph == false`, and `DisconnectedPolicy.ERROR`; the complete bounded request
  surface is available through the advanced compile boundary.

## Non-goals

- No Java, test, Javadoc, Gradle, dependency, API-shape, semantic, numerical, publication,
  preparation, runtime, backend, optimizer, or training-session behavior changes.
- No third derivative stage, general nested differentiation, persistent graph/tape lifecycle,
  Tensor gradient field, new compile mode, or broadened derivative coverage.
- No rewrite of the operation-family coverage matrix except removing claims made stale by current
  source; link to the maintained Compile API where a large inventory would duplicate it.
- No root-contract, architecture decision record (ADR), architecture-test, conformance-test, or
  integration-test change: ownership and dependency rules do not change, and
  `ARCHITECTURE.md` already routes this concern correctly and prohibits Tensor gradient state.
- No edits to the current Compile, Public, Tensor, or Training API references. If source evidence
  contradicts one of those references, stop and report the required scope expansion.

## Contracts

- [`ARCHITECTURE.md` heading `Authority, incorporation, and precedence`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  and [`Scope-indexed normative contracts`](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — the incorporated Compiler contract is the sole scoped authority for this concern.
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants)
  and [`Explicit non-goals`](../../../../../ARCHITECTURE.md#explicit-non-goals) — Tensor has no
  gradient lifecycle state and the repair must not add one.
- [`compiler-autograd.md` heading `Compiler-owned automatic differentiation`](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — reconcile the normative compile-time flow and bounded derivative lifecycle with current
  source and tests.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these seven paths:

- `docs/architecture/contracts/compiler-autograd.md` — `Compiler-owned automatic differentiation`.
- `docs/architecture/training-graph.md` — current combined graph and bounded derivative-order flow.
- `docs/user-guide/autograd.md` — current ordinary/advanced user workflow, example, errors, and
  limitations.
- `docs/glossary.md` — only the targeted autograd definition and any directly necessary links to
  already-current related entries.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/compiler/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — repository frontier only.

Read-only source and focused evidence:

- `FunctionalGradientRequest`, `GraphCompiler`, `DerivativeGraphMetadata`, and
  `GradientPublicationBinding` in `modules/compiler`.
- `Engine.backward(...)` and `AdvancedEngine.compile(...)` in `modules/engine`.
- `FunctionalGradientRequestTest`, the two-stage and Hessian-vector-product cases in
  `GraphCompilerTest`, and the ordinary backward cases in `EngineTypedLifecycleTest`.
- `docs/api/compile-api.md` headings `Current ordinary and advanced Engine compile boundaries`,
  `Current package-private pre-capture autograd`, and `Functional-gradient examples`.
- Current relevant text in `docs/api/public-api.md`, `docs/api/tensor-api.md`, and
  `docs/api/training-api.md`; these are comparison references, not edit targets.

## Acceptance criteria

- Together, the four editable product documents state the exact one/two-stage, `createGraph`,
  output-reference, derivative-order, and immediate-next-stage boundaries above without
  contradiction or implying a third stage or persistent lifecycle. Explanatory documents do not
  present themselves as authority or duplicate the contract unnecessarily.
- The authoritative contract distinguishes derivative order from `GraphPhase`, preserves one
  combined capture, and retains the existing prohibitions on Tensor state, a runtime tape, a
  second gradient algebra, model-owned rules, and backend-owned global autograd.
- The architecture explanation no longer says compile modes or autograd are unconsumed or that
  higher derivatives are absent; it distinguishes current bounded second-stage construction from
  still-unsupported arbitrary nesting and optimizer/training-session work.
- The user guide no longer claims that objective/target requests, seeds, disconnected-zero
  policy, gradient publication, public invocation, or bounded second-stage construction are
  absent. It clearly separates ordinary `Engine.backward(...)`, ordinary explicit-seed compile,
  and advanced functional requests.
- The user guide applies General, User-guide, and Example profiles. Any public example is current
  and runnable for the scope it claims, or is explicitly labeled conceptual; it states inputs,
  meaningful steps, result metadata, interpretation, and an important failure or limitation.
- The targeted glossary definition agrees with the already-current `Functional gradient
  request`, `Gradient publication binding`, `Derivative graph metadata`, and one-shot backward
  definitions. Targeted search records whether any additional glossary change is needed.
- The current API references remain unchanged after comparison with source and tests. The result
  records reasoned no-change conclusions for Java/Javadoc, root architecture, ADRs, architecture
  tests, Compile/Public/Tensor/Training API references, other modules, conformance/integration,
  and Gradle.
- Only the seven allowed paths change; task/master/roadmap statuses remain synchronized; local
  links and anchors resolve, headings are unique, fences are balanced, files have final newlines
  and no trailing whitespace, and `git diff --check` passes.

## Validation

The documentation implementation context runs the existing focused evidence once:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.FunctionalGradientRequestTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest
```

After final documentation and planning edits:

```bash
python3 /tmp/validate_synaptik_markdown.py \
  docs/architecture/contracts/compiler-autograd.md docs/architecture/training-graph.md \
  docs/user-guide/autograd.md docs/glossary.md \
  docs/planning/modules/compiler/tasks/0006b8-bounded-functional-autograd-contract-and-documentation-reconciliation.md \
  docs/planning/modules/compiler/master-plan.md docs/planning/roadmap.md
rg -ni "deliberately absent from the first implementation|current compile aggregate or compiler entry point|no public compile request for an objective|adds no request aggregate" \
  docs/architecture/contracts/compiler-autograd.md docs/architecture/training-graph.md \
  docs/user-guide/autograd.md docs/glossary.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The first `rg` and executable/build-file diff commands must produce no output. The path inventory
must contain exactly the seven allowed paths. If the temporary Markdown validator is absent,
create an equivalent validator outside the repository for changed-file local links and anchors,
unique headings, balanced fences, final newlines, and trailing whitespace.

Repository-wide validation is deferred to CI because this task changes no executable code,
dependency, build configuration, or cross-module behavior. The independent review context reuses
successful Java evidence and does not rerun it unless executable behavior changed or evidence is
missing or stale.

## Dependencies and frontier verification

- Compiler 0006's bounded functional request, publication, and derivative-order behavior is
  Complete and is the source behavior being documented.
- Engine 0011 is Complete. The user explicitly authorized this reconciliation as the second
  drift-remediation step after Engine 0011.
- This is an explicit repository-order interleave bounded to the seven paths above and closed by
  synchronized task/master/roadmap status after independent review. Compiler 0006C and 0007
  remain Draft side branches and gain no implementation authorization from this task.

## Documentation and review impact

- The implementation context follows the documentation rules and the General, Architecture,
  User-guide, Planning, and Example profiles, compares every claim with source and focused tests,
  and changes no Java or Javadoc.
- A separate clean independent review context is mandatory because this is Class C, changes an
  incorporated authoritative contract, and changes a supported user workflow explanation. It
  inspects the final diff and evidence, finalizes terminology/examples/links/glossary impact, and
  records its own completion summary before the task may become Complete.

## Result

Empty until execution.
