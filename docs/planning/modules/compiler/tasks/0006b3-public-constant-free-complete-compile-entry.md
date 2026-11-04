# Task 0006B3: [Engine](../../../../glossary.md#term-engine)-Facing Complete [Compile](../../../../glossary.md#compile) [Integration Port](../../../../glossary.md#term-integration-port)

## Status

Complete

## Goal

Make the already implemented complete [Compiler](../../../../glossary.md#term-compiler) pipeline callable by the future [Engine](../../../../glossary.md#term-engine) [module](../../../../glossary.md#term-module)
through one narrow public-for-module-[integration port](../../../../glossary.md#term-integration-port), while keeping the implementation [owner](../../../../glossary.md#term-owner)
`GraphCompiler` and its current [compile](../../../../glossary.md#compile) entries package-private.

The current complete `GraphCompiler.compile(...)` returns public [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts`, but
a class in another Java package cannot call it because `GraphCompiler`, the method, and its
`CompileTimeConstantGraph.Ingress` [parameter](../../../../glossary.md#term-parameter) are package-private. A public bridge placed in the
same Compiler package can legally supply empty [constant](../../../../glossary.md#term-constant) ingress and delegate without exposing
those implementation types or changing architecture.

## [Scope](../../../../glossary.md#term-scope)

- Add public final, non-instantiable
  `io.github.pho001.synaptik.compiler.GraphCompilationPort` as a cross-[module](../../../../glossary.md#term-module) integration [SPI](../../../../glossary.md#term-service-provider-interface-spi).
- Add exactly one public static `compile(...)` method with this logical [parameter](../../../../glossary.md#term-parameter) order:
  [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), ordered forward [`Tensor`](../../../../glossary.md#tensor) outputs, optional `FunctionalGradientRequest`,
  [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig), ordered
  [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) values, and ordered [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) values.
- Return the existing public [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts` directly.
- Delegate exactly once to the existing nine-[argument](../../../../glossary.md#term-argument) complete `GraphCompiler.compile(...)` with
  `CompileTimeConstantGraph.Ingress.empty()`.
- Keep `GraphCompiler`, its constructor, its graph-stage entry, and its complete entry
  package-private with their existing signatures and behavior.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Preserve declaration-order validation, exception identity and wrapping, provider/snapshot
  encounter order, graph semantics, optimization, publication, planning, logical memory,
  diagnostics, and derivative behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Add a focused same-package delegation/shape test and a distinct-package source-level test proving
  that an integration caller can invoke the port using only public types.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Document the port as cross-module SPI, not as the recommended Engine end-user API, through the
  mandatory separate documentation-focused pass.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Making `GraphCompiler` or either current [compile](../../../../glossary.md#compile) method public.
- A public explicit compile-time logical-splat ingress; current [constant](../../../../glossary.md#term-constant) machinery stays private.
- `CompileConfig`, `PrepareConfig`, `RunOptions`, defaults, builders, or other Config work.
- `CompiledGraph`, Engine construction, standard built-in composition, backend registration,
  Engine exceptions, user-facade signatures, or any Engine code.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [facade glossary](../../../../glossary.md#term-facade); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Prepare orchestration, prepared execution, Runtime execution, typed input binding, result access,
  host materialization, one-shot execution, backward convenience, or tuning integration.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Tensor](../../../../glossary.md#tensor) `execute`/`backward` methods, mutable gradients, hidden autograd state, or Model/Runtime
  dependencies.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [Model glossary](../../../../glossary.md#term-model))</span>
- Compiler graph transformations, new gradient behavior, recurrent BPTT, Conv3d gradients,
  exact/relaxed algebra changes, vendor CPU routes, or BFLOAT16 OpenBLAS work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [CPU glossary](../../../../glossary.md#term-cpu); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Changes to Config, Planning, Prepare, Runtime, Engine, Backend Contract, Model, CPU, Gradle,
  architecture contracts/tests, backend conformance, or integration tests.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Compiler,
  Compile artifacts, Engine, dependency rules, and compile lifecycle.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Compiler glossary](../../../../glossary.md#term-compiler); [Compile artifacts glossary](../../../../glossary.md#compile-artifacts); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile))</span>
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/planning/modules/compiler/master-plan.md`](../master-plan.md)
- [`docs/planning/modules/engine/master-plan.md`](../../engine/master-plan.md)

## Architecture constraints

- Compiler remains the owner of capture, validation, transformation, autograd, publication,
  planning orchestration, logical memory, diagnostics, and immutable compile artifacts.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- [Compiler](../../../../glossary.md#term-compiler) must not depend on [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), or a [concrete backend](../../../../glossary.md#concrete-backend).
- The new public type is a Compiler-owned integration port required by Java accessibility. It is
  technically callable by any module but must be documented as SPI and must not become an
  ordinary Engine user-facade parameter or return type.<span data-glossary-navigation="true"> ([integration port glossary](../../../../glossary.md#term-integration-port); [module glossary](../../../../glossary.md#term-module); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [facade glossary](../../../../glossary.md#term-facade); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- The port accepts only current public inward-layer contracts and retains no live provider or
  availability snapshot in <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`CompileArtifacts`.
- Compile creates no physical storage, prepared schedule, executable, runtime state, route,
  kernel, backend lowering, backend discovery, or tuning decision.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule); [kernel glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- Engine remains the future composition root. This task performs no standard or advanced backend
  composition and creates no service locator or process-global state.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [backend composition glossary](../../../../glossary.md#backend-composition); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- No [`Operation`](../../../../glossary.md#operation) or `CompiledNode` crosses into Runtime, and this task changes no Runtime surface.
- With no explicit constant ingress, every reachable provenance-free forward leaf remains
  caller-bindable. Compiler-generated request constants continue through existing internals.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- If the bridge cannot delegate without changing compile semantics, exposing another private type,
  or adding an architecture rule, stop and report the concrete gap.

## Package impact

Existing production package:

- `io.github.pho001.synaptik.compiler` owns the package-private implementation, public artifacts,
  public functional requests, and the new narrow module-integration port. Co-location is required
  because Java package access lets the port call <span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [module glossary](../../../../glossary.md#term-module); [integration port glossary](../../../../glossary.md#term-integration-port))</span>`GraphCompiler` without [widening](../../../../glossary.md#term-widening) it.
- The port uses existing public types from `io.github.pho001.synaptik.config.compile`,
  `io.github.pho001.synaptik.model.tensor`,
  `io.github.pho001.synaptik.planning.capability`, and
  `io.github.pho001.synaptik.backend.contract`; it adds no package or [dependency edge](../../../../glossary.md#term-dependency-edge).

Test packages:

- `io.github.pho001.synaptik.compiler` verifies direct delegation and package-private preservation.
- `io.github.pho001.synaptik.compiler.spi` verifies cross-package source accessibility using only
  public types; no production `spi` subpackage is added.

No [Engine](../../../../glossary.md#term-engine), [API](../../../../glossary.md#term-application-programming-interface-api) [facade](../../../../glossary.md#term-facade), or additional production package is introduced.

## Affected files

Expected production and test files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilationPortTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/spi/GraphCompilationPortPublicShapeTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files finalized in the same overall implementation change:

- `docs/api/compile-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`, limited to current [Compile](../../../../glossary.md#compile) and Compile-[artifacts](../../../../glossary.md#term-artifact) status statements
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md`
- `docs/planning/roadmap.md`

`GraphCompiler.java` and its existing tests are expected to remain unchanged. No other file is
expected to change. The documentation pass must record a reasoned no-change conclusion for
architecture pages, Training [API](../../../../glossary.md#term-application-programming-interface-api), Config/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu)/tuning plans, [Gradle](../../../../glossary.md#term-gradle) files,
architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests unless implementation exposes a
concrete contradiction.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one new [Compiler](../../../../glossary.md#term-compiler) production Java file;
- two new Compiler test Java files;
- two explanatory [API](../../../../glossary.md#term-application-programming-interface-api) Markdown files and one glossary Markdown file; and
- three [planning](../../../../glossary.md#term-planning) Markdown files, including this task specification;
- nine paths total.

If `GraphCompiler.java`, another production type/package/[module](../../../../glossary.md#term-module), a [dependency](../../../../glossary.md#term-dependency)/build file, or a
tenth path is required, stop and propose the smallest follow-up or architecture decision.

## Acceptance criteria

- `GraphCompilationPort` is public, final, non-instantiable, and contains exactly one public static
  [`compile`](../../../../glossary.md#compile) method with the eight inputs listed in [Scope](../../../../glossary.md#term-scope) and `CompileArtifacts` return type.
- `GraphCompiler` remains package-private and retains exactly its current two package-private
  compile entries and signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- The port performs no graph or plan work itself and delegates exactly once to the existing
  complete entry with empty explicit constant ingress.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant))</span>
- Forward-only port compilation is artifact-equivalent to direct package-private complete
  compilation with the same inputs and empty ingress, including publications, ownership,
  partitions, logical memory, constants, diagnostics, and derivative metadata.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [artifact glossary](../../../../glossary.md#term-artifact); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Backward-capable requests preserve the existing explicit request and mode contract; no new
  autograd behavior or inferred target/seed policy is introduced.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed))</span>
- Null, invalid request/mode, no-eligible-backend, and provider-failure behavior preserves current
  declaration-order behavior and failure identity, with no Engine exception translation.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [Engine glossary](../../../../glossary.md#term-engine))</span>
- The distinct-package test compiles and invokes the port without reflection, private imports, or
  Engine dependencies.
- Same-package reflection/source checks prove `GraphCompiler` and both of its entries remain
  package-private.
- Public Javadoc identifies module-integration status and documents constant-free leaf treatment,
  parameter constraints/order, result semantics/nullability, and expected failure classes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module); [leaf glossary](../../../../glossary.md#term-leaf); [parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Public documentation keeps the port out of the recommended end-user surface and does not claim
  an operational Engine lifecycle.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- No production/test/build file outside `modules/compiler` changes, and no [Gradle](../../../../glossary.md#term-gradle) file changes.
- A separate documentation-focused clean context finalizes affected Javadoc, API/planning text,
  and glossary impact before completion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused checks:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilationPortTest --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest
./gradlew :modules:compiler:test
```

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
git diff --check
```

Manual validation:

- inspect `javap -public` for `GraphCompilationPort` and confirm exactly one public compile method
  with only public parameter and return types;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- inspect declared modifiers for `GraphCompiler` and both current compile methods and confirm they
  remain package-private;
- inspect delegation to confirm one call with `CompileTimeConstantGraph.Ingress.empty()` and no
  other behavior;
- [scan](../../../../glossary.md#term-cumulative-scan) [Compiler](../../../../glossary.md#term-compiler) production imports and [Gradle](../../../../glossary.md#term-gradle) files for forbidden outward dependencies;
- validate changed Markdown links, anchors, headings, fences, trailing whitespace, and newline at
  end of file;<span data-glossary-navigation="true"> ([end of file glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- confirm the implementation stays within nine paths and changes no code outside Compiler.

Repository-wide validation is deferred to the [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci)
because this task adds one [module](../../../../glossary.md#term-module)-local integration entry without changing dependencies,
architecture rules, shared build configuration, or executable semantics.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0005 complete [compile artifacts](../../../../glossary.md#compile-artifacts) and package-private complete entry — Complete.
- Compiler 0006B2 current validated [compile](../../../../glossary.md#compile) pipeline — Complete.
- Config 0001–0003 standalone [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), and
  [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) contracts — Complete.
- Current public Planning capability provider and Backend Contract availability snapshot inputs —
  Complete.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Engine-frontier source inspection proving the Java accessibility blocker and same-package bridge
  feasibility — Complete as planning evidence.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Follow-up tasks

- CPU 0010F, required next: add the smallest supported CPU lifecycle integration adapter and
  production schedule assembly without leaking CPU internals.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Engine 0001, required after CPU 0010F: establish advanced explicit composition and the first
  representation-level lifecycle seam.
- Engine 0002–0004, ordered: deterministic standard built-in composition ordinary users can
  construct without naming CPU, then typed logical binding/result access and host materialization.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- Engine 0005–0006, ordered: Engine-owned one-shot forward and scalar-objective backward
  convenience after typed binding/result/materialization exist.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- [Engine](../../../../glossary.md#term-engine) 0007 and tools/tuning 0002, later optional integration; neither blocks this task.
- Config aggregate/default work, Compiler 0006C, and Compiler 0007 remain independent deferred
  work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Architecture impact

Expected impact: None.

This task realizes an existing [Compiler](../../../../glossary.md#term-compiler)-to-[Engine](../../../../glossary.md#term-engine) accessibility boundary with a same-package
public [SPI](../../../../glossary.md#term-service-provider-interface-spi). It changes no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) stage, explicit
[backend](../../../../glossary.md#backend)-registration rule, or user-facing Engine [API](../../../../glossary.md#term-application-programming-interface-api). If implementation requires any such change,
stop and report it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository on Compiler task 0006B3.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/compiler/master-plan.md, and
docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md in full.

Implement task 0006B3 exactly as specified. Add the narrow Compiler-owned
GraphCompilationPort integration SPI in the existing Compiler package; keep GraphCompiler and
both current compile entries package-private. Do not implement any Engine facade, backend
composition, Config aggregate, typed execution, or other out-of-scope work. Stop and report if
the port cannot be added within the recorded architecture and maximum scope. Do not commit or push.

After implementation and Compiler validation, hand the diff and evidence to a separate clean
documentation-focused context. That pass must follow docs/developer-guide/documentation-rules.md,
finalize affected Javadoc/API/planning text and glossary impact in the same overall change, reuse
successful Java evidence unless executable code changes, and complete the specified documentation
validation.

Update this task with implementation notes, exact validation evidence, documentation-pass identity
and result, completion summary, and final status. Do not mark it Complete before every acceptance
criterion and the documentation pass succeed.
```

## Local decisions

- `GraphCompilationPort` remains in the Compiler root package because that package owns the
  complete compile pipeline and Java package access permits delegation without widening<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [compile glossary](../../../../glossary.md#compile); [widening glossary](../../../../glossary.md#term-widening))</span>
  `GraphCompiler` or its explicit [constant](../../../../glossary.md#term-constant)-ingress type.
- The port exposes exactly the existing complete result and supplies empty explicit forward
  constant ingress. It adds no overload, request aggregate, defaults, wrapper artifact, or
  exception translation.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Documentation applies the General Style, API/Javadoc, Planning, and Example profiles. The port
  is described as cross-module SPI for future lifecycle composition, never as the recommended
  ordinary-user API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [module glossary](../../../../glossary.md#term-module); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Known limitations

- The port cannot declare explicit forward constants; every reachable provenance-free forward
  leaf remains caller-bindable. This is the task's intentional constant-free boundary.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- No production capability-provider composition, Engine facade, preparation, execution, typed
  input binding, publication delivery, or failure translation exists yet. CPU 0010F is the next
  Draft operational-lifecycle frontier.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a09f06-c837-79a0-b9f2-8e8a62a9ae01` supplied the following successful
evidence, reused by this documentation context because no executable Java changed afterward:

- `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilationPortTest --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest`
  passed all five new tests with zero failures, errors, or skips.
- `./gradlew :modules:compiler:test` passed 246 tests with zero failures, errors, or skips.
- `javap` confirmed that `GraphCompilationPort` is public and final with exactly one public static
  eight-input [`compile`](../../../../glossary.md#compile) method whose [parameter](../../../../glossary.md#term-parameter) and return types are public.
- `javap -p` confirmed that `GraphCompiler` and both existing `compile` entries remain
  package-private and retain their signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- [Bytecode](../../../../glossary.md#term-bytecode) inspection confirmed exactly one `CompileTimeConstantGraph.Ingress.empty()` call and
  one `GraphCompiler.compile(...)` call, with no other port behavior.
- The forbidden outward-import [scan](../../../../glossary.md#term-cumulative-scan) and `git diff --check` passed in the implementation context.

Documentation context `01a09f0c-b9cc-79a0-8881-4d48818daa6e` independently reviewed the final
source, both new tests, the delegate/privacy implementation, architecture and documentation
contracts, [Compiler](../../../../glossary.md#term-compiler) [planning](../../../../glossary.md#term-planning) state, affected [API](../../../../glossary.md#term-application-programming-interface-api) references, Training API, and glossary. It
selected General Style, API/[Javadoc](../../../../glossary.md#term-javadoc), Planning, and Example profiles and recorded these results:

- The prior `./gradlew :modules:compiler:javadoc` run completed successfully with seven actionable
  tasks: two executed and five up-to-date. Generated <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`GraphCompilationPort` Javadoc contains the
  finalized SPI, empty-ingress, caller-bindable-leaf, result, and failure contracts. The glossary
  correction changed no Java or Javadoc, so this successful evidence was reused without rerun.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- `python3 /tmp/validate_synaptik_markdown.py docs/api/compile-api.md docs/api/public-api.md docs/glossary.md docs/planning/modules/compiler/master-plan.md docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md docs/planning/roadmap.md`
  validated all six changed Markdown files, including local links and anchors, headings, fences,
  line endings, trailing whitespace, and final newlines.
- Direct source/test/diff review confirmed exact delegation, public-only cross-package
  accessibility, artifact equivalence coverage, backward-request preservation, provider-failure
  identity, declaration-order validation, and package-private implementation ownership.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- API text scans found no remaining claim in the changed Compile/Public API pages that a public
  compile call is absent or that complete compilation is only package-private. They also confirm
  that the port is not presented as an Engine facade and that no Engine prepare/run support is
  claimed.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [facade glossary](../../../../glossary.md#term-facade))</span>
- `docs/api/training-api.md` remains unchanged because its package-private statement describes the
  reverse-mode implementation path, not accessibility of the complete constant-free delegate.<span data-glossary-navigation="true"> ([reverse-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed); [constant glossary](../../../../glossary.md#term-constant))</span>
- `GraphCompiler.java` and its Javadoc remain unchanged and accurate: the implementation owner and
  both compile entries are still package-private, while the new public type delegates to them.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `docs/glossary.md` required no new term. Its existing Compile and Compile-artifacts entries were
  corrected narrowly because their claims that public compilation remained wholly planned and no
  public entry produced `CompileArtifacts` became false when `GraphCompilationPort` was added.
  The revised entries retain the constant-free SPI and absent ordinary-user Engine boundaries.
- Architecture contracts/pages, Config, Prepare, Runtime, Engine, CPU, and tuning plans, Gradle
  files, architecture tests, backend conformance, integration tests, and all other modules remain
  unchanged by this documentation context because the port realizes an already planned boundary
  without changing ownership, dependencies, or executable behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- Final scope accounting contains exactly nine task-owned changed paths: one production Java
  file, two test Java files, two API documents, one glossary document, and exactly three planning
  documents. The six pre-existing coordinator planning paths outside this task remain
  byte-for-byte unchanged by this context. Their final SHA-256 values remain<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
  `ff207648733f8c6f121536d5a37790e755563958681147749fe702b6ee366cf1` ([CPU](../../../../glossary.md#term-cpu)),
  `c233952dea792252fd54db18ee780a20d7787bc339d44d5ce9d74f31fbf75cd2` (Config),
  `ce8e5de4e460e6dc623d73abdb47667232e1dad26e79c39306f4b414260a7cb7` ([Engine](../../../../glossary.md#term-engine)),
  `39a20aebc4baa96a5b326e8016e040c9053d40e7e493fbe91f8894d88a8f0d40` ([Prepare](../../../../glossary.md#prepare)),
  `9c8507d2c56bcd0b9679bff612eb7e5bb24b652a00fb8232d2acabd0a2a89c3b` ([Runtime](../../../../glossary.md#term-runtime)), and
  `c856a255b5780fb5ac2b6a6d2cca2e2bf421981cbf84e532ae9bc82bfcb12d25` (tuning). The total
  worktree contains those six protected paths plus the nine task-owned paths, and the Git index
  remains empty.
- `git diff --check`, final glossary/API/status scans, task-path and total-worktree accounting,
  Ready-state uniqueness, protected-path hashes, and direct final diff review passed after the
  completion update. Compiler 0006B3 is Complete; CPU 0010F is the next Draft frontier without a
  detailed task and was not marked Ready.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Implementation notes

The implementation added one non-instantiable public final class with one public static method.
The method delegates directly to the existing complete [compiler](../../../../glossary.md#term-compiler) overload and supplies
`CompileTimeConstantGraph.Ingress.empty()`. It does not copy inputs, catch or translate failures,
retain collaborators, or perform [compile](../../../../glossary.md#compile) work itself.

The independent documentation pass refined every public [parameter](../../../../glossary.md#term-parameter), result, [nullability](../../../../glossary.md#term-nullability),
ordering, empty-ingress, and caller-visible failure statement. It added one bounded current
zero-[node](../../../../glossary.md#node) example to the Compile [API](../../../../glossary.md#term-application-programming-interface-api), synchronized the Public API inventory, corrected the two
stale glossary status statements, and kept the port out of the recommended ordinary-user
[lifecycle](../../../../glossary.md#lifecycle). No Java or [Javadoc](../../../../glossary.md#term-javadoc) changed during the glossary correction, so neither Java tests nor
Javadoc generation were repeated; their prior successful evidence was reused.

## Completion summary

- Completed changes: added and documented the narrow public constant-free Compiler integration
  port, preserved package-private compiler implementation entries and behavior, and synchronized
  Compiler task/master-plan/roadmap status with CPU 0010F as the next Draft frontier.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [Compiler glossary](../../../../glossary.md#term-compiler); [integration port glossary](../../../../glossary.md#term-integration-port); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Files changed or created: one Compiler production source, two Compiler tests, two API reference
  pages, one glossary page, and exactly three Compiler/roadmap planning documents; nine task-owned
  paths total.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the implementation context's five focused tests, 246-test Compiler
  suite, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`javap`, bytecode, import, and whitespace evidence plus the prior successful Compiler
  Javadoc generation; validated Markdown links, anchors, headings, fences, whitespace, line
  endings, final newlines, glossary/API wording, scope, protected paths, status uniqueness, and
  the final diff.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean documentation context
  `01a09f0c-b9cc-79a0-8881-4d48818daa6e` completed the independent review using the General Style,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: `compile-api.md`, `public-api.md`, and the existing glossary Compile and
  Compile-artifacts entries now distinguish the callable cross-module SPI from the absent
  ordinary-user Engine lifecycle; <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [artifacts glossary](../../../../glossary.md#term-artifact); [module glossary](../../../../glossary.md#term-module); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`training-api.md` remains accurate without change.
- Javadoc review: finalized SPI status, constant-free leaf treatment, all parameters, result
  semantics/nullability, ordering, and expected failure classes/identity.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Glossary impact: corrected two stale status statements without adding a new term.
- Unresolved issues: None.
- Follow-up required: None for this task. CPU 0010F remains the next Draft operational frontier;
  its detailed task is intentionally not created here.

Status: Complete
