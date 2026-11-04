# Task 0006B10: [Compile](../../../../glossary.md#compile) integration status reconciliation

## Status

Complete

## Change class

Class B — this documentation-only task corrects a public [Compiler](../../../../glossary.md#term-compiler) integration type and the shared
[compile](../../../../glossary.md#compile) term. It changes no [API](../../../../glossary.md#term-application-programming-interface-api) or executable behavior, but requires a clean implementation context
and independent documentation review because the stale wording misstates the current public [Engine](../../../../glossary.md#term-engine)
[lifecycle](../../../../glossary.md#lifecycle).

## Goal

Reconcile `GraphCompilationPort` [Javadoc](../../../../glossary.md#term-javadoc) and the glossary [`Compile`](../../../../glossary.md#compile) entry with the current [Engine](../../../../glossary.md#term-engine)
consumer while preserving the port's narrow cross-[module](../../../../glossary.md#term-module) [SPI](../../../../glossary.md#term-service-provider-interface-spi) role and all current compile semantics.

## [Scope](../../../../glossary.md#term-scope)

- Replace the future-[Engine](../../../../glossary.md#term-engine) wording in `GraphCompilationPort` type Javadoc with the current role:
  lifecycle-composition SPI used by Engine, not an ordinary application-facing standalone compiler.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- Replace the stale glossary sentence that says the Engine [compile](../../../../glossary.md#compile) [lifecycle](../../../../glossary.md#lifecycle) remains planned.
- Keep preparation and execution outside `GraphCompilationPort`; identify current public Engine
  compile/prepare/run composition without implying generic provider registration.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- Synchronize this task, the [Compiler](../../../../glossary.md#term-compiler) master plan, and the roadmap after independent review.

## Non-goals

- No Java [signature](../../../../glossary.md#term-signature), implementation, test, [Gradle](../../../../glossary.md#term-gradle), behavior, [dependency](../../../../glossary.md#term-dependency), or architecture change.
- No edit to `ARCHITECTURE.md`, scoped contracts, ADRs, [API](../../../../glossary.md#term-application-programming-interface-api)/[backend](../../../../glossary.md#backend) guides, or other glossary entries.
- No claim that `GraphCompiler` is public, that applications should call the SPI directly, or that
  Engine supports arbitrary backend/provider registration.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No repair of the separate [`Backend partition finalization`](../../../../glossary.md#backend-partition-finalization) glossary status paragraph.

## Contracts

- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  compile creates immutable backend-neutral artifacts and no physical buffers.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [artifacts glossary](../../../../glossary.md#term-artifact); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- [`compiler-autograd.md` heading `Scope`](../../../../architecture/contracts/compiler-autograd.md#scope)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
  — Compiler owns graph compilation and immutable compile artifacts, not preparation or execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — current Engine owns public lifecycle composition and fixed/advanced CPU-only integration.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [CPU glossary](../../../../glossary.md#term-cpu))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java` — type
  Javadoc only.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `docs/glossary.md` — heading [`Compile`](../../../../glossary.md#compile) only.
- `docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md` —
  status and compact result evidence.
- `docs/planning/modules/compiler/master-plan.md` — task row and frontier summary only.
- `docs/planning/roadmap.md` — [Compiler](../../../../glossary.md#term-compiler) area, frontier, discovered-drift list, and next step only.

Read-only evidence:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/spi/GraphCompilationPortPublicShapeTest.java`.
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`.
- `docs/api/compile-api.md` focused `GraphCompilationPort` and `AdvancedEngine.compile(...)` sections.
- `docs/api/public-api.md` focused Compiler and [Engine](../../../../glossary.md#term-engine) status sections.

No path outside the five-file implementation allowlist may change.

## Acceptance criteria

- Type [Javadoc](../../../../glossary.md#term-javadoc) states that `GraphCompilationPort` is the current narrow integration SPI used by
  Engine and other lifecycle-composition code, while ordinary application code uses Engine.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Type Javadoc still states that the port itself provides no Engine lifecycle, preparation, or
  execution.
- The glossary says current Engine composes compile, publication delivery, preparation, and
  execution, without claiming that those responsibilities belong to Compiler or the port.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [publication glossary](../../../../glossary.md#term-publication); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Package-private `GraphCompiler`, empty explicit constant ingress, compile artifacts, and all
  executable semantics remain unchanged.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- [API](../../../../glossary.md#term-application-programming-interface-api) guides and public API status require no edit because they already describe the current role.
- Independent Class B documentation review completes and records reasoned no-change conclusions
  for architecture, ADRs, other guides/glossary entries, tests, Gradle, and executable behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Only the five allowed paths change; brief stays at most 200 lines and 15 KB; Javadoc succeeds;
  Markdown links/anchors, headings, fences, whitespace, and final newlines validate; and
  `git diff --check` passes.

## [Validation](../../../../glossary.md#term-validation)

Implementation context:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/glossary.md \
  docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md \
  docs/planning/modules/compiler/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The path audit must equal the five-file allowlist. If the temporary Markdown validator is absent,
create an equivalent validator outside the repository. Review reuses successful executable
evidence unless it changes Java and reruns final documentation, [scope](../../../../glossary.md#term-scope), and whitespace checks.

Repository-wide validation is deferred because this task changes only status wording and type
[Javadoc](../../../../glossary.md#term-javadoc), with no executable, [dependency](../../../../glossary.md#term-dependency), build, or architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-authorized after clean HEAD `e25b26d6`; current source and API guides
  already agree, so 0006B10 is the sole authorized <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`Ready`](../../../../glossary.md#term-ready) frontier.
- Follow-up: separately reconcile the stale [`Backend partition finalization`](../../../../glossary.md#backend-partition-finalization) glossary paragraph.
- [Compiler](../../../../glossary.md#term-compiler) 0006C/0007 and every unrelated Draft, blocked, or review-needed item remain unauthorized.

## Documentation and review impact

- Types: public SPI Javadoc, glossary terminology/status, and planning status; apply General,
  API/Javadoc, and Planning profiles, with the General profile's glossary-alignment rules.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- A clean implementation context and independent targeted documentation review are required.

## Result

Implementation corrected the public [SPI](../../../../glossary.md#term-service-provider-interface-spi) [Javadoc](../../../../glossary.md#term-javadoc) and shared [`Compile`](../../../../glossary.md#compile) glossary entry against the
current fixed [CPU](../../../../glossary.md#term-cpu)-only [Engine](../../../../glossary.md#term-engine) composition. Independent Class B review confirmed that Engine uses
`GraphCompilationPort`, ordinary applications use Engine, and fixed ordinary plus explicitly
owned advanced composition remain CPU-only. Review clarified that [publication](../../../../glossary.md#term-publication) delivery belongs
to the ordinary [lifecycle](../../../../glossary.md#lifecycle) and corrected the read-only `AdvancedEngine` source path in this brief.

The focused public-[shape](../../../../glossary.md#shape) test and [Compiler](../../../../glossary.md#term-compiler) Javadoc succeeded in the implementation context and
were reused because review changed no executable Java or final Javadoc. Final Markdown, brief-size,
exact five-path, and whitespace checks passed. The [API](../../../../glossary.md#term-application-programming-interface-api) guides require no edit because they already
state the current boundaries. Architecture and ADRs require no edit because [ownership](../../../../glossary.md#term-ownership-meanings) and [module](../../../../glossary.md#term-module)
edges did not change; other guides/glossary entries, tests, [Gradle](../../../../glossary.md#term-gradle), and executable behavior require
no edit because this task only reconciles current documentation. The separately stale [backend](../../../../glossary.md#backend)-
[partition](../../../../glossary.md#partition)-finalization glossary status remains follow-up work.

Status: Complete
