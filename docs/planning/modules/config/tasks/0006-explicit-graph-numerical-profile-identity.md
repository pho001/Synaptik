# Task 0006: Explicit [Graph](../../../../glossary.md#graph) Numerical-Profile [Identity](../../../../glossary.md#term-identity-meanings)

> **Editorial reading key — not part of the historical record.**
>
> This task records the introduction of a numerical-profile identity, not a currently available
> policy switch. The Config master plan records its later removal by Model0032. Even in the
> historical design, Config carried a name while Model authority defined numerical meaning;
> environment, backend speed and tuning evidence did not supply semantics. Representative
> tuning-profile identity is a separate evidence identifier, not a way to revive the retired
> numerical permission.
>
> Current reading: [hard intent, compile scope and optional optimization](../../../../reference/compiler/configuration.md#current-compile-configuration-inputs); [declarative tuning budgets and their lifecycle consumers](../../../../reference/compiler/configuration-values.md).

## Status

Complete

Implementation `314e049` plus remediation `37e9e9db` completed the [identity](../../../../glossary.md#term-identity-meanings)-only [scope](../../../../glossary.md#term-scope). A fresh
Config test/[Javadoc](../../../../glossary.md#term-javadoc) [run](../../../../glossary.md#term-run) passed 33 tests with zero failures, errors, or skips; the generated enum and
package pages were inspected; baseline-aware Markdown evidence and `git diff --check` passed; and
independent Class B review returned final `APPROVE` with zero findings. [Model](../../../../glossary.md#term-model) 0026 remains Draft;
dependent [Engine](../../../../glossary.md#term-engine) 0018 is the sole [Ready](../../../../glossary.md#term-ready) repository frontier, while [CPU](../../../../glossary.md#term-cpu) 0017 and [Metal](../../../../glossary.md#term-metal) 0015 remain
Draft.

## Change class

Class B — this adds one durable public Config [API](../../../../glossary.md#term-application-programming-interface-api) [identity](../../../../glossary.md#term-identity-meanings). It changes no numerical semantics,
[module](../../../../glossary.md#term-module) edge, capability, [lifecycle](../../../../glossary.md#lifecycle) propagation, [backend](../../../../glossary.md#backend) behavior, or execution path.

## Goal

Add the exact public [graph](../../../../glossary.md#graph) numerical-profile [identity](../../../../glossary.md#term-identity-meanings) already defined by [Model](../../../../glossary.md#term-model) authority, without
moving any meaning or behavior into Config:

```java
package io.github.pho001.synaptik.config.compile;

public enum NumericalProfile {
    STRICT_IEEE,
    ACCELERATOR
}
```

The enum is [immutable](../../../../glossary.md#term-immutable) declarative vocabulary only. It is not a default, algorithm, capability,
[backend](../../../../glossary.md#backend) mode, route, or permission inferred from the environment.

## [Scope](../../../../glossary.md#term-scope)

- Add public `io.github.pho001.synaptik.config.compile.NumericalProfile` with exactly the two enum
  constants <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator), in that declaration order.
- Add meaningful type and constant Javadocs. Link meaning to the Model-owned numerical-profile
  contract: <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`STRICT_IEEE` identifies each [operation](../../../../glossary.md#operation)'s current allowed-result set;
  `ACCELERATOR` identifies only its bounded opt-in superset. Do not restate the [normative](../../../../glossary.md#term-normative) table.
- Keep the declaration field-free and method-free apart from Java's ordinary enum API. Add no
  interface, nested type, annotation-driven behavior, alias, parser, display label, property,
  factory, default, or fallback helper.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Update [compile](../../../../glossary.md#compile)-package [Javadoc](../../../../glossary.md#term-javadoc) to identify five current standalone values and to state that
  `NumericalProfile` is identity only and has no current compiler, Planning, Engine, or backend
  consumer.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Add one focused `NumericalProfileTest` that locks the exact public API shape, declaration order,
  names, ordinary enum identity, and absence of aliases or behavioral surface.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Update focused public/explanatory documentation from “selector planned” to “identity current,
  propagation and realization planned.” Keep Model as the sole semantic owner and retain current
  strict execution and fail-closed capability status.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- On completion, mark only Config 0006 Complete and promote only dependent Engine 0018 in a
  separate planning step; do not detail or implement Engine 0018 here.

## Non-goals

- A `CompileConfig` aggregate, constructor parameter, builder method, public compiler input, default
  selection, environment-derived selection, serialization, persistence, or compatibility schema<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [compiler glossary](../../../../glossary.md#term-compiler); [serialization glossary](../../../../glossary.md#term-serialization); [persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema))</span>
- Copying, translating, interpreting, validating, or calculating Model allowed-result sets in
  Config or tests
- Planning capability-query changes, Compiler artifacts, Prepare projection, Engine capture,
  Runtime state, session state, trace payloads, or any propagation of the identity<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [artifacts glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [capture glossary](../../../../glossary.md#term-capture-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [trace glossary](../../../../glossary.md#trace); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- CPU or Metal capability, routes, candidates, specialization, generated artifacts, tuning/cache
  identity, conformance, native code, or backend awareness of any kind<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [conformance glossary](../../../../glossary.md#term-conformance); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [backend glossary](../../../../glossary.md#backend))</span>
- Numerical algorithms, tolerances, fast-math switches, reduced precision, defaults, fallback,
  discovery, support checks, ranking, owner selection, or executable behavior<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Model](../../../../glossary.md#term-model) 0026 [FLOAT16](../../../../glossary.md#term-float16) work or any status change to [Engine](../../../../glossary.md#term-engine) 0018, [CPU](../../../../glossary.md#term-cpu) 0017, or [Metal](../../../../glossary.md#term-metal) 0015
- Architecture-contract, ADR, Gradle, module-dependency, service-provider interface, or existing
  Config value changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) requires
  Model to own every profile-indexed allowed-result set and keeps profile policy off the Runtime hot
  path.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [`ARCHITECTURE.md` — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  routes profile meaning and Config ownership to the one foundational contract.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  defines the two semantic identities and explicitly permits a later Config change to own only the
  immutable declarative selector.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- [Foundational modules — `modules/config`](../../../../architecture/contracts/foundational-modules.md#modulesconfig)
  permits declarative configuration and forbids services, concrete backend logic, Runtime state,
  algorithms, and mutable evidence.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [`ARCHITECTURE.md` — Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> forbids a
  concrete-backend dependency and does not require a new dependency for this JDK-only enum.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>

If implementation requires Config to interpret profile meaning, choose a default, know a [backend](../../../../glossary.md#backend),
or propagate the value, stop and return the task to [planning](../../../../glossary.md#term-planning).

## Dependencies and integration

- Depends on: completed [Model](../../../../glossary.md#term-model) 0027 at `ff86a3020de2da08a57b8786f86a3075074baacd`
- Conflicts with: concurrent edits to `config.compile` package/API documentation or numerical-
  profile planning/status rows<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None
- Common base revision: `ff86a3020de2da08a57b8786f86a3075074baacd`
- Integration order: First; before [Engine](../../../../glossary.md#term-engine) 0018, [CPU](../../../../glossary.md#term-cpu) 0017, or [Metal](../../../../glossary.md#term-metal) 0015
- Integration [validation](../../../../glossary.md#term-validation): Config numerical-profile [identity](../../../../glossary.md#term-identity-meanings) [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner

## Files and symbols

Implementation and focused tests:

- `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/NumericalProfile.java` — the
  exact public two-constant enum<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java` — current
  package inventory and identity-only boundary<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/NumericalProfileTest.java` —
  exact API-shape and identity-semantics coverage<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>

Focused current-status documentation:

- `docs/api/public-api.md` and `docs/api/compile-api.md` — public identity, example, and explicit
  absence of a current consumer or default
- `docs/api/tensor-api.md` — replace only the stale “no public Config selector” status while
  retaining absent propagation/backend support<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- `docs/architecture/module-boundaries.md` — replace only planned-selector wording with the current
  identity-only boundary; make no architecture claim beyond the incorporated contract
- `docs/glossary.md` — mark the declarative Config identity current while keeping semantics
  Model-owned and every downstream use planned<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

[Planning](../../../../glossary.md#term-planning) completion paths are this brief, `../master-plan.md`, and `../../../roadmap.md`. No other
path is authorized.

## Acceptance criteria

1. `NumericalProfile` is a public final enum in the exact requested package with exactly
   `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator), in that order and with those exact names.
2. The type declares no non-constant field, custom method, interface, nested type, custom
   constructor, alias, property, parser, factory, or default-selection surface. Only ordinary Java
   enum identity and generated <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [alias glossary](../../../../glossary.md#aliasing-and-arena); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`values()`/`valueOf(String)` behavior are observable.
3. `valueOf("STRICT_IEEE")` and `valueOf("ACCELERATOR")` return their exact singleton constants;
   unrecognized default/fast/relaxed aliases fail through ordinary enum behavior.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
4. Javadocs and public references identify the enum as immutable graph-wide selector vocabulary,
   link to Model authority for meaning, and do not duplicate or reinterpret the normative
   operation-family table.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [Model glossary](../../../../glossary.md#term-model); [normative glossary](../../../../glossary.md#term-normative); [operation glossary](../../../../glossary.md#operation))</span>
5. No API chooses one constant. No package or document describes either declaration order or enum
   ordinal as preference, default, capability, fallback, or backend suitability.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
6. Package [Javadoc](../../../../glossary.md#term-javadoc) and public [API](../../../../glossary.md#term-application-programming-interface-api) references describe five standalone [compile](../../../../glossary.md#compile) values, with
   `NumericalProfile` unconsumed until [Engine](../../../../glossary.md#term-engine) 0018 performs the coordinated propagation.
7. Existing compilation, Planning queries, Engine APIs, backends, caches, and execution behave
   exactly as before. Config gains no dependency and no production or test path outside the three
   named Config files changes.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency))</span>
8. Explanatory status text consistently says: Model semantics and Config identity are current;
   Planning/Compiler/Prepare/Engine propagation, backend realization, and relaxed capability remain
   planned; strict capability remains fail-closed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
9. Model 0026 remains independent Draft. Engine 0018, CPU 0017, and Metal 0015 remain Draft. No
   other task becomes Ready during implementation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [Ready glossary](../../../../glossary.md#term-ready))</span>
10. Focused Config tests, Config Javadoc, generated enum/package page inspection, changed-document
    link/anchor/fence/newline/whitespace validation, exact path/status audit, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check`
    pass.

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :modules:config:test :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Inspect the generated `NumericalProfile` and [compile](../../../../glossary.md#compile)-package pages. Confirm the diff contains no
production Java outside the enum and package [Javadoc](../../../../glossary.md#term-javadoc), no [Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency) change, no consumer,
and only the authorized paths.

Integration/repository validation: [run](../../../../glossary.md#term-run) the Config numerical-profile [identity](../../../../glossary.md#term-identity-meanings) [checkpoint](../../../../glossary.md#term-checkpoint) after the
independent documentation/[API](../../../../glossary.md#term-application-programming-interface-api) review. No [Planning](../../../../glossary.md#term-planning), [Engine](../../../../glossary.md#term-engine), [backend](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), integration, or
repository-wide executable suite is required because this task does not propagate or consume the
identity.

## Documentation and review impact

- This public API and durable terminology change requires a separate clean documentation/API review
  covering Javadoc, package/public references, glossary status, exact API shape, and absence of
  semantic duplication or implied defaults.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape))</span>
- The review reuses successful Config test/Javadoc evidence unless it changes executable Java or
  identifies a concrete reason to rerun it. ADR 0019 remains Accepted and needs no status change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Result

Complete at implementation `314e049fdcf399a5d7c6913a62db6b1a355e0d0e` plus remediation
`37e9e9db0bdb0b11564e3696079b0d6a5e0c8109`. Fresh Config tests and [Javadoc](../../../../glossary.md#term-javadoc) passed 33 tests with
zero failures, errors, or skips; generated enum/package pages were inspected; baseline-aware
Markdown [validation](../../../../glossary.md#term-validation) and `git diff --check` passed; and independent Class B review returned final
`APPROVE` with zero findings. The delivered enum remains [identity](../../../../glossary.md#term-identity-meanings)-only. This [planning](../../../../glossary.md#term-planning)-only
completion promotes only [Engine](../../../../glossary.md#term-engine) 0018; [Model](../../../../glossary.md#term-model) 0026, [CPU](../../../../glossary.md#term-cpu) 0017, and [Metal](../../../../glossary.md#term-metal) 0015 remain Draft.
