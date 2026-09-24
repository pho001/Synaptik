# Task 0006: Explicit Graph Numerical-Profile Identity

## Status

Ready

Frontier verification: Model 0027 is Complete at implementation commit `ff86a302`; independent
validation passed 1,114 Model tests and nine architecture tests with zero failures and zero skips,
and mandatory independent Class C review returned `APPROVE` with zero findings. Config 0006 is the
sole Ready repository frontier. Model 0026 and Engine 0018, CPU 0017, and Metal 0015 remain Draft.

## Change class

Class B — this adds one durable public Config API identity. It changes no numerical semantics,
module edge, capability, lifecycle propagation, backend behavior, or execution path.

## Goal

Add the exact public graph numerical-profile identity already defined by Model authority, without
moving any meaning or behavior into Config:

```java
package io.github.pho001.synaptik.config.compile;

public enum NumericalProfile {
    STRICT_IEEE,
    ACCELERATOR
}
```

The enum is immutable declarative vocabulary only. It is not a default, algorithm, capability,
backend mode, route, or permission inferred from the environment.

## Scope

- Add public `io.github.pho001.synaptik.config.compile.NumericalProfile` with exactly the two enum
  constants `STRICT_IEEE` and `ACCELERATOR`, in that declaration order.
- Add meaningful type and constant Javadocs. Link meaning to the Model-owned numerical-profile
  contract: `STRICT_IEEE` identifies each operation's current allowed-result set;
  `ACCELERATOR` identifies only its bounded opt-in superset. Do not restate the normative table.
- Keep the declaration field-free and method-free apart from Java's ordinary enum API. Add no
  interface, nested type, annotation-driven behavior, alias, parser, display label, property,
  factory, default, or fallback helper.
- Update compile-package Javadoc to identify five current standalone values and to state that
  `NumericalProfile` is identity only and has no current compiler, Planning, Engine, or backend
  consumer.
- Add one focused `NumericalProfileTest` that locks the exact public API shape, declaration order,
  names, ordinary enum identity, and absence of aliases or behavioral surface.
- Update focused public/explanatory documentation from “selector planned” to “identity current,
  propagation and realization planned.” Keep Model as the sole semantic owner and retain current
  strict execution and fail-closed capability status.
- On completion, mark only Config 0006 Complete and promote only dependent Engine 0018 in a
  separate planning step; do not detail or implement Engine 0018 here.

## Non-goals

- A `CompileConfig` aggregate, constructor parameter, builder method, public compiler input, default
  selection, environment-derived selection, serialization, persistence, or compatibility schema
- Copying, translating, interpreting, validating, or calculating Model allowed-result sets in
  Config or tests
- Planning capability-query changes, Compiler artifacts, Prepare projection, Engine capture,
  Runtime state, session state, trace payloads, or any propagation of the identity
- CPU or Metal capability, routes, candidates, specialization, generated artifacts, tuning/cache
  identity, conformance, native code, or backend awareness of any kind
- Numerical algorithms, tolerances, fast-math switches, reduced precision, defaults, fallback,
  discovery, support checks, ranking, owner selection, or executable behavior
- Model 0026 FLOAT16 work or any status change to Engine 0018, CPU 0017, or Metal 0015
- Architecture-contract, ADR, Gradle, module-dependency, service-provider interface, or existing
  Config value changes

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) requires
  Model to own every profile-indexed allowed-result set and keeps profile policy off the Runtime hot
  path.
- [`ARCHITECTURE.md` — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  routes profile meaning and Config ownership to the one foundational contract.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  defines the two semantic identities and explicitly permits a later Config change to own only the
  immutable declarative selector.
- [Foundational modules — `modules/config`](../../../../architecture/contracts/foundational-modules.md#modulesconfig)
  permits declarative configuration and forbids services, concrete backend logic, Runtime state,
  algorithms, and mutable evidence.
- [`ARCHITECTURE.md` — Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) forbids a
  concrete-backend dependency and does not require a new dependency for this JDK-only enum.

If implementation requires Config to interpret profile meaning, choose a default, know a backend,
or propagate the value, stop and return the task to planning.

## Dependencies and integration

- Depends on: completed Model 0027 at `ff86a3020de2da08a57b8786f86a3075074baacd`
- Conflicts with: concurrent edits to `config.compile` package/API documentation or numerical-
  profile planning/status rows
- Parallel group: None
- Common base revision: `ff86a3020de2da08a57b8786f86a3075074baacd`
- Integration order: First; before Engine 0018, CPU 0017, or Metal 0015
- Integration validation: Config numerical-profile identity checkpoint
- Shared-document integration owner: Main planner

## Files and symbols

Implementation and focused tests:

- `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/NumericalProfile.java` — the
  exact public two-constant enum
- `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java` — current
  package inventory and identity-only boundary
- `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/NumericalProfileTest.java` —
  exact API-shape and identity-semantics coverage

Focused current-status documentation:

- `docs/api/public-api.md` and `docs/api/compile-api.md` — public identity, example, and explicit
  absence of a current consumer or default
- `docs/api/tensor-api.md` — replace only the stale “no public Config selector” status while
  retaining absent propagation/backend support
- `docs/architecture/module-boundaries.md` — replace only planned-selector wording with the current
  identity-only boundary; make no architecture claim beyond the incorporated contract
- `docs/glossary.md` — mark the declarative Config identity current while keeping semantics
  Model-owned and every downstream use planned

Planning completion paths are this brief, `../master-plan.md`, and `../../../roadmap.md`. No other
path is authorized.

## Acceptance criteria

1. `NumericalProfile` is a public final enum in the exact requested package with exactly
   `STRICT_IEEE` and `ACCELERATOR`, in that order and with those exact names.
2. The type declares no non-constant field, custom method, interface, nested type, custom
   constructor, alias, property, parser, factory, or default-selection surface. Only ordinary Java
   enum identity and generated `values()`/`valueOf(String)` behavior are observable.
3. `valueOf("STRICT_IEEE")` and `valueOf("ACCELERATOR")` return their exact singleton constants;
   unrecognized default/fast/relaxed aliases fail through ordinary enum behavior.
4. Javadocs and public references identify the enum as immutable graph-wide selector vocabulary,
   link to Model authority for meaning, and do not duplicate or reinterpret the normative
   operation-family table.
5. No API chooses one constant. No package or document describes either declaration order or enum
   ordinal as preference, default, capability, fallback, or backend suitability.
6. Package Javadoc and public API references describe five standalone compile values, with
   `NumericalProfile` unconsumed until Engine 0018 performs the coordinated propagation.
7. Existing compilation, Planning queries, Engine APIs, backends, caches, and execution behave
   exactly as before. Config gains no dependency and no production or test path outside the three
   named Config files changes.
8. Explanatory status text consistently says: Model semantics and Config identity are current;
   Planning/Compiler/Prepare/Engine propagation, backend realization, and relaxed capability remain
   planned; strict capability remains fail-closed.
9. Model 0026 remains independent Draft. Engine 0018, CPU 0017, and Metal 0015 remain Draft. No
   other task becomes Ready during implementation.
10. Focused Config tests, Config Javadoc, generated enum/package page inspection, changed-document
    link/anchor/fence/newline/whitespace validation, exact path/status audit, and `git diff --check`
    pass.

## Validation

Worker validation:

```bash
./gradlew :modules:config:test :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Inspect the generated `NumericalProfile` and compile-package pages. Confirm the diff contains no
production Java outside the enum and package Javadoc, no Gradle/dependency change, no consumer,
and only the authorized paths.

Integration/repository validation: run the Config numerical-profile identity checkpoint after the
independent documentation/API review. No Planning, Engine, backend, conformance, integration, or
repository-wide executable suite is required because this task does not propagate or consume the
identity.

## Documentation and review impact

- This public API and durable terminology change requires a separate clean documentation/API review
  covering Javadoc, package/public references, glossary status, exact API shape, and absence of
  semantic duplication or implied defaults.
- The review reuses successful Config test/Javadoc evidence unless it changes executable Java or
  identifies a concrete reason to rerun it. ADR 0019 remains Accepted and needs no status change.

## Result

Empty until execution.
