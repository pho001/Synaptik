# AGENTS.md

This file defines mandatory working instructions for AI agents and automated contributors.

These instructions apply to the entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## Required reading

Reading is indexed by role and actual change scope. Every contributor or agent reads:

- this `AGENTS.md`;
- the current task brief or user request;
- affected source and focused tests; and
- the exact authoritative contract headings named by the brief.

Planners and coordinators that create, reorder, or materially update plans also read
`docs/planning/planning-guide.md`, `docs/planning/roadmap.md`, and the relevant active master-plan
section. Before launching an executor, they verify the task is `Ready`, is at an authorized
frontier, and has satisfied dependencies, then record that verification in the task brief.

Executors do not read the full planning guide, full roadmap, completed predecessor tasks, task
histories, the full glossary, unrelated architecture sections, or prior completion evidence by
default. They use the scope index in `ARCHITECTURE.md`, then read the exact root and incorporated
scoped-contract headings identified by the task brief for every affected module or boundary. If
an applicable contract is missing or ambiguous, stop and ask for clarification rather than
inventing architecture.

Current source, focused tests, and authoritative current contracts take precedence over historical
task narratives. Read a predecessor task only when unresolved historical evidence is explicitly
needed.

`ARCHITECTURE.md` is the authoritative architecture root and sole authority index. Only the six
scoped files it explicitly incorporates are normative, and only within their stated scopes. All
other files under `docs/` are explanatory or coordinative and cannot override the contract.

## Change classification and task isolation

Classify work by actual impact, not Java visibility. If uncertain, classify upward.

- **Class A — Local:** bounded internal, test, or documentation-only work that changes none of the
  following: observable semantics or numerics; public or supported API; lifecycle or ownership;
  concurrency; persistence or application binary interface (ABI); backend capability; dependency
  direction; cross-module workflow; or a hot-path contract. One context is sufficient, and the
  main coordination context may implement it directly. The implementer performs the targeted
  documentation and Javadoc impact review.
- **Class B — Capability or public API:** a module capability, public or supported API,
  user-visible behavior, or lasting module-local lifecycle contract. Non-trivial work uses a clean
  implementation context. An independent documentation or review context is required when public
  API, user workflow or behavior, terminology, or a durable contract changes; otherwise record a
  reasoned no-change conclusion.
- **Class C — Architecture or high-risk boundary:** a module edge or ownership rule, the
  Runtime/Prepare boundary, native ABI or resource lifetime, concurrency, mixed-backend
  composition, generated-code or hot-path invariant, or a similarly high-risk cross-module
  contract. A clean implementation context and an independent targeted review/documentation
  context are mandatory, together with ADR, architecture, conformance, or integration validation
  where applicable.

A private method is not automatically Class A. A behavior-changing bug fix is at least Class B
and may be Class C.

The main context coordinates Classes B and C and may directly implement Class A. Give every clean
execution context a compact packet containing the exact goal, relevant files and symbols, exact
contract links or headings, acceptance criteria, validation, documentation impact, and completion
status format. The task brief plus a standard short launcher is the execution packet; do not rely
on remembered conversation context.

The implementation agent must work toward a complete result and should not intentionally leave the task half-finished.

If the task cannot be fully completed, the implementation agent must clearly state:

- what was completed
- what was not completed
- why it could not be completed
- what follow-up is required
- which files or areas are affected

At the end of every execution context, the agent must provide a completion summary.

The completion summary must include:

- completed changes
- files changed or created
- tests or validation performed
- unresolved issues, if any
- required follow-up, if any
- whether the task is complete or incomplete

Use this completion status format:

```text
Status: Complete
```

or:

```text
Status: Incomplete
Follow-up required: <specific follow-up>
```

A task should only be marked complete when the requested work has been implemented, documented when needed, and validated as far as possible within the task context.

If architectural uncertainty appears during implementation, the implementation agent must stop, explain the uncertainty, and request clarification instead of inventing new architecture.

## Architecture contract

All changes must preserve the architecture contract indexed by `ARCHITECTURE.md`, including its
global rules and the one incorporated scoped contract that owns the affected concern.

If a requested change conflicts with the root or an incorporated scoped contract, stop and explain
the conflict before editing code.

Do not duplicate architecture rules in this file. Architecture rules belong in the root or the
one scoped contract that the root explicitly incorporates for that concern.

When an architectural decision changes, update all relevant files in the same change:

1. `ARCHITECTURE.md` and the one owning scoped contract
2. the relevant explanatory document under `docs/architecture/`
3. an ADR under `docs/design/decisions/`, when the decision is significant
4. architecture tests under `testing/architecture-tests/`, when dependency rules change

## Documentation discipline

Documentation review is risk-based. A separate clean documentation/review context is mandatory
when work changes public or supported API, user-visible behavior or workflow, architecture or
module boundaries, project terminology, a cross-module contract, a lasting module-local lifecycle
or other durable contract, or any Class C boundary. It is not mandatory for ordinary internal
changes. In that case, the implementer reviews affected explanatory documentation and Javadocs
and records `Documentation impact: none; <reason>` when no update is needed.

An implementation agent may draft Javadoc while coding. When an independent pass is required, it
finalizes affected Javadoc, explanatory documentation, examples, links, and glossary impact in the
same overall change. The pass is targeted to the actual diff and relevant contracts. It reuses
successful executable-test evidence unless it changes executable behavior or identifies a concrete
reason to rerun a test. Follow `docs/developer-guide/documentation-rules.md` for the detailed
workflow.

Before writing documentation, identify its document type and apply the matching profile under
`docs/developer-guide/documentation/` together with the general style. Explain terms at first use
and include examples appropriate to that document type. Review the glossary with targeted search
for relevant terms and anchors; never require reading the entire glossary by default.

Documentation changes should keep the following distinction clear:

```text
ARCHITECTURE.md
  authoritative root and sole authority index

docs/architecture/contracts/
  six scoped normative contracts incorporated by the root

docs/
  other explanations, guides, design notes, examples, ADRs

docs/planning/
  non-authoritative implementation plans

AGENTS.md
  agent working instructions
```

When adding or changing architecture-related behavior, update documentation in the same change.

After every code change, review the affected documentation. Update or extend it in the same change when public APIs, behavior, configuration, architecture, module boundaries, workflows, or examples have changed. If no documentation update is needed, state that explicitly in the completion summary.

Every Java code change must be reflected in the Javadoc for the affected API or implementation contract. In the same change, add or update Javadoc whenever behavior, invariants, ownership, lifecycle, side effects, threading, nullability, parameters, return values, or failure modes change. Do not add Javadoc that merely restates the implementation. If the affected Javadoc remains accurate without modification, review it and state that explicitly in the completion summary.

Javadoc must always provide a meaningful, detailed description of the documented type, constructor, or method. Constructor and method Javadoc must document every input with `@param`, including relevant constraints, units, nullability, ownership, or mutation behavior. Every non-`void` method must document its result with `@return`, including result semantics and nullability. Document expected failure conditions with `@throws`. Constructors and `void` methods do not use `@return`.

## Planning discipline

Implementation plans and task briefs live under `docs/planning/`. Planners that create, reorder,
or materially update them read the planning guide, roadmap, and relevant active master-plan
section. Executors use the verified brief and scoped contracts rather than re-reading global
history.

Planning documents are not authoritative architecture contracts. If a planning document conflicts
with the root or an incorporated scoped contract, the architecture contract wins and
implementation must stop until the conflict is resolved.

Represent non-trivial implementation work as a cohesive compact task brief under the relevant
`tasks/` directory. A task should normally deliver one complete capability inside one module
rather than split its semantic type, public facade, tests, and documentation into separate
mechanical tasks. Follow the compact format and size guardrails in the planning guide. Existing
completed task specifications remain valid and need no retrospective rewrite.

Execute tasks in the order listed by the relevant master plan. Create a detailed task brief for the
next unfinished task only. Parallel or out-of-order execution is an explicit exception that must
be justified and recorded in the master plan.

## Legacy implementation reference

The `legacy/pre-rewrite` branch is a read-only reference for capabilities, observable behavior, tests, and historical context. Implement the new architecture from scratch. Do not copy or import legacy source files, internal package structure, dependency direction, runtime coupling, or implementation shortcuts into the new project. Reproduce only explicitly selected capabilities, expressed through new designs that comply with `ARCHITECTURE.md` and verified by new or adapted tests.

## Testing expectations

Use the validation tiers defined in `docs/planning/planning-guide.md`. A normal task validates the affected module and its documentation once. Run repository-wide validation at a recorded capability checkpoint, in CI, or when a change affects dependencies, architecture boundaries, shared build configuration, or multiple modules. Do not repeat a successful validation command in a second agent merely to reproduce the same evidence.

When changing module boundaries or dependencies, add or update architecture tests under:

```text
testing/architecture-tests/
```

When changing backend behavior, add or update backend conformance tests under:

```text
testing/backend-conformance/
```

When changing end-to-end behavior, add or update integration tests under:

```text
testing/integration-tests/
```

## Code discipline

Prefer small, focused, readable classes, and split classes that own multiple concepts. Do not create god classes, catch-all managers, broad facades, or vague utilities; place each responsibility in its owning module and layer. Keep public APIs minimal, prefer explicit domain names over generic `Manager`, `Helper`, `Util`, `Processor`, or `Service` names, and avoid unrelated refactors.

Add abstractions and interfaces only for a concrete current need, a real boundary, multiple implementations, a test seam, or an architecture contract. Do not conceal architecture violations behind facades, adapters, registries, or service locators.

## Performance discipline

Treat performance as a design priority, especially on runtime hot paths. Keep code readable without adding avoidable overhead per tensor element, operation, graph node, or execution step, and move expensive decisions to compile or prepare time when possible.

Avoid unnecessary allocation, boxing, reflection, string dispatch, map lookup, synchronization, and virtual indirection in hot paths. Do not obscure code for speculative optimization; document necessary optimizations that reduce readability and add tests or benchmarks when appropriate.

For every generated-code implementation, use a well-written, optimal clean Java implementation of the same specialized case as the design and review oracle. Directly generated bytecode must preserve that reference's semantic algorithm, hot-loop and dataflow shape, and avoidable-overhead profile. This does not require generating Java source and invoking `javac`, literal byte-for-byte Class-File identity, or guaranteed identical just-in-time compiler assembly. Any deviation requires an explicit technical reason and supporting evidence.

Validate generated code proportionately with semantic tests, generated Class-File or decompilation inspection, checks for hidden helper calls and avoidable allocation, boxing, reflection, or dispatch, and generated-versus-direct performance evidence for hot paths. A safe generic fallback may remain when specialization cannot be proved, but a proved specialized hot path must not silently inherit the fallback's avoidable overhead.

## Change discipline

Prefer small, focused changes.

Do not perform unrelated refactors in the same change.

Do not silently change architecture rules.

Do not leave work intentionally half-finished.

When unsure whether a change violates the architecture, stop and explain the uncertainty before editing.
