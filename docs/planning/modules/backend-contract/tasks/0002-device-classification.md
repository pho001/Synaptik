# Task 0002: [Device](../../../../glossary.md#term-device) Classification

## Status

Complete

## Goal

Add the smallest [backend](../../../../glossary.md#backend)-neutral [device](../../../../glossary.md#term-device) classification needed by later availability snapshots,
declarative requirements, and configuration: [`CPU`](../../../../glossary.md#term-cpu) versus [`ACCELERATOR`](../../../../glossary.md#term-accelerator).

[Mental model](../../../../glossary.md#term-mental-model):

```text
BackendDeviceId
  = exact backend-scoped device identity

DeviceClass.CPU or DeviceClass.ACCELERATOR
  = coarse declarative category attached by a later availability fact
```

This task defines the category vocabulary only. It does not attach a class to an [identity](../../../../glossary.md#term-identity-meanings), inspect
hardware, publish availability, select [ownership](../../../../glossary.md#term-ownership-meanings), estimate performance, or identify an execution
route.

## [Scope](../../../../glossary.md#term-scope)

- Add public enum [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) in `io.github.pho001.synaptik.backend.contract` with exactly these
  constants in order:<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
  1. [`CPU`](../../../../glossary.md#term-cpu)
  2. [`ACCELERATOR`](../../../../glossary.md#term-accelerator)
- Define `CPU` as a device executing through a general-purpose central processing unit. CPU
  scalar, vector, assembly, and OpenBLAS choices remain routes inside a CPU backend and do not
  create additional device classes.<span data-glossary-navigation="true"> ([central processing unit glossary](../../../../glossary.md#cpu-gpu-host-and-device); [vector glossary](../../../../glossary.md#term-vector))</span>
- Define `ACCELERATOR` as a non-CPU compute device intended for offloaded computation. GPU, neural
  processing unit, and other accelerator implementations may use this category, but this task
  introduces no subcategory, performance, memory, vendor, or execution promise.<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu))</span>
- Preserve the exact declaration order for stable enum identity and diagnostics. Document that
  ordinal order is not preference, score, priority, capability, or fallback order.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add no project field, constructor, method, nested type, implemented project interface, alias,
  sentinel, metadata, label, parser, or formatter. Compiler-generated enum members and ordinary<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
  `Enum` behavior are the entire executable surface.
- Keep `DeviceClass` distinct from:
  - [`BackendId`](../../../../glossary.md#backend-identity--backendid), which names an [ownership](../../../../glossary.md#term-ownership-meanings) domain such as CPU, [Metal](../../../../glossary.md#term-metal), or [CUDA](../../../../glossary.md#term-cuda);
  - [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), which names one [device](../../../../glossary.md#term-device) inside a [backend](../../../../glossary.md#backend);
  - a backend-internal route such as [scalar](../../../../glossary.md#term-scalar), [OpenBLAS](../../../../glossary.md#term-openblas), [MPSGraph](../../../../glossary.md#term-mpsgraph), or a CUDA [kernel](../../../../glossary.md#kernel); and
  - availability, capability, configuration preference, and selected ownership.
- Do not add `DeviceClass` to `BackendDeviceId`. A later immutable availability snapshot will
  associate a reported device identity with its class.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Update the package Javadoc to include the current category contract while preserving every
  identity boundary from task 0001.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Add one focused test class covering exact enum/API shape, constant names and order, value lookup,
  ordinary enum identity, absence of project metadata/aliases/nested types, and separation from
  backend/device identity records.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Finalize Javadocs and update Public API status, the capability-provider guide, glossary,
  backend-contract master plan, and roadmap in the same overall change.

## Out of [scope](../../../../glossary.md#term-scope)

- [`GPU`](../../../../glossary.md#term-gpu), `NPU`, `FPGA`, `TPU`, `DSP`, [`HOST`](../../../../glossary.md#term-host), `REMOTE`, `UNKNOWN`, `OTHER`, `DEFAULT`, or another
  class or sentinel
- vendor, architecture, instruction-set, memory-topology, unified/discrete-memory, local/remote,
  integrated/discrete, power, throughput, latency, capacity, precision, or capability metadata<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [throughput glossary](../../../../glossary.md#term-throughput); [latency glossary](../../../../glossary.md#term-latency); [precision glossary](../../../../glossary.md#term-precision); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- modifying the components, [validation](../../../../glossary.md#term-validation), equality, or Javadocs of [`BackendId`](../../../../glossary.md#backend-identity--backendid) or [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid)
- attaching a class directly to a [device](../../../../glossary.md#term-device) [identity](../../../../glossary.md#term-identity-meanings)
- [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), backend intent, profile, capability query,
  scoring, or prepare configuration<span data-glossary-navigation="true"> ([backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [prepare glossary](../../../../glossary.md#prepare))</span>
- backend or device registration, discovery, lookup, probing, service locator, allocation,
  availability, liveness, health, capability, ownership, partitioning, preparation, execution,
  runtime residency, kernel route, transfer, storage, or tracing behavior<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [service locator glossary](../../../../glossary.md#term-service-locator); [liveness glossary](../../../../glossary.md#term-liveness); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), [OpenBLAS](../../../../glossary.md#term-openblas), [MPSGraph](../../../../glossary.md#term-mpsgraph), or [kernel](../../../../glossary.md#kernel) implementation changes
- dependencies, Gradle, Java version, architecture contract, ADR, architecture test, another
  module, concrete backend, backend conformance, or integration changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [module glossary](../../../../glossary.md#term-module); [concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed [backend](../../../../glossary.md#backend)-contract task 0003 or later specification
- returning to or implementing [trace](../../../../glossary.md#trace) task 0003 or another trace task
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Backend-contract master plan](../master-plan.md)
- [Task 0001](0001-backend-and-device-identifiers.md)
- [Public API status](../../../../api/public-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/backend-contract` owns [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) as minimal declarative [backend](../../../../glossary.md#backend) vocabulary.
- The selected classification follows the current architecture distinction between CPU-specific
  preparation and general accelerator preparation. It does not classify concrete implementation
  routes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- A class is descriptive data only. It neither proves availability nor selects a backend or
  device.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- `modules/backend-contract` remains a JDK-only leaf independent of model, trace, config,
  planning, compiler, runtime, prepare, engine, and concrete backend modules.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [leaf glossary](../../../../glossary.md#term-leaf); [model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace); [planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
- Stop if implementation requires a third class, metadata, association record, availability or
  requirement type, another module, dependency edit, or architecture decision.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package retained:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identities and
  declarative vocabulary<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

Type added:

- `io.github.pho001.synaptik.backend.contract.DeviceClass` — coarse CPU/accelerator device
  category<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator); [device glossary](../../../../glossary.md#term-device))</span>

Existing package [Javadoc](../../../../glossary.md#term-javadoc) is updated to explain the new current type. [`BackendId`](../../../../glossary.md#backend-identity--backendid) and
[`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) declarations and behavior remain unchanged. The focused test mirrors the
production package. No subpackage or helper type is added.

## Affected files

Production — exactly two paths:

- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/DeviceClass.java`
- [Javadoc](../../../../glossary.md#term-javadoc)-only update to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/package-info.java`

Tests — exactly one path:

- add `modules/backend-contract/src/test/java/io/github/pho001/synaptik/backend/contract/DeviceClassTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) — exactly six paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), `BackendIdentityTest`, completed task
0001, [trace](../../../../glossary.md#trace) source/tests/master plan, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr)
documents/tests, [Gradle](../../../../glossary.md#term-gradle), config/planning placeholders and master plans, concrete [backends](../../../../glossary.md#backend), backend
[conformance](../../../../glossary.md#term-conformance), integration tests, and legacy evidence.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact nine paths above. Stop if implementation requires another production type,
test, document, package, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), architecture change, build edit, or detailed
follow-up specification.

## Acceptance criteria

- [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) is a public enum in the exact [backend](../../../../glossary.md#backend)-contract package with exactly [`CPU`](../../../../glossary.md#term-cpu) and
  [`ACCELERATOR`](../../../../glossary.md#term-accelerator) in that order.
- The enum declares no project field, constructor, method, nested type, interface, alias,
  metadata, or additional constant. Only compiler-generated enum mechanics and inherited <span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler))</span>`Enum`
  behavior exist.
- Javadocs define both classes, explain intended later use, distinguish backend/device/class/route,
  and state that declaration/ordinal order conveys no preference, score, priority, capability, or
  fallback meaning.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), and their focused tests remain source- and behavior-unchanged.
- No class association is stored in `BackendDeviceId`; availability snapshots remain planned.
- Production imports remain [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only; the preferred enum implementation needs no import.
- The focused test fails on [constant](../../../../glossary.md#term-constant)/order/[API](../../../../glossary.md#term-application-programming-interface-api) drift and confirms ordinary `valueOf`, `values`,
  enum identity, and separation from the two record types without testing a nonexistent policy.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Package Javadoc, Public API status, capability-provider guide, and glossary mark only the two
  identities plus coarse device classification current. Availability, requirements, capability
  providers, planning, prepare, runtime, and concrete backends remain planned.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Backend-contract master plan and roadmap identify 0002 as Complete. Backend-contract task 0003+
  and trace task 0003+ remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- A separate clean-context documentation-focused pass finalizes the enum/package Javadocs and six
  documentation/planning paths after Java tests pass.
- Exact nine-path [scope](../../../../glossary.md#term-scope), Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no identity behavior, architecture, dependency, Gradle, trace, config/planning Java,
  concrete backend, or cross-module implementation path changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [concrete backend glossary](../../../../glossary.md#concrete-backend); [module glossary](../../../../glossary.md#term-module))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing. After executable Java stabilizes, run exactly one final
[backend](../../../../glossary.md#backend)-contract [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Do not run repository-wide tests: this is a small
[JDK](../../../../glossary.md#term-java-development-kit-jdk)-only additive enum task with no [dependency](../../../../glossary.md#term-dependency) or architecture-rule change. A later
backend-contract capability [checkpoint](../../../../glossary.md#term-checkpoint) owns repository-wide validation.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and task-0001 contracts, applies General style with [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Backend guide, and
[Planning](../../../../glossary.md#term-planning) profiles where relevant, finalizes the enum/package Javadocs and six
documentation/planning paths, records reasoned no-change conclusions, and runs:

```bash
./gradlew :modules:backend-contract:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package and [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) pages; rely on the
focused automated enum/API-[shape](../../../../glossary.md#shape) checks; confirm exactly nine task paths; and confirm
backend-contract 0002 Complete/0003 Draft/no detailed 0003 task before completion.

## Dependencies

- Completed [backend](../../../../glossary.md#backend)-contract task 0001 [identity](../../../../glossary.md#term-identity-meanings) values and package foundation.
- The selected model milestone and trace tasks 0001–0002 are Complete as roadmap sequencing
  context only; this task adds no model or trace dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- Task 0003 backend availability snapshot remains Draft without a detailed specification. It will
  associate reported <span data-glossary-navigation="true"> ([backend availability snapshot glossary](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot))</span>[`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) values with [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) and availability facts without
  performing discovery.
- Task 0004 declarative backend requirements remains Draft and may later match exact identities or
  coarse classes without embedding live backend services.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [embedding glossary](../../../../glossary.md#embedding))</span>
- [Trace](../../../../glossary.md#trace) tasks remain Draft until their relevant [producer](../../../../glossary.md#term-producer)-owned contracts are concrete.

## Architecture impact

Expected impact: None.

The task implements the existing [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) [ownership](../../../../glossary.md#term-ownership-meanings) rule and current [CPU](../../../../glossary.md#term-cpu)-versus-[accelerator](../../../../glossary.md#term-accelerator)
[planning](../../../../glossary.md#term-planning) vocabulary. It changes no [module](../../../../glossary.md#term-module) direction, [lifecycle](../../../../glossary.md#lifecycle) decision, registration mechanism,
or [backend](../../../../glossary.md#backend) behavior. Documentation changes implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/partition/prepare-backend architecture
documents, documentation/planning rules and profiles, roadmap, backend-contract and trace master
plans, completed backend-contract task 0001, task 0002, current identity/package source/tests,
Public API, capability-provider guide, glossary, relevant config/planning master plans, and Java
26 root configuration.

Implement docs/planning/modules/backend-contract/tasks/0002-device-classification.md exactly
inside its nine authorized paths. Add only DeviceClass with CPU and ACCELERATOR in exact order,
one focused test, and the permitted package-Javadoc/status documentation updates. Preserve the
identity contracts and exact category/route/availability boundaries. Add no third class,
metadata, association, snapshot, requirement, provider, dependency, or later task. Stop on
architecture, taxonomy, dependency, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:backend-contract:test after
executable Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-context
documentation agent in the same overall change. That pass finalizes Javadocs, Public API,
capability-provider guide, glossary, task/master/roadmap status, runs backend-contract Javadoc and
documentation/scope checks, and reuses successful Java evidence unless executable behavior
changes. Mark 0002 Complete only after both passes succeed. Leave 0003+ and trace 0003+ Draft
without detailed specifications.
```

## Local decisions

- Select exactly [`CPU`](../../../../glossary.md#term-cpu) and [`ACCELERATOR`](../../../../glossary.md#term-accelerator) because current configuration and partition-scoring
  language distinguishes CPU preparation from accelerator preparation but has no concrete need
  for GPU/NPU/FPGA subcategories.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [GPU glossary](../../../../glossary.md#term-gpu))</span>
- Classify devices, not backends. A backend owns identities and later reports class through an
  availability fact; identity remains stable and independent of current availability metadata.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Keep OpenBLAS, scalar, vector, MPSGraph, and concrete kernels out of the enum because they are
  prepare-time routes, not device classes.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [device glossary](../../../../glossary.md#term-device))</span>
- Add no `UNKNOWN` or `OTHER` sentinel. A producer that cannot truthfully supply the selected
  vocabulary requires a future explicit planning decision rather than silent classification.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [planning glossary](../../../../glossary.md#term-planning))</span>
- Give enum order no scoring meaning. Later config and planning must express preference through
  their own typed policies.
- Preserve the trace interleave: this task makes producer vocabulary more concrete without adding
  a trace dependency or prematurely designing trace DTOs.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Known limitations

- The two-class vocabulary cannot distinguish GPU, NPU, FPGA, integrated/discrete, local/remote,
  or memory topology. No current consumer requires those distinctions.<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu); [topology glossary](../../../../glossary.md#term-topology))</span>
- A class does not indicate availability, capability, performance, power, precision support,
  storage model, or selected ownership.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No current public type associates a [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) with its class; task 0003 will own that
  immutable availability relationship.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- No external [serialization](../../../../glossary.md#term-serialization) or compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning inspected authoritative backend-contract ownership, dependency, partition-scoring,
  and prepare/backend boundaries; current identities/tests/package Javadocs; config/planning
  master plans; Public API, capability guide, glossary, roadmap; documentation/planning profiles;
  and the Java 26 module baseline.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [partition glossary](../../../../glossary.md#partition); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- Current architecture and planning consumers distinguish CPU from a general accelerator but
  contain no requirement for a narrower accelerator subtype. The selected two-value taxonomy is
  therefore the smallest current useful contract.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- The planned change contains two production, one test, and six documentation/planning paths:
  exactly nine total.
- Planning-time repository Markdown validation passed for 217 Markdown files, 3,734 local links,
  218 local anchors, 2,722 fence markers, final newlines, and trailing whitespace.
- [Planning](../../../../glossary.md#term-planning)-time `git diff --check` passed. At that frontier, planning contained exactly one Ready
  row, for backend-contract task 0002. No detailed backend-contract task-0003 or trace task-0003
  specification existed.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [trace glossary](../../../../glossary.md#trace))</span>
- The implementation context `/root/implement_backend_contract_0002` ran the focused
  `DeviceClassTest` twice during development; all three selected tests passed in each run. After
  executable Java stabilized, that context ran exactly one final<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `./gradlew :modules:backend-contract:test`; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL` in 704 ms. XML
  reports contain 9 tests across 2 suites: `BackendIdentityTest` has 6 tests and
  `DeviceClassTest` has 3, with zero failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_backend_contract_0002/backend_contract_0002_docs` applied General style with
  API/Javadoc, Backend guide, Planning, and Example profiles as applicable. It independently
  inspected the architecture boundaries, documentation/planning rules, completed identity and
  trace foundations, final source/tests, public API, backend guide, glossary, module plans,
  roadmap, and Java 26 root/module build configuration before finalizing the documentation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The documentation context changed only Javadoc and Markdown after the successful final module
  suite. Executable Java and tests remained unchanged, so it reused the implementation context's
  Java-test evidence rather than rerunning it.
- The documentation context ran `./gradlew :modules:backend-contract:javadoc`; Gradle reported
  `BUILD SUCCESSFUL` in 1 s with two executed tasks. Inspection of generated `DeviceClass.html`
  and `package-summary.html` confirmed the two constants plus rendered category, association,
  identity/route-separation, ordinal, and unsupported-policy boundaries.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- Final `python3 /tmp/validate_synaptik_markdown.py` passed 217 Markdown files, 3,736 local links,
  220 local anchors, 2,724 fence markers, final newlines, and trailing whitespace.
- Final `git diff --check`, the exact-[scope](../../../../glossary.md#term-scope) command, and `git status --short` passed. They reported
  exactly the nine authorized task paths: two production, one test, and six
  documentation/planning paths, with no unrelated working-tree path.
- Status and task-directory inspection confirmed backend-contract 0002 is Complete, 0003–0004
  remain Draft, and no detailed backend-contract 0003 specification exists. Trace 0001–0002
  remain Complete, trace 0003–0008 remain Draft, and no detailed trace 0003 specification exists.
- [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), their Javadocs/source, and `BackendIdentityTest` remain unchanged
  because the new enum neither modifies nor contains identity. Their validation, equality,
  retention, and no-availability contracts remain accurate.
- No architecture contract, focused architecture explanation, ADR, or architecture test changed:
  the task implements the existing backend-contract ownership rule without changing a module or
  dependency boundary. Gradle, dependencies, and the Java 26 baseline remain unchanged because
  the enum is JDK-only.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- Trace source/tests/documentation and its master plan remain unchanged because device class is
  producer vocabulary, not a trace correlation or payload. Config and planning Java plus their
  master plans remain unchanged because no configuration, capability, availability, scoring, or
  ownership behavior was added.<span data-glossary-navigation="true"> ([device class glossary](../../../../glossary.md#device-class--deviceclass); [producer glossary](../../../../glossary.md#term-producer); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Other modules, concrete backends, backend conformance, integration tests, and legacy remain
  unchanged because this task adds no provider, discovery, registration, route, preparation,
  execution, serialization, or external compatibility behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [serialization glossary](../../../../glossary.md#term-serialization))</span>

## Implementation notes

- Added the exact two-constant enum without fields, constructors, methods, nested types,
  interfaces, aliases, metadata, or a third category; compiler-generated enum mechanics remain the
  only executable surface beyond inherited <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler))</span>`Enum` behavior.
- Kept class vocabulary separate from both identity records and from backend-internal routes. No
  current type associates <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend))</span>[`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) with [`DeviceClass`](../../../../glossary.md#device-class--deviceclass); that relationship remains in the
  Draft availability-snapshot frontier.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- The documentation pass refined only `DeviceClass` and package Javadocs and finalized the six
  authorized Markdown/planning paths. <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid), `BackendDeviceId`, their Javadocs, and
  `BackendIdentityTest` remained unchanged and accurate.

## Completion summary

- Completed changes: added the coarse [`CPU`](../../../../glossary.md#term-cpu)/[`ACCELERATOR`](../../../../glossary.md#term-accelerator) device-class vocabulary, focused exact-
  shape tests, finalized enum/package Javadocs, and synchronized public, glossary, guide, task,
  master-plan, and roadmap status.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [shape glossary](../../../../glossary.md#shape))</span>
- Files changed or created: exactly the two production, one test, and six
  documentation/planning paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the implementation context passed two focused three-test runs and one
  final nine-test/two-suite backend-contract module run; the documentation context reused that
  evidence because executable Java remained unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0002/backend_contract_0002_docs` independently reviewed and
  finalized all affected Javadocs and documentation.
- Documentation impact: current identity and device-class roles are explicit; association,
  availability, requirements, capabilities, configuration policy, planning, prepare, runtime,
  concrete backend behavior, routes, and execution remain planned or separately owned.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) and package documentation define the two classes, distinct
  identity/category/route roles, later association boundary, and absence of availability,
  capability, preference, scoring, ownership-selection, or execution meaning.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: added `DeviceClass` and updated the current-versus-planned backend-contract
  foundation summary.
- Unresolved issues: None.
- Follow-up required: None. Task 0003 remains Draft without a detailed specification.

Status: Complete
