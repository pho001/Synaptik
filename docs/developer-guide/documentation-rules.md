# Documentation rules

## Purpose

This guide defines the required documentation workflow. Detailed writing rules live in the [documentation style profiles](documentation/README.md) so each rule has one clear home.

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the authoritative architecture root and sole
authority index. Only the six scoped contracts it explicitly incorporates are [normative](../glossary.md#term-normative) within
their stated scopes. Documentation profiles and all other documentation control presentation and
review; they do not create architecture, change [module](../glossary.md#term-module) [ownership](../glossary.md#term-ownership-meanings), or override the contract.

Here **normative** means that a document states [binding](../glossary.md#term-normative) rules within its assigned [scope](../glossary.md#term-scope);
**explanatory** material helps readers understand those rules without creating new ones.
An **[API](../glossary.md#term-application-programming-interface-api) contract** describes what callers can rely on. **[Javadoc](../glossary.md#term-javadoc)** is Java's documentation
comment format and the tool-generated reference built from those comments.

## Select the document type

Before drafting or reviewing documentation:

1. identify the document's primary audience and purpose;
2. apply [General style](documentation/general-style.md);
3. apply the matching type profile from the [profile index](documentation/README.md);
4. use the [example format](documentation/example-format.md) when the document contains examples; and
5. define project terminology at first use and search the central [glossary](../glossary.md) for
   the relevant terms and anchors.

If a document genuinely serves two purposes, choose one primary profile and apply only the relevant requirements from the secondary profile. Do not combine every profile into a larger checklist.

## Follow one change through the review boundary

The implementer supplies a diff, relevant source/tests and concise execution
evidence. When the rules below require an independent pass, a separate reviewer
reads those inputs and produces corrected explanations plus a bounded review result.
The reviewer does not inherit the implementer's assumptions or automatically rerun
all successful tests.

A documentation handoff transfers information and editorial responsibility, not
native-memory ownership or cleanup duties. For example, when a parameter-copy API
changes, the reviewer checks who supplies the Arena, which values are copied and
when the session stops using them; a list of new method names is not enough.
Use [API/Javadoc style](documentation/api-and-javadoc-style.md) for that boundary
and [example format](documentation/example-format.md) for its worked usage.

## Risk-based documentation workflow

Every change receives a targeted documentation and [Javadoc](../glossary.md#term-javadoc) impact review. A separate clean
documentation/review context is mandatory only when the change affects public or supported [API](../glossary.md#term-application-programming-interface-api),
user-visible behavior or workflow, architecture or [module](../glossary.md#term-module) boundaries, project terminology, a
cross-module contract, a lasting module-local [lifecycle](../glossary.md#lifecycle) or other durable contract, or a Class C
boundary as defined in `AGENTS.md`. Required documentation lands in the same overall change before
completion.

For ordinary internal changes that do not meet those triggers, the implementer performs the
review. Update affected Javadocs and documents when needed; otherwise record
`Documentation impact: none; <reason>`. A Java visibility modifier does not determine risk:
durable internal lifecycle or behavior changes may still require an independent pass.

A **clean review context** is a separate review pass with an explicit [handoff](../glossary.md#term-handoff), rather than the
implementer's own recollection of what changed. A **diff** is the set of changes being reviewed.
The aim is to check the affected claims independently, not to repeat every historical check.

Every review is targeted to the actual diff. Read the selected documentation profiles, affected
source and focused tests, directly affected documentation, and exact authoritative contract
headings named by the task brief. Start with the root [scope](../glossary.md#term-scope) index and read only the applicable
root and incorporated scoped-contract headings for affected modules or boundaries. Do not require
the full [architecture contract](../glossary.md#architecture-contract),
roadmap, glossary, historical tasks, unrelated API guides, or unrelated modules by default. If the
applicable contract is missing or ambiguous, stop rather than assume.

An implementation agent may draft Javadoc or explanatory text while coding. When an independent
pass is required, it inspects the final diff and focused tests, then finalizes affected Javadoc,
explanatory documentation, examples, links, and glossary impact.

The handoff to that agent must include:

- the exact task goal and compact task brief;
- the implementation or documentation diff to review;
- affected APIs, behavior, workflows, and architecture boundaries;
- applicable architecture constraints;
- documentation already drafted or expected; and
- [validation](../glossary.md#term-validation) commands and required completion evidence.

An independent documentation/review context must:

1. read `AGENTS.md`, the task brief, this guide, applicable profiles, affected source and focused
   tests, and exact scoped contract headings;
2. inspect behavior and tests rather than relying only on the handoff summary;
3. finalize the content using the selected type profile;
4. search the glossary for relevant terms and anchors, then update it for new or changed project
   terms or record why no glossary change is needed;
5. validate links, anchors, examples, terminology, and formatting; and
6. record files reviewed, changes made, commands and results, limitations, and unresolved issues.

The implementation pass owns executable tests and hands their concise evidence to an independent
review context. That context must not rerun a successful Java test suite unless it changes
executable Java behavior after the [run](../glossary.md#term-run), the evidence is missing or stale, or the task identifies a
concrete cross-check risk. Javadoc-only edits do not require Java tests, but final Javadoc
generation must occur after those edits.

When an independent pass is required, the task remains incomplete until its result and evidence
are present in the same overall change.

## [Validation](../glossary.md#term-validation) and evidence

Use the validation section in the selected profile. At minimum:

- check local Markdown links and anchors;
- search relevant terminology and anchors in the [glossary](../glossary.md);
- verify examples against current behavior or label conceptual examples clearly;
- generate [Javadoc](../glossary.md#term-javadoc) for affected Java [modules](../glossary.md#term-module) after final Javadoc edits when Java APIs changed;
- [run](../glossary.md#term-run) `git diff --check` and check new files for trailing whitespace; and
- review the final diff for accidental authority changes, duplicated rules, and unrelated edits.

Evidence identifies the reviewer, selected profile, files or topics reviewed, and exact validation
commands with outcomes. Name a separate context only when one was required. Keep detailed [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings) in
[CI](../glossary.md#term-continuous-integration-ci) or tool [artifacts](../glossary.md#term-artifact). A no-change conclusion must include a reason; `N/A` alone is not evidence.

**[Continuous integration](../glossary.md#adr-ci-and-qualification) (CI)** means automated checks run by the project's integration service.
CI logs can retain detailed evidence, but a job's success still applies only to the source,
configuration and checks that job actually exercised.
