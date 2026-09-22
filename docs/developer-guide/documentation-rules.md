# Documentation rules

## Purpose

This guide defines the required documentation workflow. Detailed writing rules live in the [documentation style profiles](documentation/README.md) so each rule has one clear home.

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the authoritative architecture root and sole
authority index. Only the six scoped contracts it explicitly incorporates are normative within
their stated scopes. Documentation profiles and all other documentation control presentation and
review; they do not create architecture, change module ownership, or override the contract.

## Select the document type

Before drafting or reviewing documentation:

1. identify the document's primary audience and purpose;
2. apply [General style](documentation/general-style.md);
3. apply the matching type profile from the [profile index](documentation/README.md);
4. use the [example format](documentation/example-format.md) when the document contains examples; and
5. define project terminology at first use and search the central [glossary](../glossary.md) for
   the relevant terms and anchors.

If a document genuinely serves two purposes, choose one primary profile and apply only the relevant requirements from the secondary profile. Do not combine every profile into a larger checklist.

## Risk-based documentation workflow

Every change receives a targeted documentation and Javadoc impact review. A separate clean
documentation/review context is mandatory only when the change affects public or supported API,
user-visible behavior or workflow, architecture or module boundaries, project terminology, a
cross-module contract, a lasting module-local lifecycle or other durable contract, or a Class C
boundary as defined in `AGENTS.md`. Required documentation lands in the same overall change before
completion.

For ordinary internal changes that do not meet those triggers, the implementer performs the
review. Update affected Javadocs and documents when needed; otherwise record
`Documentation impact: none; <reason>`. A Java visibility modifier does not determine risk:
durable internal lifecycle or behavior changes may still require an independent pass.

Every review is targeted to the actual diff. Read the selected documentation profiles, affected
source and focused tests, directly affected documentation, and exact authoritative contract
headings named by the task brief. Start with the root scope index and read only the applicable
root and incorporated scoped-contract headings for affected modules or boundaries. Do not require
the full architecture contract,
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
- validation commands and required completion evidence.

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
executable Java behavior after the run, the evidence is missing or stale, or the task identifies a
concrete cross-check risk. Javadoc-only edits do not require Java tests, but final Javadoc
generation must occur after those edits.

When an independent pass is required, the task remains incomplete until its result and evidence
are present in the same overall change.

## Validation and evidence

Use the validation section in the selected profile. At minimum:

- check local Markdown links and anchors;
- search relevant terminology and anchors in the [glossary](../glossary.md);
- verify examples against current behavior or label conceptual examples clearly;
- generate Javadoc for affected Java modules after final Javadoc edits when Java APIs changed;
- run `git diff --check` and check new files for trailing whitespace; and
- review the final diff for accidental authority changes, duplicated rules, and unrelated edits.

Evidence identifies the reviewer, selected profile, files or topics reviewed, and exact validation
commands with outcomes. Name a separate context only when one was required. Keep detailed logs in
CI or tool artifacts. A no-change conclusion must include a reason; `N/A` alone is not evidence.
