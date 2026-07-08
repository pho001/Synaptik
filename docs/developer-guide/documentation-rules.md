# Documentation rules

## Purpose

This guide defines the required documentation workflow. Detailed writing rules live in the [documentation style profiles](documentation/README.md) so each rule has one clear home.

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) remains the authoritative architecture contract. Documentation profiles control presentation and review; they do not create architecture, change module ownership, or override the contract.

## Select the document type

Before drafting or reviewing documentation:

1. identify the document's primary audience and purpose;
2. apply [General style](documentation/general-style.md);
3. apply the matching type profile from the [profile index](documentation/README.md);
4. use the [example format](documentation/example-format.md) when the document contains examples; and
5. define project terminology at first use and review the central [glossary](../glossary.md).

If a document genuinely serves two purposes, choose one primary profile and apply only the relevant requirements from the secondary profile. Do not combine every profile into a larger checklist.

## Documentation-focused agent workflow

Substantive documentation work and every code or behavior change require a documentation-focused pass in a clean agent or thread context, distinct from the implementation context. Separate context provides an independent review perspective; it does not mean a separate branch, commit, pull request, or future task. Required documentation must land in the same overall change before the task is complete.

The pass must be targeted to the actual diff. Read the authoritative contract, selected documentation profiles, affected source and tests, and directly affected documentation. Do not require unrelated architecture pages, API guides, historical tasks, or modules merely to enlarge the checklist. Add broader reading only when the change can affect those contracts.

An implementation agent may draft Javadoc or explanatory text while coding. The documentation-focused agent must inspect the final diff and tests, then independently finalize affected Javadoc, explanatory documentation, examples, links, and glossary impact.

The handoff to that agent must include:

- the exact task goal and task specification;
- the implementation or documentation diff to review;
- affected APIs, behavior, workflows, and architecture boundaries;
- applicable architecture constraints;
- documentation already drafted or expected; and
- validation commands and required completion evidence.

The documentation-focused agent must:

1. read `AGENTS.md`, `ARCHITECTURE.md`, this guide, the applicable profiles, and directly relevant source and documentation;
2. inspect behavior and tests rather than relying only on the handoff summary;
3. finalize the content using the selected type profile;
4. update the glossary for new or changed project terms, or record why no glossary change is needed;
5. validate links, anchors, examples, terminology, and formatting; and
6. record files reviewed, changes made, commands and results, limitations, and unresolved issues.

The implementation pass owns executable tests and hands their exact evidence to the documentation-focused agent. The documentation pass must not rerun a successful Java test suite unless it changes executable Java behavior after that run, the original evidence is missing or stale, or the task identifies a concrete cross-check risk. Javadoc-only edits do not require Java tests, but final Javadoc generation must occur after those edits.

The task remains incomplete until this pass and its evidence are present in the same overall change.

## Validation and evidence

Use the validation section in the selected profile. At minimum:

- check local Markdown links and anchors;
- verify terminology against the [glossary](../glossary.md);
- verify examples against current behavior or label conceptual examples clearly;
- generate Javadoc for affected Java modules after final Javadoc edits when Java APIs changed;
- run `git diff --check` and check new files for trailing whitespace; and
- review the final diff for accidental authority changes, duplicated rules, and unrelated edits.

Evidence must name the documentation-focused context, identify the selected profile, list the files or topics reviewed, and report exact validation commands and outcomes. A no-change conclusion must include a reason; `N/A` alone is not evidence.
