# Developer guide style

## Purpose

A developer guide teaches a contributor how a Synaptik concept or internal workflow works well enough to use, debug, or extend it safely. It is instructional, not merely a catalog of classes.

Apply [General style](general-style.md) and the [example format](example-format.md) alongside this profile.

## Required content

Every substantive developer guide must include:

- **What the reader will learn:** concrete capabilities or understanding gained.
- **Prerequisites:** required repository knowledge, tools, completed setup, and linked background.
- **Terms:** short definitions for the concepts needed in the guide, with glossary links.
- **Mental model:** the components, ownership, sequence, or data flow before implementation detail.
- **Complete example:** enough context to follow the whole concept, not an isolated unexplained fragment.
- **Meaningful-line explanation:** explain every line or group of lines that changes meaning, state, ownership, or output. Do not mechanically narrate braces, imports, or repeated syntax.
- **Concrete inputs:** exact values, configuration, initial state, and assumptions.
- **Intermediate results:** important state or values after each meaningful transformation.
- **Final result and interpretation:** what was produced and what the result tells the reader.
- **Typical mistakes:** likely misconceptions, symptoms, causes, and corrections.
- **Limitations:** unsupported cases, boundaries, lifecycle constraints, and deferred behavior.
- **Related links:** authoritative architecture, API contracts, glossary entries, and adjacent guides.

When the concept performs a calculation or transforms data, include a concrete numerical walkthrough. Show the input values, calculation or transformation, intermediate values, final value, and interpretation. Do not force arithmetic into topics where it adds no understanding, such as repository navigation or a decision-record workflow; use a concrete nonmathematical scenario instead.

## Avoid

- source-code tours that list files without explaining responsibility;
- fragments that omit setup, inputs, output, or interpretation;
- line-by-line narration of syntax with no semantic effect;
- replacing the architecture contract with tutorial prose;
- presenting a future API as implemented;
- hiding failure paths or prerequisites; and
- artificial numerical examples for concepts that do not involve computation or data transformation.

## Validation

- Follow one newcomer through the guide from prerequisites to interpreted result.
- Verify the example against current code and tests, or label it conceptual.
- Confirm every meaningful example line is explained.
- For computation or data transformation, recalculate the numerical walkthrough independently.
- Check that mistakes and limitations correspond to real boundaries.
- Validate all links and glossary references.

## Template

```markdown
# <Concept or workflow>

## What you will learn

## Prerequisites

## Terms

## Mental model

## Complete example

### Inputs and initial state

### Code or procedure

### What each meaningful line does

### Intermediate results

### Final result and interpretation

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|

## Limitations and boundaries

## Related documentation
```
