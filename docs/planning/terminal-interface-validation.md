# CLI/TUI behavior and terminal qualification

Status: Complete for the exercised macOS arm64 surface. Performance comparisons are separate work.

## Source and supported boundary

The qualified implementation is `575c0e9ef95efb5f3eed24b6bf1911cfdbdd5d75`, following the
[bounded autotuning audit](autotuning-audit.md). The user selected an interactive
[TUI](../glossary.md#term-tui) with retained [CLI](../glossary.md#term-cli), then requested common
[ASCII](../glossary.md#term-ascii)/[ANSI](../glossary.md#term-ansi) branding. The [practical guide](../how-to/train-with-cli.md) supplies the actual launcher,
commands, file format, controls, measurement definitions and ownership explanation.

The local application has real `train`, `predict`, `inspect` and `tui` commands. Training/prediction
admit the stated single-input/single-output static rank-two [FLOAT32](../glossary.md#term-float32)
geometry; inspection reads public saved-source metadata without imposing that computation geometry.
[Model](../reference/nn/owned-model.md) owns private parameters/[TrainingSession](../glossary.md#training-session--trainingsession),
[Engine](../reference/engine/lifecycle.md) owns execution, and the application owns file traversal,
separate [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) buffers and presentation. TUI numerical owners live on one worker; the terminal
thread receives detached measurements and requests cooperative stop only.

This is a separate private local application image, not a nineteenth library publication or
reissue of the sealed eighteen-product release. JDK/numerical native libraries remain external.
JLine terminal/[FFM](../glossary.md#term-ffm)/native artifacts are pinned to **4.5.0** with the complete
[component notices](../../THIRD_PARTY_NOTICES.md). No Model arithmetic, numerical owner, Runtime
search, native Synaptik ABI or inward dependency contract changed.

## Actually executed verification

| Scope | Tests | Failures | Errors | Skips |
|---|---:|---:|---:|---:|
| CLI/TUI behavior, data, numerical, editor, geometry and lifecycle suites | 89 | 0 | 0 | 0 |
| Architecture suites, explicitly forced with task `--rerun` | 25 | 0 | 0 | 0 |

These are affected scopes, not a claim that every repository test or every backend operation ran.
Earlier CLI qualification had 58 passing tests. The final expanded suite includes actual Model
updates/predictions, cancellation races, input/output aliases, nonfinite committed objectives,
Unicode path identity, metadata navigation and retained loss/gap rendering at 80×24.
Incidental metadata-format/record-identity checks were removed rather than re-pinned.

Two independent source reviewers found and then checked repairs to controller and presentation
boundaries. Their final follow-ups found no additional actionable defect. Reviewers ran no tests
or native/terminal checks; the coordinator performed the runs below.

### Real external-directory consumers

The generated `installDist` launcher ran outside the checkout:

- Numeric examples for $y=2x+1$ trained a real saved model for 200 epochs. A fresh invocation
  loaded it and wrote predictions `0.99999857`, `2.9999986`, `4.9999986` for inputs 0, 1, 2.
- The actual TUI keyboard entered training fields, saved the model, returned to the menu,
  predicted to a real CSV, inspected metadata and resized the viewport. Its resulting prediction
  file contained the same three values. No test-owned mock supplied these numbers.
- Explicit Metal and exact oneDNN configurations each completed actual training/save/cold-predict
  runs. This proves those cases, **not** that the one-feature CPU example selected a oneDNN kernel
  or that every operation has native eligibility. Native preference is not route attestation.
- NonTTY training without `--plain` still produced append-only stderr without escapes; stdout
  remained clean. Explicit `tui` without a qualified terminal failed with the prerequisite rather
  than pretending to show an interactive screen.

### Real controlling-terminal interaction

An owned [PTY](../glossary.md#term-pty) driver ran the actual launcher with genuine stdin/stderr terminal descriptors,
not a fake `isTTY` flag. A shell-like supervisor kept the terminal alive after Java exit so
restoration could be measured before session hangup. Raw streams and timestamps were preserved.
The bundled JLine `ScreenTerminal` decoded those captured bytes to inspect actual screen grids;
those grids were not manually laid out from expected values.

Observed scenarios:

- Default redraw intervals around 500 ms; real epoch/step/time/payload counters and actual loss
  points. Resizing 120×32 → 72×24 and a separate 80×24 run produced fitted live displays.
- The 80×24 training screen retained loss history, recent epoch timing, decoded payload rate,
  stop status and bounded events after the epoch-history buffer filled.
- `s` stopped training and retained the result screen. A typed `q` then exited 130; a preexisting
  checkpoint remained byte-for-byte unchanged. A separate stopped run created no output checkpoint.
- Invalid data/path produced a failed job, then an editable form. Correcting the field and
  submitting again completed real training and saved a checkpoint.
- Raw Ctrl-C exited 130 without a hard worker interrupt. `NO_COLOR` removed logo color while
  retaining the control sequences required for a real TUI.
- A native INT during an admitted prediction waited for normal completion, published all
  20,000 prediction rows, and exited 130. The screen/controller did not claim rollback.
- Different actual stdin/stderr TTY devices selected plain progress with no ANSI; they were not
  incorrectly rendered using the input device's dimensions.
- Cursor visibility and alternate screen were restored. CLI's full native terminal attributes
  matched before/after with the corrected provider. After a TUI [raw](../glossary.md#term-raw-terminal-mode)→[canonical](../glossary.md#term-canonical-terminal-mode) transition the only
  native difference was Darwin `PENDIN` (`0x20000000`), documented by the installed SDK as pending
  input-retype **state**; all other flags, speeds and control characters matched. This is not
  described as byte-identical kernel state.

## Findings repaired before closure

| Finding | Evidence boundary | Repair and exercised result |
|---|---|---|
| Pinned FFM conversion discarded native terminal fields | Actual 3.30.6 terminal run changed both speeds from 9600 to 0 and reset spare control slots even for non-raw progress; independent pinned-source trace confirmed zeroed reconstruction. | Upgrade to 4.5.0's read-before-apply conversion. Actual CLI restores every field; TUI restores configuration with only the documented kernel PENDIN state difference. No reflection, copied native layout, new binding or skipped close. |
| Native ANSI logo failed on the supplied xterm foreground template | Actual run failed with `IOError: NoSuchElementException`; an isolated real-terminal probe traced `Curses` conditional-stack underflow. | Use JLine's attributed ANSI renderer rather than manually expanding conditional terminfo per character. Actual colored CLI/TUI runs succeed; NO_COLOR is exercised. No exception suppression or fake color fallback. |
| Rendering Error could bypass cleanup or a delivered failure could suppress itself | Source review plus lifecycle regressions distinguish RuntimeException, IOError, training primary and cleanup failure identities. | Retain unchecked failures, attempt all owned cleanup, stop future admissions on known presentation failure and never rethrow the same delivered primary from resource closure. Broken diagnostics cannot publish a successful model. |
| Late interrupt lost its status or safe stop was labeled failure | Independent source findings in the consuming controller/view; not a claimed baseline native reproduction. | Preserve interrupt-exit precedence and show cancellation separately. Actual admitted prediction exits 130 after full atomic publication; stopped training shows CANCELLED and no pre-save publication. |
| Hidden metadata, narrowed format-character paths and lost narrow-screen loss plot | Independent source findings and an inconsistent refresh-sampling fixture. | Wrapped, cached, keyboard-scrollable inspection; exact stored paths with sanitized display copies; adaptive plot budget before optional epoch rows; sample the fixture at real refresh boundaries. Behavioral regressions and real resized screens exercise these paths. |

The upstream [4.5.0 FFM setter](https://github.com/jline/jline3/blob/4.5.0/terminal-ffm/src/main/java/org/jline/terminal/impl/ffm/CLibrary.java)
reads existing `termios` before applying supported attributes. The
[release and component changes](https://github.com/jline/jline3/releases/tag/4.5.0) are external
terminal-library changes, not changes to numerical Synaptik architecture. Automatic grapheme
probing is disabled; the application owns raw control-key handling and restores the original
per-terminal mappings instead of changing process-global signal properties.

## Evidence and limitations

Commands, raw terminal streams, decoded screens, XML totals, independent reviews and sources of
owned diagnostic programs are under `build/tuning-cli-performance/`. The pre-fix local application
image is preserved as `cli-before-ffm-fix-image.zip`. Actual hardware is Apple M3 Max, macOS arm64,
JDK 26; exact external native paths/hashes belong to the recorded receipts. No data or artifacts
were publicly uploaded.

Live FFM rendering remains qualified only for the checked POSIX/matching-terminal contract;
Windows uses plain progress, and Linux/other terminal emulators are not claimed as exercised.
No guarantee is made after SIGKILL, VM failure or terminal disconnection. A completed save cannot
be rolled back by a later interrupt or presentation/cleanup failure. Model files preserve current
graph/values, not optimizer history or exact training continuation. A finite example or small
training loss is not broad numerical qualification or performance evidence.
