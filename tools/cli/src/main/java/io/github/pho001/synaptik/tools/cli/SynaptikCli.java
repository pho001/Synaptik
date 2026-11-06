package io.github.pho001.synaptik.tools.cli;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Objects;

/**
 * Numeric regression command-line application using public Synaptik Model owners.
 * Data/results/help go to stdout and training presentation/diagnostics go to stderr.
 */
public final class SynaptikCli {
    private SynaptikCli() { }

    /**
     * Runs the application and exits with its conventional command status.
     * @param args train, predict, inspect, tui or help followed by command-local long options
     */
    public static void main(String[] args) {
        System.exit(run(args, System.out, System.err));
    }

    /**
     * Runs one command synchronously without closing the caller's streams or exiting the JVM.
     * Calls must not overlap: live presentation temporarily owns the process terminal/INT handler.
     * @param args non-null command/options; filenames are real filesystem paths
     * @param output non-null borrowed result stream, including CSV predictions when no output path is given
     * @param error non-null borrowed diagnostic/progress stream
     * @return zero on success/help, 2 for option/data/signature contract errors, 1 for I/O/execution/cleanup
     *     failures, or 130 after cooperative training cancellation; already completed calls are not undone
     * @throws NullPointerException if an argument is null
     */
    public static int run(String[] args, PrintStream output, PrintStream error) {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(error, "error");
        try {
            CliArguments options = CliArguments.parse(args);
            int status = switch (options.command()) {
                case "help" -> {
                    output.print(HELP);
                    yield 0;
                }
                case "train" -> CsvRegressionWorkflow.train(options, error);
                case "predict" -> CsvRegressionWorkflow.predict(options, output);
                case "tui" -> SynaptikTui.run(error);
                case "inspect" -> {
                    CliModel.inspect(CliModel.load(options.requiredPath("model")), output);
                    yield 0;
                }
                default -> throw new AssertionError("unhandled parsed command");
            };
            output.flush();
            error.flush();
            return output.checkError() || error.checkError() ? 1 : status;
        } catch (IllegalArgumentException failure) {
            report(error, failure);
            return error.checkError() ? 1 : 2;
        } catch (IOException | RuntimeException | LinkageError | java.io.IOError failure) {
            report(error, failure);
            return 1;
        }
    }

    private static void report(PrintStream error, Throwable failure) {
        error.println("synaptik: " + failure.getClass().getSimpleName() + ": " + failure.getMessage());
        for (Throwable suppressed : failure.getSuppressed()) {
            error.println("  cleanup: " + suppressed.getClass().getSimpleName() + ": " + suppressed.getMessage());
        }
        error.flush();
    }

    private static final String HELP = """
            Usage:
              synaptik train --data numeric.csv --output trained.synaptik [options]
              synaptik predict --model trained.synaptik --data features.csv [--output predictions.csv] [backend options]
              synaptik inspect --model trained.synaptik
              synaptik tui
              synaptik help | --help | <command> --help

            Interactive workbench:
              tui requires a supported local stdin/stderr TTY; use train/predict/inspect otherwise.
              Menu/forms start the same workflows. Training stop is cooperative; prediction/inspection
              finish admitted work before back/exit. Prediction requires an output file in the TUI.

            Train options:
              --model PATH          Start fresh training from saved graph/current weights (not optimizer resume)
              --epochs N            Positive epoch count (default 20)
              --batch-size B        New static batch size (default 1); for a loaded model, must match if supplied
              --outputs O           Final O CSV columns are targets (new model default 1); loaded width must match
              --seed N              New biased Linear model's deterministic initialization seed (default 0)
              --optimizer sgd|adam  Public optimizer (default sgd); --learning-rate positive finite FLOAT32
                                    SGD default 0.01; Adam default 0.001, beta1=.9, beta2=.999, epsilon=1e-8, decay=0
              --refresh-ms N        Positive presentation interval in milliseconds (default 500)
              --plain               Append-only stderr logs, no terminal controls

            Backend options (train/predict only):
              --backend cpu         Default Engine.standard CPU composition
              --backend metal --metal-library /absolute/path/to/bridge
                                    Metal-only Engine, no CPU fallback
              --backend onednn --onednn-library /absolute/path/to/library
                  [--onednn-api auto|primitives|graph]
                                    Exact oneDNN 3.14.0 LP64 SEQ library required; native API preference,
                                    not proof a native provider wins. Ordinary CPU eligibility remains owner-defined.

            Data and model boundaries:
              UTF-8 finite decimal FLOAT32 comma-separated values, one example per nonblank row.
              No header, quotes, empty fields, NaN or infinity; surrounding field whitespace is allowed.
              Training rows: F features then O targets. Prediction rows: F features only.
              The entire CSV is validated/counted before execution. Keep its bytes stable throughout the command.
              New models are biased F32 rank-two Linear [B,F] -> [B,O]. Saved models require one declared
              static, positive rank-two F32 input and one rank-two F32 output with the same B.
              Complete input descriptors (including requiresGrad and layout presence) are retained. Inputs
              may have absent or canonical contiguous offset-zero non-view layout. Absence is preserved,
              not silently resolved: a backend may reject it (the current CPU MUL path does).
              Output values are detached dense rows. Other graph/AD/backend capability failures are reported,
              never replaced by an alternate model. Inspect shows public descriptors/leaves without an Engine.
              Static B>1 requires row count exactly divisible by B, for both train and predict. No dropped,
              repeated, padded or masked tails. Default B=1 supports every positive row count.
              Loaded graph is decoded once, then privately acquired with the public Model.graph cold copy.
              MSE uses MEAN over B*O. Displayed loss is actual pre-update objective; an epoch mean follows
              changing parameters, not a final-model evaluation. Nonfinite successful objectives remain visible.
              Payload Mbit/s counts 4*B*(F+O) decoded training input+target bytes per completed successful step,
              not CSV/disk/network/device bandwidth. Progress is stderr; prediction stdout stays CSV-only.
              Ctrl-C requests cooperative stop after the current admitted call: no next step and status 130
              when signal registration succeeds. Cancellation observed before save admission prevents saving.
              An admitted save may finish: late cancellation reports whether the trained checkpoint was saved.
              Final outcome is admitted after numerical cleanup; later presentation failures cannot undo a save.
              No hard cancellation or rollback of completed calls/publication is promised.
              Model files save only graph/current weights, never optimizer history or an exact-resume cursor.
              Output parent directories must exist. Prediction files publish atomically on complete success;
              stdout streams rows and cannot retract earlier rows after a later failure. No native/JDK bundles.
            """;
}
