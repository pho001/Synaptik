package io.github.pho001.synaptik.tools.cli;

import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Prediction file publication is atomic; borrowed stdout is streamed and never closed. */
final class PredictionOutput implements AutoCloseable {
    private final PrintStream stdout;
    private final Path destination;
    private final Path temporary;
    private BufferedWriter writer;
    private boolean committed;

    PredictionOutput(Path destination, PrintStream stdout) throws IOException {
        this.destination = destination;
        this.stdout = stdout;
        if (destination == null) {
            temporary = null;
        } else {
            temporary = Files.createTempFile(destination.getParent(), ".synaptik-predict-", ".csv");
            try {
                writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8);
            } catch (IOException | RuntimeException | Error failure) {
                try {
                    Files.deleteIfExists(temporary);
                } catch (IOException cleanup) {
                    failure.addSuppressed(cleanup);
                }
                throw failure;
            }
        }
    }

    void batch(HostTensorValue prediction, CliModel.Signature signature) throws IOException {
        if (prediction.dataType() != DataType.FLOAT32
                || !prediction.shape().equals(signature.prediction().shape())
                || prediction.elementCount() != (long) signature.batch() * signature.outputs()) {
            throw new IllegalStateException("Model returned an incompatible prediction descriptor");
        }
        ByteBuffer values = prediction.bytes();
        StringBuilder row = new StringBuilder();
        for (int r = 0; r < signature.batch(); r++) {
            row.setLength(0);
            for (int c = 0; c < signature.outputs(); c++) {
                if (c != 0) row.append(',');
                row.append(Float.toString(values.getFloat()));
            }
            if (writer != null) {
                writer.write(row.toString());
                writer.newLine();
            } else {
                stdout.println(row);
                if (stdout.checkError()) throw new IOException("prediction stdout write failed");
            }
        }
    }

    /**
     * Publishes only after all rows and source-content validation complete successfully.
     * @throws IOException for output failure; there is no non-atomic file replacement fallback
     */
    void commit() throws IOException {
        if (writer != null) {
            writer.close();
            writer = null;
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } else {
            stdout.flush();
            if (stdout.checkError()) throw new IOException("prediction stdout write failed");
        }
        committed = true;
    }

    @Override
    public void close() throws IOException {
        IOException failure = null;
        try {
            if (writer != null) writer.close();
        } catch (IOException closeFailure) {
            failure = closeFailure;
        }
        if (temporary != null && !committed) {
            try {
                Files.deleteIfExists(temporary);
            } catch (IOException deleteFailure) {
                if (failure == null) failure = deleteFailure;
                else failure.addSuppressed(deleteFailure);
            }
        }
        if (failure != null) throw failure;
    }
}
