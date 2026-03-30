import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class LowercasePackageNamingTest {
    private static final Pattern UPPERCASE_ROOT = Pattern.compile(
            "^(package|import)\\s+(Backend|Benchmark|Config|Graph|Numerics|Operations|Tensor|Utils)(\\.|;)"
    );

    @Test
    void sourceFilesDoNotUseUppercaseRootPackages() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            List<String> lines = Files.readAllLines(path);
                            for (int i = 0; i < lines.size(); i++) {
                                if (UPPERCASE_ROOT.matcher(lines.get(i)).find()) {
                                    return Stream.of(path + ":" + (i + 1) + ": " + lines.get(i).trim());
                                }
                            }
                            return Stream.empty();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Uppercase root package references found: " + offenders);
        }
    }
}
