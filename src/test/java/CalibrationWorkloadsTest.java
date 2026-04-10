import org.junit.jupiter.api.Test;
import tuning.workload.CalibrationWorkloads;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class CalibrationWorkloadsTest {
    @Test
    void defaultCalibrationCatalogContainsExpectedFamilies() {
        var catalog = CalibrationWorkloads.defaultCatalog();

        assertTrue(catalog.names().contains("calib_matmul_square"));
        assertTrue(catalog.names().contains("calib_matmul_tall_skinny"));
        assertTrue(catalog.names().contains("calib_matmul_attention_like"));
        assertTrue(catalog.names().contains("calib_fused_cheap"));
        assertTrue(catalog.names().contains("calib_fused_transcendental"));
        assertTrue(catalog.names().contains("calib_reduction_sum"));
        assertTrue(catalog.names().contains("calib_scheduler_cheap"));
        assertTrue(catalog.names().contains("calib_conv2d_resnet_3x3"));
    }
}
