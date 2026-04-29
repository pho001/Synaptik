package config.profile;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WorkloadProfilePresetTest {
    @Test
    void transformerPresetsExposeStableNamesAndDimensions() {
        assertEquals("medium", WorkloadProfile.transformerHotPathMedium().transformerPresetName());
        assertEquals("large", WorkloadProfile.transformerHotPathLarge().transformerPresetName());
        assertEquals("long_seq", WorkloadProfile.transformerHotPathLongSeq().transformerPresetName());
        assertEquals("ffn_heavy", WorkloadProfile.transformerHotPathFfnHeavy().transformerPresetName());
        assertEquals("attention_heavy", WorkloadProfile.transformerHotPathAttentionHeavy().transformerPresetName());

        WorkloadProfile large = WorkloadProfile.transformerHotPathLarge();
        assertEquals(8, large.batch());
        assertEquals(12, large.heads());
        assertEquals(256, large.seqLen());
        assertEquals(64, large.headDim());
        assertEquals(64, large.valueDim());
        assertEquals(3072, large.ffHiddenDim());
        assertEquals(768, large.modelDim());
    }
}
