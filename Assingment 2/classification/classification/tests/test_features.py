from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    extracter = ExtractLetterTransformer(
        variables=config.model_config.cabin_var_imputation
    )

    SAMPLE_INDEX = 14
    SAMPLE_VALUE = "B"
    SAMPLE_RAW_VALUE = "B57"
    FEATURE = "cabin"

    assert sample_input_data[FEATURE].iat[SAMPLE_INDEX] == SAMPLE_RAW_VALUE

    # When
    subject = extracter.fit_transform(sample_input_data)

    # Then
    assert subject[FEATURE].iat[SAMPLE_INDEX] == SAMPLE_VALUE
