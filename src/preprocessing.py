from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Create a scikit-learn preprocessing pipeline.
    - OneHotEncode categorical columns.
    - Scale numerical columns (StandardScaler).
    """
    categorical_transformer = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def build_full_pipeline(preprocessor, model):
    """Combine preprocessor and model into a single Pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
