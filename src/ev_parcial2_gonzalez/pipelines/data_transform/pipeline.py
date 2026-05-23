from kedro.pipeline import Pipeline, node, pipeline
from .nodes import merge_datasets, feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=[
                    "cleaned_employee_data",
                    "cleaned_engagement_data",
                    "cleaned_recruitment_data",
                    "cleaned_training_data"
                ],
                outputs="merged_data",
                name="merge_datasets_node",
            ),
            node(
                func=feature_engineering,
                inputs="merged_data",
                outputs="master_table",
                name="feature_engineering_node",
            ),
        ]
    )
