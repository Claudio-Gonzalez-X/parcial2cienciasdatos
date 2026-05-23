from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data_node,
                inputs=["employee_data", "params:cleaning"],
                outputs="cleaned_employee_data",
                name="clean_employee_data_node",
            ),
            node(
                func=clean_data_node,
                inputs=["employee_engagement_survey_data", "params:cleaning"],
                outputs="cleaned_engagement_data",
                name="clean_engagement_data_node",
            ),
            node(
                func=clean_data_node,
                inputs=["recruitment_data", "params:cleaning"],
                outputs="cleaned_recruitment_data",
                name="clean_recruitment_data_node",
            ),
            node(
                func=clean_data_node,
                inputs=["training_and_development_data", "params:cleaning"],
                outputs="cleaned_training_data",
                name="clean_training_data_node",
            ),
        ]
    )
