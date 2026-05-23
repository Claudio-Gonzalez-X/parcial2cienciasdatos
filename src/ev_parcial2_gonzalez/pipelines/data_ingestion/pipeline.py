from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_diagnostic_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_diagnostic_report,
                inputs=[
                    "employee_data",
                    "employee_engagement_survey_data",
                    "recruitment_data",
                    "training_and_development_data"
                ],
                outputs="initial_diagnostic_report",
                name="generate_diagnostic_report_node",
            ),
        ]
    )
