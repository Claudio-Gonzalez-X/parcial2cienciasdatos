from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_post_transformation

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_post_transformation,
                inputs=["employee_data", "master_table"],
                outputs="quality_report",
                name="validate_post_transformation_node",
            ),
        ]
    )
