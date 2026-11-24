
import mlflow
from mlflow import MlflowClient
from app.config.settings import settings
logger = settings.get_logger()


TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "hospital-los-regressor"
MODEL_VERSION = "1"
MODEL_ALIAS = "champion"

def promote_model():

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    try:
        # Verify the registered model version
        version = client.get_model_version(
            name=MODEL_NAME,
            version=MODEL_VERSION,
        )

        if version.status != "READY":
            raise RuntimeError(
                f"Model version is not READY: {version.status}"
            )

        # Assign champion alias
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
            version=MODEL_VERSION,
        )

        # Verify promotion
        promoted = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
        )

        logger.success(
            "Model promoted successfully | model={} | version={} | alias={}",
            MODEL_NAME,
            promoted.version,
            MODEL_ALIAS,
        )

    except Exception:
        logger.exception("Model promotion failed.")
        raise


if __name__ == "__main__":
    promote_model()
