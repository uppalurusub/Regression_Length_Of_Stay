
import mlflow
from mlflow import MlflowClient
from app.config.settings import settings
logger = settings.get_logger()


TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "hospital-los-regressor"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)


def inspect_registry():
    try:
        logger.info("Checking MLflow registry: {}", TRACKING_URI)

        registered_model = client.get_registered_model(MODEL_NAME)

        logger.info("Registered model: {}", registered_model.name)

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            logger.warning("No model versions found.")
            return

        for version in versions:
            logger.info(
                "Model version={} | status={} | run_id={}",
                version.version,
                version.status,
                version.run_id,
            )

        try:
            champion = client.get_model_version_by_alias(
                MODEL_NAME,
                MODEL_ALIAS,
            )

            logger.success(
                "Champion alias exists | version={}",
                champion.version,
            )

        except Exception as exc:
            logger.error(
                "Alias '{}' does not exist or cannot be resolved: {}",
                MODEL_ALIAS,
                exc,
            )

    except Exception:
        logger.exception("Registry inspection failed.")
        raise


if __name__ == "__main__":
    inspect_registry()
