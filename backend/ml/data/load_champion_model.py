import mlflow.pyfunc

from app.config.settings import settings
logger = settings.get_logger()


def load_champion_model():

    model_uri = (
        "models:/hospital-los-regressor@champion"
    )

    try:
        model = mlflow.pyfunc.load_model(model_uri)

        logger.success(
            "Champion model loaded | model_uri={}",
            model_uri,
        )

        return model

    except Exception:
        logger.exception(
            "Failed to load champion model | model_uri={}",
            model_uri,
        )
        raise