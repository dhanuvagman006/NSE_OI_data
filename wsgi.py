from core import app, Config
from waitress import serve


if __name__ == "__main__":
    serve(
        app,
        host=Config.HOST,
        port=Config.PORT,
        threads=4,
    )
