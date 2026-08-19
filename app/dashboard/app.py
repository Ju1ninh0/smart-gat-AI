from flask import Flask
from .routes import routes
from .database import criar_banco

app = Flask(__name__)

app.register_blueprint(routes)

criar_banco()


if __name__ == "__main__":
    app.run(debug=True)