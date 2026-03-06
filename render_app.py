import os

from dash_app import app


server = app.server


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8060"))
    app.run(host="0.0.0.0", port=port, debug=False)
