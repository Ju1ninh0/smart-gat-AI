import threading
import time
import webbrowser

from app.dashboard.app import app
from app.ai.vehicle_detector import start_detection


def start_dashboard():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        daemon=True
    )

    dashboard_thread.start()

    time.sleep(2)

    webbrowser.open(
        "http://127.0.0.1:5000"
    )

    start_detection()