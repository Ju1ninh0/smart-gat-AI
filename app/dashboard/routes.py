from flask import Blueprint, render_template, jsonify

from .database import (
    contar_veiculos,
    contar_placas,
    contar_placas_unicas,
    contar_cameras,
    ultimas_deteccoes
)

routes = Blueprint("routes", __name__)


@routes.route("/")
def dashboard():

    return render_template(
        "dashboard.html",
        veiculos=contar_veiculos(),
        placas=contar_placas_unicas(),
        cameras=contar_cameras()
    )


@routes.route("/cameras")
def cameras():
    return render_template("cameras.html")


@routes.route("/vehicles")
def vehicles():

    detections = ultimas_deteccoes(50)

    return render_template(
        "vehicles.html",
        detections=detections
    )


@routes.route("/plates")
def plates():

    detections = ultimas_deteccoes(50)

    return render_template(
        "plates.html",
        detections=detections
    )


@routes.route("/api/detections")
def api_detections():

    detections = ultimas_deteccoes(50)

    data = []

    for detection in detections:

        data.append({
            "id": detection[0],
            "tracking_id": detection[1],
            "vehicle_type": detection[2],
            "plate": detection[3],
            "confidence": detection[4],
            "camera": detection[5],
            "detected_at": detection[6]
        })

    return jsonify(data)


@routes.route("/api/stats")
def api_stats():

    return jsonify({
        "veiculos": contar_veiculos(),
        "placas": contar_placas_unicas(),
        "cameras": contar_cameras()
    })