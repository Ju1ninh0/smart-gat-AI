import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "smart_gat.db"


def conectar():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DB_PATH)


def criar_banco():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            vehicle_type TEXT DEFAULT 'Veículo',
            plate TEXT,
            confidence REAL DEFAULT 0,
            camera TEXT DEFAULT 'Câmera 1',
            detected_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def adicionar_deteccao(
    tracking_id,
    vehicle_type="Veículo",
    plate=None,
    confidence=0.0,
    camera="Câmera 1"
):

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections (
            tracking_id,
            vehicle_type,
            plate,
            confidence,
            camera,
            detected_at
        )
        VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
    """, (
        tracking_id,
        vehicle_type,
        plate,
        confidence,
        camera
    ))

    conn.commit()
    conn.close()


def contar_veiculos():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT tracking_id)
        FROM detections
        WHERE tracking_id IS NOT NULL
    """)

    resultado = cursor.fetchone()[0]

    conn.close()

    return resultado


def contar_placas():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM detections
        WHERE plate IS NOT NULL
        AND plate != ''
    """)

    resultado = cursor.fetchone()[0]

    conn.close()

    return resultado


def contar_placas_unicas():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT plate)
        FROM detections
        WHERE plate IS NOT NULL
        AND plate != ''
    """)

    resultado = cursor.fetchone()[0]

    conn.close()

    return resultado


def contar_cameras():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT camera)
        FROM detections
        WHERE camera IS NOT NULL
        AND camera != ''
    """)

    resultado = cursor.fetchone()[0]

    conn.close()

    return resultado


def ultimas_deteccoes(limite=20):

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            tracking_id,
            vehicle_type,
            plate,
            confidence,
            camera,
            detected_at
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    """, (limite,))

    resultado = cursor.fetchall()

    conn.close()

    return resultado


def ultima_deteccao():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            tracking_id,
            vehicle_type,
            plate,
            confidence,
            camera,
            detected_at
        FROM detections
        ORDER BY id DESC
        LIMIT 1
    """)

    resultado = cursor.fetchone()

    conn.close()

    return resultado