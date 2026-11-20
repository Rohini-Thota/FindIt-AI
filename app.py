
import os
import re
import uuid
import sqlite3
import smtplib
import logging

from pathlib import Path
from contextlib import contextmanager
from email.message import EmailMessage

import cv2
from dotenv import load_dotenv

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify
)

from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend"),
    static_folder=str(BASE_DIR / "static")
)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "finditai-development-key"
)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
DATABASE_FOLDER = BASE_DIR / "database"
DATABASE_PATH = DATABASE_FOLDER / "finditai.db"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Standard pretrained YOLOv8 model
MODEL_PATH = BASE_DIR / "yolov8n.pt"

# COCO classes supported by yolov8n.pt
ALLOWED_OBJECTS = {
    "bottle",
    "backpack",
    "handbag",
    "suitcase",
    "laptop",
    "cell phone",
    "keyboard",
    "mouse",
    "book",
    "umbrella",
    "sports ball",
    "scissors",
    "remote",
    "clock",
    "tie",
    "cup",
    "bowl",
    "vase"
}

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
MAX_DETECTIONS = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FindItAI")


# ==================================================
# LOAD YOLO MODEL
# ==================================================

model = None
OBJECT_CATEGORY_MAP = {
    "bottle": "Bottle",
    "handbag": "Handbag",
    "backpack": "Handbag",
    "suitcase": "Suitcase",
    "laptop": "Laptop",
    "cell phone": "Phone",
    "phone": "Phone",
    "book": "Other",
    "clock": "Other",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "umbrella": "Other",
    "sports ball": "Other"
}
ALLOWED_CLASS_IDS = []

if YOLO is not None and MODEL_PATH.exists():

    try:
        model = YOLO(str(MODEL_PATH))

        # Find the actual class IDs supported by this model.
        ALLOWED_CLASS_IDS = [
            class_id
            for class_id, class_name in model.names.items()
            if str(class_name).lower() in ALLOWED_OBJECTS
        ]

        logger.info("YOLOv8 model loaded successfully.")
        logger.info("Model path: %s", MODEL_PATH)
        logger.info(
            "Allowed classes: %s",
            [model.names[i] for i in ALLOWED_CLASS_IDS]
        )

    except Exception:
        logger.exception("Could not load YOLO model.")
        model = None

else:
    logger.warning(
        "yolov8n.pt not found or Ultralytics unavailable."
    )


# ==================================================
# COMMON OBJECT DETECTION FUNCTION
# ==================================================

def detect_objects(image_path):

    if model is None:
        raise RuntimeError("YOLO model is unavailable.")

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=ALLOWED_CLASS_IDS,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Plot only the detections returned after class filtering.
    annotated_image = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detections.append({
                "name": object_name,
                "confidence": round(confidence * 100, 2)
            })

    return annotated_image, detections


# ==================================================
# DATABASE
# ==================================================

@contextmanager
def get_db():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database():

    with get_db() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                item_name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                location TEXT NOT NULL,
                report_date TEXT NOT NULL,
                contact TEXT NOT NULL DEFAULT '',
                email TEXT,
                phone TEXT,
                image_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(reports)"
            ).fetchall()
        }

        if "email" not in existing_columns:
            connection.execute(
                "ALTER TABLE reports ADD COLUMN email TEXT"
            )

        if "phone" not in existing_columns:
            connection.execute(
                "ALTER TABLE reports ADD COLUMN phone TEXT"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lost_report_id INTEGER NOT NULL,
                found_report_id INTEGER NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lost_report_id, found_report_id, channel)
            )
        """)


initialize_database()


# ==================================================
# FILE HELPERS
# ==================================================

def allowed_file(filename):

    return (
        Path(filename).suffix.lower()
        in ALLOWED_EXTENSIONS
    )


def save_uploaded_image(file):

    if not file or not file.filename:
        return None

    if not allowed_file(file.filename):
        return None

    safe_name = secure_filename(file.filename)

    extension = Path(safe_name).suffix.lower()

    filename = f"{uuid.uuid4().hex}{extension}"

    file.save(UPLOAD_FOLDER / filename)

    return filename


# ==================================================
# CONTACT HELPERS
# ==================================================

def get_contact_details(form):

    email = form.get("email", "").strip()
    phone = form.get("phone", "").strip()
    contact = form.get("contact", "").strip()

    if contact and not email and not phone:

        if "@" in contact:
            email = contact
        else:
            phone = contact

    return email, phone, contact


# ==================================================
# EMAIL NOTIFICATIONS
# ==================================================

def send_email_notification(receiver, item_name, location):

    smtp_server = os.getenv(
        "SMTP_SERVER",
        "smtp.gmail.com"
    )

    smtp_port = int(
        os.getenv("SMTP_PORT", "587")
    )

    sender_email = (
        os.getenv("SMTP_USERNAME")
        or os.getenv("SENDER_EMAIL")
    )

    sender_password = (
        os.getenv("SMTP_PASSWORD")
        or os.getenv("SENDER_PASSWORD")
    )

    if not all([
        sender_email,
        sender_password,
        receiver
    ]):
        logger.warning("SMTP configuration incomplete.")
        return False

    message = EmailMessage()

    message["Subject"] = (
        f"Possible match found for your lost item: {item_name}"
    )

    message["From"] = sender_email
    message["To"] = receiver

    message.set_content(f"""
Hello,

Good news!

A possible match has been found for the item
you reported as lost.

Item: {item_name}
Found location: {location}

Please visit FindIt AI and check the community reports.

Please verify the item carefully before claiming it.

Regards,
FindIt AI - Lost and Found Portal
""")

    try:

        with smtplib.SMTP(
            smtp_server,
            smtp_port,
            timeout=20
        ) as server:

            server.starttls()

            server.login(
                sender_email,
                sender_password
            )

            server.send_message(message)

        logger.info("Match email sent to %s", receiver)

        return True

    except Exception:

        logger.exception("Email notification failed.")

        return False


# ==================================================
# SMS NOTIFICATIONS
# ==================================================

def send_sms_notification(receiver, item_name, location):

    logger.info(
        "SMS disabled. Email notification is used instead."
    )

    return False


# ==================================================
# ITEM MATCHING
# ==================================================

def normalize_text(value):

    return re.findall(
        r"[a-z0-9]+",
        (value or "").lower()
    )


def calculate_name_similarity(first, second):

    first_words = set(normalize_text(first))
    second_words = set(normalize_text(second))

    if not first_words or not second_words:
        return 0

    intersection = len(first_words & second_words)
    union = len(first_words | second_words)

    return intersection / union


def is_possible_match(lost_report, found_report):

    lost_category = (
        lost_report["category"] or ""
    ).strip().lower()

    found_category = (
        found_report["category"] or ""
    ).strip().lower()

    category_match = bool(
        lost_category
        and found_category
        and lost_category == found_category
    )

    name_score = calculate_name_similarity(
        lost_report["item_name"],
        found_report["item_name"]
    )

    return category_match or name_score >= 0.5


# ==================================================
# MATCH NOTIFICATIONS
# ==================================================

def notify_matching_users(found_report):

    with get_db() as connection:

        lost_reports = connection.execute("""
            SELECT *
            FROM reports
            WHERE LOWER(report_type) = 'lost'
            ORDER BY created_at DESC
        """).fetchall()

    for lost_report in lost_reports:

        if not is_possible_match(
            lost_report,
            found_report
        ):
            continue

        email = (
            lost_report["email"]
            or (
                lost_report["contact"]
                if "@" in (lost_report["contact"] or "")
                else ""
            )
        )

        phone = (
            lost_report["phone"]
            or (
                lost_report["contact"]
                if "@" not in (lost_report["contact"] or "")
                else ""
            )
        )

        if email:

            with get_db() as connection:

                already_sent = connection.execute("""
                    SELECT id FROM notifications
                    WHERE lost_report_id = ?
                    AND found_report_id = ?
                    AND channel = 'email'
                """, (
                    lost_report["id"],
                    found_report["id"]
                )).fetchone()

            if not already_sent:

                success = send_email_notification(
                    email,
                    found_report["item_name"],
                    found_report["location"]
                )

                with get_db() as connection:

                    connection.execute("""
                        INSERT OR IGNORE INTO notifications
                        (lost_report_id, found_report_id, channel, status)
                        VALUES (?, ?, 'email', ?)
                    """, (
                        lost_report["id"],
                        found_report["id"],
                        "sent" if success else "failed"
                    ))

        if phone:

            with get_db() as connection:

                already_sent = connection.execute("""
                    SELECT id FROM notifications
                    WHERE lost_report_id = ?
                    AND found_report_id = ?
                    AND channel = 'sms'
                """, (
                    lost_report["id"],
                    found_report["id"]
                )).fetchone()

            if not already_sent:

                with get_db() as connection:

                    connection.execute("""
                        INSERT OR IGNORE INTO notifications
                        (lost_report_id, found_report_id, channel, status)
                        VALUES (?, ?, 'sms', 'disabled')
                    """, (
                        lost_report["id"],
                        found_report["id"]
                    ))


# ==================================================
# HOME + IMAGE UPLOAD DETECTION
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    detections = []

    if request.method == "POST":

        if model is None:

            flash(
                "YOLO model unavailable. Check yolov8n.pt."
            )

            return redirect(url_for("index"))

        file = request.files.get("image")

        if not file or not file.filename:

            flash("Please select an image.")

            return redirect(url_for("index"))

        if not allowed_file(file.filename):

            flash(
                "Upload JPG, JPEG, PNG or WEBP images."
            )

            return redirect(url_for("index"))

        filename = save_uploaded_image(file)

        filepath = UPLOAD_FOLDER / filename

        try:

            annotated, detections = detect_objects(filepath)

            output_filename = (
                f"result_{uuid.uuid4().hex}.jpg"
            )

            saved = cv2.imwrite(
                str(UPLOAD_FOLDER / output_filename),
                annotated
            )

            if not saved:
                raise RuntimeError(
                    "Could not save detection result."
                )

            result_image = output_filename

        except Exception:

            logger.exception("Image detection failed.")

            flash(
                "Image detection failed. Please try again."
            )

    return render_template(
        "index.html",
        result_image=result_image,
        detections=detections
    )


# ==================================================
# REPORT LOST / FOUND ITEMS
# ==================================================

@app.route("/report/<report_type>", methods=["GET", "POST"])
def report_item(report_type):

    if report_type not in ("lost", "found"):

        return redirect(url_for("index"))

    if request.method == "POST":

        item_name = request.form.get(
            "item_name", ""
        ).strip()

        category = request.form.get(
            "category", ""
        ).strip()

        description = request.form.get(
            "description", ""
        ).strip()

        location = request.form.get(
            "location", ""
        ).strip()

        report_date = request.form.get(
            "report_date", ""
        ).strip()

        email, phone, contact = get_contact_details(
            request.form
        )

        image = request.files.get("image")

        if not all([
            item_name,
            location,
            report_date
        ]):

            flash("Please fill in all required fields.")

            return redirect(
                url_for(
                    "report_item",
                    report_type=report_type
                )
            )

        if not (email or phone):

            flash(
                "Please provide an email address or phone number."
            )

            return redirect(
                url_for(
                    "report_item",
                    report_type=report_type
                )
            )

        image_filename = None

        if image and image.filename:

            if not allowed_file(image.filename):

                flash(
                    "Image must be JPG, JPEG, PNG or WEBP."
                )

                return redirect(
                    url_for(
                        "report_item",
                        report_type=report_type
                    )
                )

            image_filename = save_uploaded_image(image)

        with get_db() as connection:

            cursor = connection.execute("""
                INSERT INTO reports (
                    report_type,
                    item_name,
                    category,
                    description,
                    location,
                    report_date,
                    contact,
                    email,
                    phone,
                    image_filename
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_type,
                item_name,
                category,
                description,
                location,
                report_date,
                contact or email or phone,
                email,
                phone,
                image_filename
            ))

            report_id = cursor.lastrowid

        if report_type == "found":

            with get_db() as connection:

                found_report = connection.execute("""
                    SELECT * FROM reports WHERE id = ?
                """, (report_id,)).fetchone()

            if found_report:
                notify_matching_users(found_report)

        flash(
            f"Your {report_type} item report was submitted successfully!"
        )

        return redirect(url_for("view_reports"))

    return render_template(
        "report_form.html",
        report_type=report_type
    )


# ==================================================
# CAMERA DETECTION
# ==================================================

@app.route("/camera-detect", methods=["POST"])
def camera_detect():

    if model is None:

        return jsonify({
            "success": False,
            "message": "YOLO model is unavailable."
        }), 503

    image = request.files.get("image")

    if not image or not image.filename:

        return jsonify({
            "success": False,
            "message": "No camera image received."
        }), 400

    if not allowed_file(image.filename):

        return jsonify({
            "success": False,
            "message": "Unsupported image format."
        }), 400

    filename = save_uploaded_image(image)

    if not filename:

        return jsonify({
            "success": False,
            "message": "Image upload failed."
        }), 400

    filepath = UPLOAD_FOLDER / filename

    try:

        annotated, detections = detect_objects(filepath)

        output_filename = (
            f"camera_result_{uuid.uuid4().hex}.jpg"
        )

        saved = cv2.imwrite(
            str(UPLOAD_FOLDER / output_filename),
            annotated
        )

        if not saved:

            raise RuntimeError(
                "Could not save camera detection image."
            )

        return jsonify({

            "success": True,

            "result_image": url_for(
                "static",
                filename=f"uploads/{output_filename}"
            ),

            "detections": detections,

            "count": len(detections)

        })

    except Exception:

        logger.exception("Camera detection failed.")

        return jsonify({
            "success": False,
            "message": "Object detection failed. Please try again."
        }), 500


# ==================================================
# VIEW ALL REPORTS
# ==================================================

@app.route("/reports")
def view_reports():

    with get_db() as connection:

        reports = connection.execute("""
            SELECT *
            FROM reports
            ORDER BY created_at DESC
        """).fetchall()

    return render_template(
        "reports.html",
        reports=reports
    )


# ==================================================
# NOTIFICATION HISTORY
# ==================================================

@app.route("/notifications")
def view_notifications():

    with get_db() as connection:

        notifications = connection.execute("""
            SELECT
                n.id,
                n.lost_report_id,
                n.found_report_id,
                n.channel,
                n.status,
                n.created_at,
                lost.item_name AS lost_item_name,
                found.item_name AS found_item_name

            FROM notifications n

            LEFT JOIN reports lost
                ON n.lost_report_id = lost.id

            LEFT JOIN reports found
                ON n.found_report_id = found.id

            ORDER BY n.created_at DESC
        """).fetchall()

    return render_template(
        "notifications.html",
        notifications=notifications
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
