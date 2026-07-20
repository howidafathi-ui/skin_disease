import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # ---- إعدادات عامة ----
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

    # ---- قاعدة البيانات ----
    # SQLite افتراضياً لسهولة التشغيل المحلي، يمكن استبدالها بـ MySQL/PostgreSQL
    # مثال MySQL:  "mysql+pymysql://user:password@localhost/skin_disease_db"
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'skin_disease.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ---- رفع الصور ----
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB كحد أقصى لحجم الصورة

    # ---- إعدادات النموذج الذكي ----
    MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet50_skin_disease.h5")
    IMAGE_SIZE = (224, 224)  # المقاس القياسي لـ ResNet50
    CLASS_NAMES = [
        "Melanocytic Nevi (شامة حميدة)",
        "Melanoma (سرطان الجلد - ميلانوما)",
        "Basal Cell Carcinoma (سرطان الخلايا القاعدية)",
        "Actinic Keratoses (تقرن سفعي)",
        "Benign Keratosis (تقرن حميد)",
        "Dermatofibroma (ورم ليفي جلدي)",
        "Vascular Lesion (آفة وعائية)",
    ]
