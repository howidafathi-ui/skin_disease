# -*- coding: utf-8 -*-
"""
نظام ويب ذكي لدعم التشخيص الأولي للأمراض الجلدية
Flask + ResNet50 (Transfer Learning) + SQLAlchemy

تشغيل التطبيق:
    pip install -r requirements.txt
    python app.py
"""
import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, redirect, url_for, request, flash, abort, jsonify
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.utils import secure_filename

from config import Config
from extensions import db, login_manager
from models import User, Patient, UserAction, Prediction
from decorators import role_required
import ml_model


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.dirname(app.config["MODEL_PATH"]), exist_ok=True)

    with app.app_context():
        db.create_all()
        # تحميل نموذج الذكاء الاصطناعي مرة واحدة عند إقلاع الخادم
        ml_model.load_model(app.config)

    register_routes(app)
    return app


# ---------------------------------------------------------------------------
# دوال مساعدة
# ---------------------------------------------------------------------------
def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def log_action(user_id, action_type, screen_name, description=""):
    """تسجيل حركة المستخدم في جدول user_actions (سجل تدقيق)."""
    action = UserAction(
        user_id=user_id,
        action_type=action_type,
        screen_name=screen_name,
        description=description,
    )
    db.session.add(action)
    db.session.commit()


# ---------------------------------------------------------------------------
# تسجيل المسارات (Routes)
# ---------------------------------------------------------------------------
def register_routes(app):

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ------------------ الصفحة الرئيسية ------------------
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            if current_user.is_doctor:
                return redirect(url_for("doctor_dashboard"))
            return redirect(url_for("patient_dashboard"))
        return redirect(url_for("login"))

    # ------------------ تسجيل حساب جديد ------------------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            phone = request.form.get("phone", "").strip()
            role = request.form.get("role", "patient")
            age = request.form.get("age", type=int)

            if role not in ("doctor", "patient"):
                role = "patient"

            if not full_name or not username or not password:
                flash("الرجاء تعبئة جميع الحقول الإلزامية", "danger")
                return render_template("register.html")

            if User.query.filter_by(username=username).first():
                flash("اسم المستخدم موجود مسبقاً، الرجاء اختيار اسم آخر", "danger")
                return render_template("register.html")

            new_user = User(
                username=username, full_name=full_name, phone=phone, role=role
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.flush()  # للحصول على new_user.id قبل الـ commit

            # إذا كان الحساب لمريض، ننشئ سجل مريض مرتبط تلقائياً
            if role == "patient":
                patient = Patient(
                    user_id=new_user.id, name=full_name, age=age, phone=phone
                )
                db.session.add(patient)

            db.session.commit()
            log_action(new_user.id, "register", "التسجيل", f"تم إنشاء حساب جديد بدور {role}")

            flash("تم إنشاء الحساب بنجاح، الرجاء تسجيل الدخول", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    # ------------------ تسجيل الدخول ------------------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("index"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = User.query.filter_by(username=username).first()
            if user is None or not user.check_password(password):
                flash("اسم المستخدم أو كلمة المرور غير صحيحة", "danger")
                return render_template("login.html")

            login_user(user)
            log_action(user.id, "login", "تسجيل الدخول", "تسجيل دخول ناجح")
            return redirect(url_for("index"))

        return render_template("login.html")

    # ------------------ تسجيل الخروج ------------------
    @app.route("/logout")
    @login_required
    def logout():
        log_action(current_user.id, "logout", "تسجيل الخروج", "تسجيل خروج")
        logout_user()
        return redirect(url_for("login"))

    # =======================================================================
    # واجهات المريض
    # =======================================================================
    @app.route("/patient/dashboard")
    @login_required
    @role_required("patient")
    def patient_dashboard():
        patient = current_user.patient_profile
        history = patient.predictions if patient else []
        return render_template("patient_dashboard.html", patient=patient, history=history)

    @app.route("/patient/diagnose", methods=["GET", "POST"])
    @login_required
    @role_required("patient")
    def diagnose():
        """واجهة تشخيص الحالة: رفع صورة -> معالجة -> استدعاء نموذج ResNet50 -> حفظ النتيجة."""
        if request.method == "POST":
            if "image" not in request.files:
                flash("الرجاء اختيار صورة للحالة الجلدية", "danger")
                return redirect(url_for("diagnose"))

            file = request.files["image"]
            if file.filename == "" or not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
                flash("صيغة الصورة غير مدعومة (استخدم png أو jpg أو jpeg)", "danger")
                return redirect(url_for("diagnose"))

            # حفظ الصورة باسم فريد لمنع تعارض الأسماء
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filename = secure_filename(unique_name)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # --- تمرير الصورة للنموذج الذكي (ResNet50) للحصول على التشخيص ---
            try:
                diagnosis, confidence, all_probs = ml_model.predict_image(save_path)
            except Exception as exc:
                flash(f"حدث خطأ أثناء تحليل الصورة: {exc}", "danger")
                return redirect(url_for("diagnose"))

            patient = current_user.patient_profile
            notes = request.form.get("notes", "").strip()

            prediction = Prediction(
                patient_id=patient.id,
                image_path=f"uploads/{filename}",
                diagnosis=diagnosis,
                confidence=confidence,
                all_probabilities=json.dumps(all_probs, ensure_ascii=False),
                doctor_notes=notes,
            )
            db.session.add(prediction)
            db.session.commit()

            log_action(
                current_user.id, "create", "تشخيص الحالة",
                f"تم رفع صورة وتحليلها، النتيجة: {diagnosis} ({confidence:.1f}%)"
            )

            return redirect(url_for("prediction_result", prediction_id=prediction.id))

        return render_template("diagnose.html")

    @app.route("/patient/result/<int:prediction_id>")
    @login_required
    def prediction_result(prediction_id):
        """واجهة نتائج التشخيص: تعرض قائمة الاحتمالات مرتبة حسب نسبة الثقة."""
        prediction = Prediction.query.get_or_404(prediction_id)

        # التحقق من الصلاحية: المريض يرى نتائجه فقط، والطبيب يرى كل الحالات
        if current_user.role == "patient" and prediction.patient.user_id != current_user.id:
            abort(403)

        all_probs = json.loads(prediction.all_probabilities or "{}")
        return render_template("result.html", prediction=prediction, all_probs=all_probs)

    # =======================================================================
    # واجهات الطبيب
    # =======================================================================
    @app.route("/doctor/dashboard")
    @login_required
    @role_required("doctor")
    def doctor_dashboard():
        """لوحة تحكم الطبيب: قائمة بجميع الحالات المشخّصة حديثاً."""
        recent_cases = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
        total_patients = Patient.query.count()
        total_cases = Prediction.query.count()
        pending_review = Prediction.query.filter_by(reviewed_by_doctor=False).count()
        return render_template(
            "doctor_dashboard.html",
            recent_cases=recent_cases,
            total_patients=total_patients,
            total_cases=total_cases,
            pending_review=pending_review,
        )

    @app.route("/doctor/patients")
    @login_required
    @role_required("doctor")
    def doctor_patients():
        """قائمة جميع المرضى مع خاصية البحث بالاسم أو رقم الهاتف."""
        query = request.args.get("q", "").strip()
        patients_query = Patient.query
        if query:
            like = f"%{query}%"
            patients_query = patients_query.filter(
                db.or_(Patient.name.ilike(like), Patient.phone.ilike(like))
            )
        patients = patients_query.order_by(Patient.created_at.desc()).all()
        return render_template("doctor_patients.html", patients=patients, query=query)

    @app.route("/doctor/patient/<int:patient_id>")
    @login_required
    @role_required("doctor")
    def doctor_patient_detail(patient_id):
        """واجهة تفاصيل مريض معين وسجل حالاته التشخيصية."""
        patient = Patient.query.get_or_404(patient_id)
        return render_template("doctor_patient_detail.html", patient=patient)

    @app.route("/doctor/case/<int:prediction_id>/note", methods=["POST"])
    @login_required
    @role_required("doctor")
    def add_doctor_note(prediction_id):
        """إضافة/تحديث ملاحظات الطبيب على حالة معينة وتأكيد المراجعة."""
        prediction = Prediction.query.get_or_404(prediction_id)
        prediction.doctor_notes = request.form.get("notes", "").strip()
        prediction.reviewed_by_doctor = True
        db.session.commit()

        log_action(
            current_user.id, "update", "مراجعة حالة",
            f"تمت مراجعة الحالة رقم {prediction.id} للمريض {prediction.patient.name}"
        )
        flash("تم حفظ الملاحظات وتأكيد مراجعة الحالة", "success")
        return redirect(url_for("prediction_result", prediction_id=prediction.id))

    @app.route("/doctor/logs")
    @login_required
    @role_required("doctor")
    def doctor_logs():
        """استعراض سجل حركات المستخدمين (Audit Log)."""
        logs = UserAction.query.order_by(UserAction.created_at.desc()).limit(200).all()
        return render_template("logs.html", logs=logs)

    # =======================================================================
    # API داخلي (اختياري) لتحليل صورة عبر AJAX دون إعادة تحميل الصفحة
    # =======================================================================
    @app.route("/api/predict", methods=["POST"])
    @login_required
    @role_required("patient")
    def api_predict():
        if "image" not in request.files:
            return jsonify({"error": "لم يتم إرسال صورة"}), 400

        file = request.files["image"]
        if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
            return jsonify({"error": "صيغة غير مدعومة"}), 400

        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            diagnosis, confidence, all_probs = ml_model.predict_image(save_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probs,
            "image_url": url_for("static", filename=f"uploads/{filename}"),
        })

    # ------------------ صفحات الأخطاء ------------------
    @app.errorhandler(403)
    def forbidden(e):
        return render_template("error.html", code=403, message="لا تملك صلاحية الوصول لهذه الصفحة"), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="الصفحة غير موجودة"), 404


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
