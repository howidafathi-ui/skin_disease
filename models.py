# -*- coding: utf-8 -*-
"""
نماذج قاعدة البيانات (ORM) بناءً على تصميم الفصل الرابع من المشروع:
- جدول المستخدمين      (User)
- جدول المرضى          (Patient)
- جدول حركات المستخدم   (UserAction)
- جدول التنبؤات         (Prediction)
"""
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db


class User(db.Model, UserMixin):
    """جدول المستخدمين: يحفظ بيانات تسجيل الدخول والصلاحيات (طبيب / مريض)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)              # معرف المستخدم
    username = db.Column(db.String(80), unique=True, nullable=False)   # اسم المستخدم
    password_hash = db.Column(db.String(255), nullable=False)          # كلمة المرور (مشفرة)
    phone = db.Column(db.String(20))                                   # رقم الهاتف
    full_name = db.Column(db.String(120), nullable=False)              # الاسم الكامل
    role = db.Column(db.String(20), nullable=False, default="patient")  # الصلاحيات: doctor / patient
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # علاقة: إذا كان المستخدم مريضاً يرتبط بسجل مريض واحد (بيانات طبية إضافية)
    patient_profile = db.relationship(
        "Patient", backref="user", uselist=False, cascade="all, delete-orphan"
    )
    actions = db.relationship("UserAction", backref="user", cascade="all, delete-orphan")

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)

    @property
    def is_doctor(self):
        return self.role == "doctor"

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class Patient(db.Model):
    """جدول المرضى: البيانات الأساسية الخاصة بكل مريض."""
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)               # معرف المريض
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)           # اسم المريض
    age = db.Column(db.Integer)                                 # العمر
    phone = db.Column(db.String(20))                            # رقم الهاتف
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship(
        "Prediction", backref="patient", cascade="all, delete-orphan",
        order_by="desc(Prediction.created_at)"
    )

    def __repr__(self):
        return f"<Patient {self.name}>"


class UserAction(db.Model):
    """جدول حركات المستخدم: سجل تدقيق (Audit Log) لكل إجراء يقوم به المستخدم."""
    __tablename__ = "user_actions"

    id = db.Column(db.Integer, primary_key=True)                # معرف الحركة
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)      # نوع الحركة: login / create / update / delete
    screen_name = db.Column(db.String(80))                      # اسم الشاشة التي حدثت فيها الحركة
    description = db.Column(db.String(255))                     # وصف الحركة
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UserAction {self.action_type} by user {self.user_id}>"


class Prediction(db.Model):
    """جدول التنبؤات: نتيجة تحليل صورة المريض بواسطة نموذج ResNet50."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)                 # معرف التنبؤ
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)       # مسار ملف الصورة
    diagnosis = db.Column(db.String(120), nullable=False)        # التشخيص الأولي (اسم الفئة)
    confidence = db.Column(db.Float, nullable=False)             # نسبة الثقة (0-100)
    all_probabilities = db.Column(db.Text)                       # JSON لكل الاحتمالات (لكل الأمراض)
    doctor_notes = db.Column(db.Text)                            # ملاحظات الطبيب على الحالة
    reviewed_by_doctor = db.Column(db.Boolean, default=False)    # هل راجعها الطبيب
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.diagnosis} ({self.confidence:.1f}%) for patient {self.patient_id}>"
