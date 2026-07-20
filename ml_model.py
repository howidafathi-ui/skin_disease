# -*- coding: utf-8 -*-
"""
وحدة نموذج الذكاء الاصطناعي.
تبني معمارية ResNet50 (Transfer Learning) وتحمّل الأوزان المدرّبة (إن وجدت)
ثم توفر دالة predict_image() لتحليل صورة المرض الجلدي وإرجاع التشخيص.
"""
import os
import numpy as np
from PIL import Image

# نستورد Keras/TensorFlow بشكل كسول (Lazy) لتقليل زمن إقلاع السيرفر أثناء التطوير
_model = None
_config = None


def _build_model(num_classes):
    """بناء معمارية ResNet50 مع رأس تصنيف مخصص (نفس فكرة نقل التعلم Transfer Learning)."""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base_model = ResNet50(
        weights="imagenet",   # أوزان مدربة مسبقاً على ImageNet (نقل التعلم)
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # تجميد الطبقات التلافيفية الأساسية (Feature Extractor)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


def load_model(app_config):
    """تحميل النموذج مرة واحدة عند إقلاع التطبيق (Singleton)."""
    global _model, _config
    _config = app_config

    if _model is not None:
        return _model

    _model = _build_model(num_classes=len(app_config["CLASS_NAMES"]))

    weights_path = app_config["MODEL_PATH"]
    if os.path.exists(weights_path):
        _model.load_weights(weights_path)
        print(f"[ML] تم تحميل الأوزان المدربة من: {weights_path}")
    else:
        print(
            "[ML] تحذير: لم يتم العثور على ملف الأوزان المدرّب "
            f"({weights_path}). سيعمل النموذج بأوزان ImageNet الأولية فقط، "
            "وهو غير مناسب للتشخيص الفعلي. الرجاء تشغيل train_model.py "
            "على مجموعة بيانات جلدية (مثل HAM10000) أولاً."
        )
    return _model


def preprocess_image(image_path, target_size=(224, 224)):
    """قراءة الصورة وتجهيزها لتتوافق مع مدخلات ResNet50."""
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    array = np.array(img, dtype=np.float32)
    array = np.expand_dims(array, axis=0)   # (1, 224, 224, 3)
    array = preprocess_input(array)
    return array


def predict_image(image_path):
    """
    تشغيل النموذج على صورة واحدة وإرجاع:
    - التشخيص الأعلى احتمالاً (label)
    - نسبة الثقة (confidence %)
    - كل الاحتمالات لكل الفئات (dict)
    """
    if _model is None:
        raise RuntimeError("النموذج غير محمّل بعد. تأكد من استدعاء load_model() عند إقلاع التطبيق.")

    class_names = _config["CLASS_NAMES"]
    target_size = _config["IMAGE_SIZE"]

    x = preprocess_image(image_path, target_size)
    preds = _model.predict(x, verbose=0)[0]   # مصفوفة احتمالات بحجم عدد الفئات

    top_index = int(np.argmax(preds))
    diagnosis = class_names[top_index]
    confidence = float(preds[top_index] * 100)

    all_probs = {
        class_names[i]: round(float(preds[i] * 100), 2) for i in range(len(class_names))
    }
    # ترتيب تنازلي حسب نسبة الثقة (كما هو موصوف في شاشة "نتائج التشخيص")
    all_probs = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))

    return diagnosis, confidence, all_probs
