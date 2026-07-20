# -*- coding: utf-8 -*-
"""
سكربت تدريب/ضبط نموذج ResNet50 باستخدام تقنية نقل التعلم (Transfer Learning)
على مجموعة بيانات الأمراض الجلدية (مثال: HAM10000 أو أي مجموعة بيانات مشابهة).

هيكل مجلد البيانات المتوقع (ImageFolder classique):
dataset/
    train/
        Melanoma/
            img1.jpg ...
        Nevus/
            img1.jpg ...
        ...
    val/
        Melanoma/
        Nevus/
        ...

تشغيل السكربت:
    python train_model.py --data_dir dataset --epochs 15 --batch_size 32
"""
import argparse
import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def build_model(num_classes):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # تجميد الطبقات الأساسية (المرحلة الأولى من نقل التعلم)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="مسار مجلد البيانات (train/val)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                         help="عدد حقب الضبط الدقيق بعد فك تجميد آخر الطبقات")
    parser.add_argument("--output", type=str, default="model/resnet50_skin_disease.h5")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=args.batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=args.batch_size, class_mode="categorical"
    )

    num_classes = train_gen.num_classes
    print("الفئات المكتشفة (يجب أن تطابق CLASS_NAMES في config.py):")
    print(train_gen.class_indices)

    model, base_model = build_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = ModelCheckpoint(args.output, save_best_only=True, save_weights_only=True, monitor="val_accuracy")
    early_stop = EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy")

    # --- المرحلة الأولى: تدريب رأس التصنيف فقط (الطبقات الأساسية مجمّدة) ---
    print("\n=== المرحلة 1: تدريب رأس التصنيف (Feature Extraction) ===")
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs,
              callbacks=[checkpoint, early_stop])

    # --- المرحلة الثانية: ضبط دقيق (Fine-Tuning) لآخر طبقات ResNet50 ---
    print("\n=== المرحلة 2: الضبط الدقيق (Fine-Tuning) ===")
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=args.fine_tune_epochs,
              callbacks=[checkpoint, early_stop])

    model.save_weights(args.output)
    print(f"\nتم حفظ أوزان النموذج النهائية في: {args.output}")


if __name__ == "__main__":
    main()
