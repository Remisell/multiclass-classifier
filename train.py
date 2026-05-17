import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import os

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'


class MulticlassGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, batch_size, preprocessing_function=None):
        self.generator = generator
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function

        # Получаем пути к файлам и метки
        self.file_paths = np.array(generator.filepaths)
        self.labels = np.array(generator.classes)
        self.class_names = list(generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        self.class_indices = generator.class_indices

        # Группируем индексы по классам
        self.class_indices_dict = {}
        for class_idx in range(self.num_classes):
            self.class_indices_dict[class_idx] = np.where(self.labels == class_idx)[0]

        # Вычисляем количество примеров каждого класса в батче
        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.adjusted_batch_size = self.samples_per_class * self.num_classes

        # Сохраняем параметры генератора
        self.target_size = generator.target_size
        self.interpolation = generator.interpolation
        self.shuffle = generator.shuffle

        # Количество шагов за эпоху
        self.steps_per_epoch = self.calculate_steps()

        print(f"Мультиклассовый генератор:")
        print(f"  Количество классов: {self.num_classes}")
        print(f"  Примеров каждого класса в батче: {self.samples_per_class}")
        print(f"  Размер батча: {self.adjusted_batch_size}")
        for class_idx in range(self.num_classes):
            print(f"  Класс {self.class_names[class_idx]}: {len(self.class_indices_dict[class_idx])} примеров")
        print(f"  Шагов за эпоху: {self.steps_per_epoch}")

    def calculate_steps(self):
        # Минимальное количество примеров среди всех классов
        min_samples = min(len(indices) for indices in self.class_indices_dict.values())
        steps = max(1, min_samples // self.samples_per_class)
        return steps

    def __len__(self):
        return self.steps_per_epoch

    def load_image(self, filepath):
        img = tf.keras.utils.load_img(
            filepath,
            target_size=self.target_size,
            interpolation=self.interpolation
        )
        img = tf.keras.utils.img_to_array(img)

        if self.preprocessing_function:
            img = self.preprocessing_function(img)

        return img

    def __getitem__(self, idx):
        batch_images = []
        batch_labels = []

        # Для каждого класса выбираем равное количество примеров
        for class_idx in range(self.num_classes):
            if len(self.class_indices_dict[class_idx]) > 0:
                class_indices = np.random.choice(
                    self.class_indices_dict[class_idx],
                    size=self.samples_per_class,
                    replace=len(self.class_indices_dict[class_idx]) < self.samples_per_class
                )

                for idx in class_indices:
                    img = self.load_image(self.file_paths[idx])
                    batch_images.append(img)
                    batch_labels.append(class_idx)

        # Преобразуем в numpy массивы
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Перемешиваем батч
        shuffle_idx = np.random.permutation(len(batch_images))
        batch_images = batch_images[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]

        # One-hot encoding
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        return batch_images, y

    def on_epoch_end(self):
        if self.shuffle:
            for class_idx in self.class_indices_dict:
                np.random.shuffle(self.class_indices_dict[class_idx])


def create_data_generators(train_dir, val_dir, img_size, batch_size):
    # Аугментация данных для обучения
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )

    # Без аугментации для валидации
    val_datagen = ImageDataGenerator()

    # Создание оригинальных генераторов для получения структуры данных
    original_train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    original_val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Создание сбалансированных генераторов
    train_gen = MulticlassGenerator(
        original_train_gen,
        batch_size,
        preprocessing_function=preprocess_input
    )

    val_gen = MulticlassGenerator(
        original_val_gen,
        batch_size,
        preprocessing_function=preprocess_input
    )

    class_names = list(original_train_gen.class_indices.keys())

    return train_gen, val_gen, class_names


def create_multiclass_model(input_shape=(512, 512, 3), num_classes=None):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model


def train_multiclass_classifier(train_dir, val_dir):
    print("Обучение мультиклассового классификатора")

    # Создание сбалансированных генераторов
    train_gen, val_gen, class_names = create_data_generators(
        train_dir, val_dir, IMG_SIZE, BATCH_SIZE
    )

    num_classes = len(class_names)
    print(f"Количество классов: {num_classes}")
    print(f"Классы: {class_names}")

    # Создание модели
    model, base_model = create_multiclass_model(num_classes=num_classes)

    # Сохраняем имена классов в модель
    model.class_names = class_names

    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'multiclass_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    # Этап 1: Тренируем только добавленные слои классификатора
    print("\nЭтап 1: Тренируем только добавленные слои классификатора")
    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks,
        verbose=1
    )

    # Этап 2: Размораживаем часть EfficientNet для тонкой настройки
    base_model.trainable = True
    # Замораживаем первые 100 слоев
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nЭтап 2: Размораживаем часть EfficientNet для тонкой настройки")
    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks,
        verbose=1
    )

    # Сохраняем отдельно имена классов
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Оценка модели на валидации
    val_loss, val_acc = model.evaluate(val_gen)

    print(f"\nОбучение завершено!")
    print(f"Финальная точность на валидации: {val_acc:.4f}")

    return model, class_names, history_stage1, history_stage2


def main():
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Директория {TRAIN_DIR} не найдена")
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Директория {VAL_DIR} не найдена")

    # Обучение мультиклассового классификатора
    model, class_names, hist1, hist2 = train_multiclass_classifier(
        TRAIN_DIR, VAL_DIR
    )

    # Сохранение информации о модели
    model_info = {
        'class_names': class_names,
        'img_size': IMG_SIZE,
        'num_classes': len(class_names),
        'best_val_acc': max(hist2.history['val_accuracy']),
        'history_stage1': hist1.history,
        'history_stage2': hist2.history
    }

    print(f"\nОбучение завершено!")
    print(f"Количество классов: {len(class_names)}")
    print(f"Классы: {class_names}")
    print(f"Лучшая точность: {model_info['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()