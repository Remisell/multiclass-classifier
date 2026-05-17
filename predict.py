import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import threading
from PIL import Image, ImageTk
import json

class MultiClassClassifier:
    def __init__(self, model_path='multiclass_model.keras'):
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Получаем имена классов из модели или задаем вручную
            self.class_names = self._get_class_names()
            self.num_classes = len(self.class_names)
            print(f"Модель загружена. Количество классов: {self.num_classes}")
            print(f"Классы: {self.class_names}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

    # Получение имён классов
    def _get_class_names(self):
        try:
            with open('class_names.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("class_names.json не найден, используются стандартные имена")
            return [f"Класс_{i}" for i in range(self.model.output_shape[-1])]
        except Exception as e:
            print(f"Ошибка загрузки class_names.json: {e}")
            return [f"Класс_{i}" for i in range(self.model.output_shape[-1])]

    # Предсказание класса для одного изображения
    def predict_image(self, image_path, threshold=None):
        try:
            # Загрузка и преобразование изображения
            img = image.load_img(image_path, target_size=(512, 512))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Предсказание
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Получаем топ-3 предсказания
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'probability': float(predictions[idx]),
                    'index': int(idx)
                }
                for idx in top_indices
            ]

            # Основной результат
            best_class_idx = top_indices[0]
            best_probability = float(predictions[best_class_idx])

            if threshold is not None:
                if best_probability >= threshold:
                    predicted_class = self.class_names[best_class_idx]
                    confidence = best_probability
                else:
                    predicted_class = "Неопределено"
                    confidence = best_probability
            else:
                predicted_class = self.class_names[best_class_idx]
                confidence = best_probability

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'all_probabilities': {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(self.class_names))
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': 'Ошибка',
                'confidence': 0,
                'top_predictions': [],
                'all_probabilities': {}
            }

    # Пакетное предсказание дял нескольких изображений
    def predict_batch(self, image_paths, threshold=None, progress_callback=None):
        results = []
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            result = self.predict_image(image_path, threshold)
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            results.append(result)

            if progress_callback:
                progress_callback((i + 1) / total * 100)

        return results


class ClassifierGUI:
    def __init__(self, classifier):
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Мультиклассовый классификатор изображений")
        self.root.geometry("900x700")

        self.setup_ui()

    # Настройка пользовательского интерфейса
    def setup_ui(self):
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Заголовок
        title_label = ttk.Label(main_frame, text="Классификатор изображений",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Информация о модели
        model_info = f"Модель: {self.classifier.num_classes} классов"
        model_label = ttk.Label(main_frame, text=model_info, font=('Arial', 10))
        model_label.grid(row=1, column=0, columnspan=3, pady=5)

        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Загрузить изображение",
                   command=self.load_single_image).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Загрузить папку",
                   command=self.load_folder).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Очистить",
                   command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # Настройки порога
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(settings_frame, text="Порог уверенности:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.0)
        threshold_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0,
                                    variable=self.threshold_var, orient=tk.HORIZONTAL,
                                    length=200)
        threshold_scale.pack(side=tk.LEFT, padx=5)

        self.threshold_label = ttk.Label(settings_frame, text="0.0")
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        # Обновление метки порога
        def update_threshold_label():
            self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

        self.threshold_var.trace('w', update_threshold_label)

        # Область отображения изображения
        self.image_frame = ttk.LabelFrame(main_frame, text="Изображение", padding="5")
        self.image_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.image_label = ttk.Label(self.image_frame, text="Изображение не загружено")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Результаты классификации
        self.results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="5")
        self.results_frame.grid(row=4, column=1, columnspan=2, pady=10, padx=10,
                                sticky=(tk.W, tk.E, tk.N, tk.S))

        # Текстовое поле для результатов
        self.results_text = tk.Text(self.results_frame, height=15, width=50)
        self.results_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Скроллбар
        scrollbar = ttk.Scrollbar(self.results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=3, pady=10,
                               sticky=(tk.W, tk.E))

        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Настройка расширения колонок
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)

    # Загрузка одного изображения
    def load_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Все файлы", "*.*")
            ]
        )

        if file_path:
            self.process_image(file_path)

    # Загрузка папки с изображениями
    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с изображениями")

        if folder_path:
            # Поиск изображений в папке
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []

            for file in os.listdir(folder_path):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(folder_path, file))

            if image_files:
                self.process_batch(image_files)
            else:
                messagebox.showwarning("Предупреждение",
                                       "В выбранной папке нет изображений")

    # Обработка одного изображения
    def process_image(self, image_path):
        # Отображение изображения
        self.display_image(image_path)

        # Классификация
        threshold = self.threshold_var.get() if self.threshold_var.get() > 0 else None
        result = self.classifier.predict_image(image_path, threshold)

        # Отображение результатов
        self.display_results(result, single=True)

    # Обработка пакета изображений
    def process_batch(self, image_paths):
        self.status_var.set("Обработка изображений...")
        self.progress_var.set(0)

        def process_thread():
            threshold = self.threshold_var.get() if self.threshold_var.get() > 0 else None
            results = self.classifier.predict_batch(
                image_paths,
                threshold,
                progress_callback=lambda p: self.root.after(0, self.update_progress, p)
            )

            self.root.after(0, self.display_batch_results, results)
            self.root.after(0, lambda: self.status_var.set("Обработка завершена"))

        threading.Thread(target=process_thread, daemon=True).start()

    # Отображение изображения в GUI
    def display_image(self, image_path):
        try:
            # Загрузка и изменение размера изображения
            pil_image = Image.open(image_path)
            pil_image.thumbnail((400, 400), Image.Resampling.LANCZOS)

            # Конвертация для tkinter
            tk_image = ImageTk.PhotoImage(pil_image)

            # Обновление метки
            self.image_label.config(image=tk_image, text="")
            self.image_label.image = tk_image  # Сохраняем ссылку

        except Exception as e:
            self.image_label.config(text=f"Ошибка загрузки изображения: {e}")

    # Отображение результатов в текстовом поле
    def display_results(self, result, single=True):
        self.results_text.delete(1.0, tk.END)

        if 'error' in result:
            self.results_text.insert(tk.END, f"Ошибка: {result['error']}\n")
            return

        if single:
            # Результат для одного изображения
            self.results_text.insert(tk.END, "=== Результат классификации ===\n\n")
            self.results_text.insert(tk.END,
                                     f"Предсказанный класс: {result['predicted_class']}\n")
            self.results_text.insert(tk.END,
                                     f"Уверенность: {result['confidence'] * 100:.1f}%\n\n")

            self.results_text.insert(tk.END, "Топ-3 предсказания:\n")
            for i, pred in enumerate(result['top_predictions'], 1):
                self.results_text.insert(tk.END,
                                         f"  {i}. {pred['class']}: {pred['probability'] * 100:.1f}%\n")

            self.results_text.insert(tk.END, "\nВсе вероятности:\n")
            for class_name, prob in result['all_probabilities'].items():
                self.results_text.insert(tk.END,
                                         f"  {class_name}: {prob * 100:.1f}%\n")

    # Отображение результатов пакетной обработки
    def display_batch_results(self, results):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"=== Результаты обработки {len(results)} изображений ===\n\n")

        # Статистика
        class_counts = {}
        for result in results:
            predicted_class = result.get('predicted_class', 'Ошибка')
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

        self.results_text.insert(tk.END, "Статистика предсказаний:\n")
        for class_name, count in sorted(class_counts.items()):
            self.results_text.insert(tk.END, f"  {class_name}: {count} изображений\n")

        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n\n")

        # Детальные результаты
        for i, result in enumerate(results, 1):
            self.results_text.insert(tk.END, f"{i}. {result.get('image_name', 'Неизвестно')}\n")

            if 'error' in result:
                self.results_text.insert(tk.END, f"   Ошибка: {result['error']}\n")
            else:
                self.results_text.insert(tk.END,
                                         f"   Класс: {result['predicted_class']}\n")
                self.results_text.insert(tk.END,
                                         f"   Уверенность: {result['confidence'] * 100:.1f}%\n")

                # Топ-3 предсказания
                top3 = result.get('top_predictions', [])[:3]
                if top3:
                    self.results_text.insert(tk.END, "   Топ-3:\n")
                    for pred in top3:
                        self.results_text.insert(tk.END,
                                                 f"     - {pred['class']}: {pred['probability'] * 100:.1f}%\n")

            self.results_text.insert(tk.END, "\n")

    # Обновление прогресс-бара
    def update_progress(self, value):
        self.progress_var.set(value)

    # Очистка результатов
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.image_label.config(image="", text="Изображение не загружено")
        self.progress_var.set(0)
        self.status_var.set("Готов к работе")

    # Запуск GUI
    def run(self):
        self.root.mainloop()


def main():
    try:
        # Инициализация классификатора
        print("Загрузка модели...")
        classifier = MultiClassClassifier('multiclass_model.keras')

        # Создание и запуск GUI
        gui = ClassifierGUI(classifier)
        gui.run()

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()