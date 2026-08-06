import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.title("👁️ Детекция блоков в Minecraft (YOLO)")
st.markdown("Загрузите скриншот из игры, чтобы модель обнаружила и выделила блоки.")


# Загрузка модели (кешируем, чтобы не грузить заново при каждом клике)
@st.cache_resource
def load_yolo_model():
    current_dir = Path(__file__).parent.parent
    model_path = current_dir / "models" / "best.pt"
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Не удалось загрузить модель по пути: `{model_path}`. Ошибка: {e}")
        return None


model = load_yolo_model()

# Загрузка файла от пользователя
uploaded_file = st.file_uploader("Выберите изображение (PNG, JPG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    # Открываем картинку через PIL и переводим в формат OpenCV (numpy array)
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Изменяем размер под стандартный для модели (как у вас в коде: 640x480)
    frame_resized = cv2.resize(frame_bgr, (640, 480))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходный кадр")
        st.image(image, use_container_width=True)

    if st.button("🚀 Запустить детекцию объектов"):
        with st.spinner("Модель анализирует кадр..."):

            # --- ВАШ КОД ИНФЕРЕНСА ИЗ СКРИПТА ---
            results = model(frame_resized, verbose=False)[0]

            # Используем встроенный метод YOLO для отрисовки рамок (results.plot())
            # Он возвращает изображение в формате BGR
            res_plotted = results.plot()

            # Переводим обратно из BGR в RGB для корректного отображения в Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Результат работы модели")
                st.image(res_rgb, use_container_width=True)

            # --- АНАЛИЗ НАЙДЕННЫХ ОБЪЕКТОВ (как в вашем коде) ---
            st.divider()
            st.subheader("📊 Отчет анализа кадра:")

            detected_items = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if confidence > 0.5:
                    target = class_name.lower()
                    detected_items.append((class_name, confidence))

                    # Логика из вашего скрипта
                    if target in ["diamond ore", "iron ore"]:
                        st.info(
                            f"💎 Обнаружено: **{class_name}** (Уверенность: {confidence * 100:.1f}%). *Решение агента: Иду копать!*")
                    elif target in ["tree", "wood"]:
                        st.info(
                            f"🪵 Обнаружено: **{class_name}** (Уверенность: {confidence * 100:.1f}%). *Решение агента: Иду рубить дерево!*")

            if not detected_items:
                st.warning("Модель не обнаружила целевых объектов с уверенностью выше 50%.")
elif model is None:
    st.error("⚠️ Модель не загружена. Проверьте путь к папке `best_openvino_model` в коде.")

# Дополнительная информация о проекте
st.divider()
st.markdown("""
### ℹ️ О проекте:
* **Технологии:** Python,  OpenCV / PyAutoGUI,  YOLO, Torch,
* **Описание:** Модель обучена на моем датасете (сам создал), Компьютерное зрение (Vision): Агент распознает блоки, объекты в режиме реального времени на основе захвата экрана.
Навигация и движение (Movement): Автономное перемещение смотря на блок.
""")