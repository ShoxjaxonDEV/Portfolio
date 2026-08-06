import streamlit as st
import joblib
from pathlib import Path
import os

st.title("✉ Классификатор спама в Gmail (Scikit-Learn)")
st.markdown(
    "Интерактивный текстовый классификатор, определяющий нежелательные сообщения (спам) с помощью моделей Machine Learning.")

@st.cache_resource
def load_sklearn_assets():
    base_dir = Path(r"D:\Python project\first ai\spam detector")

    model_path = base_dir / "spam_model_n1.pkl"
    vectorizer_path = base_dir / "vector.pkl"

    # Проверим прямо в коде, существуют ли файлы на диске
    if not model_path.exists():
        st.error(f"Файл модели НЕ найден по пути: {model_path}")
        return None, None
    if not vectorizer_path.exists():
        st.error(f"Файл векторизатора НЕ найден по пути: {vectorizer_path}")
        return None, None

    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = joblib.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Ошибка при открытии pickle-файлов: {e}")
        return None, None


model, vectorizer = load_sklearn_assets()

# 2. Интерфейс для ввода текста письма
st.subheader("Проверить письмо на спам")
email_text = st.text_area(
    "Введите или вставьте текст электронного письма:",
    placeholder="Например: Поздравляем! Вы выиграли миллион долларов, нажмите на ссылку..."
)

if st.button("🔍 Проверить текст"):
    if not email_text.strip():
        st.warning("⚠️ Пожалуйста, введите текст письма.")
    elif model is None or vectorizer is None:
        st.error(
            "⚠️ Не удалось найти файлы модели (`spam_model.pkl`) или векторизатора (`vectorizer.pkl`). Проверьте пути к файлам в коде.")
    else:
        with st.spinner("Анализ текста..."):
            # --- ВАШ КОД ПРЕДСКАЗАНИЯ SCIKIT-LEARN ---
            # Векторизуем введенный текст
            text_vectorized = vectorizer.transform([email_text])

            # Предсказание модели
            prediction = model.predict(text_vectorized)[0]

            # Получаем вероятности, если модель их поддерживает
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(text_vectorized)[0]

            st.divider()
            st.subheader("📊 Результат анализа:")

            # Вывод в зависимости от результата (в зависимости от вашей разметки: 1/0 или 'spam'/'ham')
            is_spam = str(prediction).lower() in ["spam", "1", "true"]

            if is_spam:
                st.error("🚨 **Внимание: Это СПАМ!**")
                if proba is not None:
                    confidence = max(proba) * 100
                    st.write(f"Уверенность модели: **{confidence:.1f}%**")
            else:
                st.success("✅ **Это обычное письмо (Не спам)**")
                if proba is not None:
                    confidence = max(proba) * 100
                    st.write(f"Уверенность модели: **{confidence:.1f}%**")

# Дополнительная информация о проекте
st.divider()
st.markdown("""
### ℹ️ О проекте:
* **Технологии:** Python, Scikit-Learn, Pandas, Pickle.
* **Описание:** Модель обучена на размеченном датасете сообщений, использует векторизацию текста (например, TF-IDF / CountVectorizer) и алгоритм классификации для фильтрации нежелательной почты. Она не 100% угадывает спам сообщение или нет, в большинстве дает спам потому что в датасете было больше спама чем чистых сообщеий.
""")