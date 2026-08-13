import streamlit as st

st.set_page_config(
    page_title="AI Portfolio",
    layout="wide"
)

# Главная информация о вас
st.title("Добро пожаловать в мое портфолио AI-проектов")
st.markdown("""
Меня зовут Shoxjahon, и я специализируюсь на разработке решений в области "Искусственного Интеллекта и Машинного Обучения". 
Здесь вы можете в интерактивном режиме протестировать мои проекты, посмотреть исходный код и оценить практическую пользу.
""")

st.divider()

# Секция с контактами и ссылками
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔗 GitHub Profile: https://github.com/ShoxjaxonDEV")
# with col2:
#     st.markdown("[💼 LinkedIn](https://linkedin.com/in/your_profile)")
with col3:
    st.markdown("✉ Telegram / Email: shoxmamurjonov02@gmail.com or contact to: @shoxa016")

st.info("💡 **Как пользоsваться:** Используйте боковое меню слева для переключения между проектами и тестирования моделей.")
