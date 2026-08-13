import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from types import ModuleType


def show_click_analytics():
    """
    Модуль ИИ-аналитики отзывов Click.
    Можно вызвать внутри твоего существующего меню страниц.
    """
    st.title("📊 Click Analytics AI")
    st.caption("Интеллектуальный ИИ-помощник продуктовой команды и службы поддержки Click")

    api_key = st.secrets["GOOGLE_API_KEY"]

    os.environ["GOOGLE_API_KEY"] = api_key
    DB_FAISS_PATH = "faiss_index_click"

    # КЭШИРОВАННАЯ ЗАГРУЗКА СИСТЕМЫ
    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(DB_FAISS_PATH):
            return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            st.error(
                f"❌ Ошибка: Папка векторной базы '{DB_FAISS_PATH}' не найдена! Убедитесь, что она лежит в корне проекта.")
            return None

    # Инициализация
    vector_store = load_vector_db()
    if vector_store is None:
        return
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.3)
    except Exception as e:
        st.error(f"Ошибка инициализации модуля: {e}")
        return

    st.divider()

    total_reviews = vector_store.index.ntotal  # Вытаскиваем точное число строк прямо из FAISS (покажет 229)
    negative_reviews = 98  # Сумма отзывов с оценкой 1★ (78) и 2★ (20)
    positive_reviews = 92  # Отзывы с оценкой 5★

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Всего отзывов в базе ИИ", value=total_reviews)
    with col2:
        st.metric(label="Критические жалобы (1-2★)", value=negative_reviews, delta="-Внимание", delta_color="inverse")
    with col3:
        st.metric(label="Позитивные отзывы (5★)", value=positive_reviews, delta="+Успех")

    st.divider()

    # ТАБЫ ИНТЕРФЕЙСА
    tab1, tab2 = st.tabs(["🔎 Поиск инсайтов (RAG)", "Автоответчик поддержки"])

    # Вкладка 1: RAG Поиск
    with tab1:
        st.subheader("Задайте вопрос по фидбеку пользователей")
        user_question = st.text_input(
            "Введите ваш запрос:",
            placeholder="Например: На что чаще всего жалуются в негативных отзывах?",
            key="click_rag_input"
        )

        if user_question:
            with st.spinner("Нейросеть анализирует контекст отзывов..."):
                relevant_docs = vector_store.similarity_search(user_question, k=5)
                context = "".join(
                    [f"Отзыв (Рейтинг: {d.metadata['rating']}★): {d.page_content}\n\n" for d in relevant_docs])

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system",
                     "Ты — ИИ-аналитик платежной системы Click в Узбекистане. Дай четкий, структурированный ответ на русском языке, опираясь строго на факты."
                     "Тебе предоставлен контекст, состоящий из реальных отзывов пользователей на разных языках. "
                     "Твоя задача — внимательно изучить эти отзывы и дать четкий, структурированный ответ на вопрос менеджера на РУССКОМ или УЗБЕКСТКОМ языке. "
                     "Пиши только факты, опираясь строго на предоставленные отзывы. Не придумывай лишнего.\n\n"
                     "Если вопрос не по системе Click говори пользователю задать вопрос по системе Click. "
                     ),
                    ("human", "КОНТЕКСТ С ОТЗЫВАМИ:\n{context}\n\nВопрос: {question}")
                ])

                chain = prompt_template | llm
                response = chain.invoke({"context": context, "question": user_question})

                st.markdown("Аналитический отчет ИИ:")
                st.info(response.content)

                with st.expander("Посмотреть оригиналы найденных отзывов"):
                    for doc in relevant_docs:
                        st.write(f"**Рейтинг:** {doc.metadata['rating']}★ | **Текст:** {doc.page_content}")
                        st.write("---")

    # Вкладка 2: Автоответчик
    with tab2:
        st.subheader("Генератор официальных ответов для клиентов")
        customer_complaint = st.text_area(
            "Вставьте текст жалобы или отзыва клиента:",
            placeholder="Например: Почему при переводе на Uzcard сняли комиссию?!",
            key="click_complaint_input"
        )

        if customer_complaint:
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                btn_ru = st.button("Сгенерировать ответ на русском", key="click_btn_ru")
            with col_lang2:
                btn_uz = st.button("Сгенерировать ответ на узбекском", key="click_btn_uz")

            if btn_ru or btn_uz:
                target_lang = "русском языке" if btn_ru else "узбекском языке (O'zbek tilida, рассудительно и официально)"

                with st.spinner("Формирую вежливый ответ..."):
                    reply_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         f"Ты — специалист поддержки Click. Напиши вежливый ответ клиенту на его жалобу строго на {target_lang}."
                         "Тебе предоставлен контекст, состоящий из реальных отзывов пользователей на разных языках. "
                         "Твоя задача — внимательно изучить эти отзывы и дать четкий, структурированный ответ на вопрос менеджера на РУССКОМ или УЗБЕКСТКОМ языке. "
                         "Пиши только факты, опираясь строго на предоставленные отзывы. Не придумывай лишнего.\n\n"
                         "Если вопрос не по системе Click говори пользователю задать вопрос по системе Click. "
                         "КОНТЕКСТ С ОТЗЫВАМИ:\n{context}"
                         ),
                        ("human", "Жалоба: {complaint}")
                    ])

                    reply_chain = reply_prompt | llm
                    bot_reply = reply_chain.invoke({"complaint": customer_complaint})

                    st.markdown("Шаблон ответа для отправки клиенту:")
                    st.success(bot_reply.content)
