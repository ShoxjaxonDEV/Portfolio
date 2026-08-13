import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate



def show_click_analytics():
    """
    Модуль ИИ-аналитики отзывов Click.
    РАБОТАЕТ НАПРЯМУЮ С ВЕКТОРНОЙ БАЗОЙ (Без использования CSV-файла).
    """
    st.title("📊 Click Analytics AI")
    st.caption("Интеллектуальный ИИ-помощник продуктовой команды и службы поддержки Click")

    # БЕЗОПАСНАЯ НАСТРОЙКА КЛЮЧА
    api_key = st.secrets["GOOGLE_API_KEY"]

    os.environ["GOOGLE_API_KEY"] = api_key
    DB_FAISS_PATH = "faiss_index_click"

    # МГНОВЕННАЯ ЗАГРУЗКА ВЕКТОРНОЙ БАЗЫ С ДИСКА
    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(DB_FAISS_PATH):
            return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            st.error(
                f"Ошибка: Папка векторной базы '{DB_FAISS_PATH}' не найдена! Убедитесь, что она лежит в корне проекта.")
            return None

    vector_store = load_vector_db()
    if vector_store is None:
        return

    try:
        # Фиксируем стабильную версию модели
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.3)
    except Exception as e:
        st.error(f"Ошибка инициализации Gemini API: {e}")
        return

    st.divider()

    # СЕКЦИЯ МЕТРИК
    total_reviews = vector_store.index.ntotal
    negative_reviews = 98
    positive_reviews = 92

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

    # Вкладка 1: RAG Поиск с умной фильтрацией звезд
    with tab1:
        st.subheader("Задайте вопрос по фидбеку пользователей")
        user_question = st.text_input(
            "Введите ваш запрос:",
            placeholder="Например: На что чаще всего жалуются в отзывах с оценкой 1 звезда?",
            key="click_rag_input_new"
        )

        if user_question:
            with st.spinner("Нейросеть анализирует контекст отзывов..."):
                # УМНЫЙ ФИЛЬТР ЗВЕЗД
                filter_dict = None
                q_lower = user_question.lower()

                if "1 звезда" in q_lower or "1★" in q_lower or "оценкой 1" in q_lower or "единиц" in q_lower:
                    filter_dict = {"rating": "1"}
                elif "5 звезд" in q_lower or "5★" in q_lower or "оценкой 5" in q_lower or "пятер" in q_lower:
                    filter_dict = {"rating": "5"}
                elif "2 звезд" in q_lower or "2★" in q_lower or "оценкой 2" in q_lower:
                    filter_dict = {"rating": "2"}

                # Выполняем поиск по базе FAISS
                if filter_dict:
                    relevant_docs = vector_store.similarity_search(user_question, k=5, filter=filter_dict)
                else:
                    relevant_docs = vector_store.similarity_search(user_question, k=5)

                # Собираем контекст
                context = ""
                for index, d in enumerate(relevant_docs, 1):
                    # Важнейший фикс: убираем фигурные скобки из текста отзывов, чтобы LangChain не падал!
                    safe_text = str(d.page_content).replace("{", "[").replace("}", "]")
                    context += f"Отзыв №{index} (Рейтинг: {d.metadata['rating']}★): {safe_text}\n\n"
                
                # 5. Шаблон промпта
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "Ты — ИИ-аналитик платежной системы Click в Узбекистане. Дай структурированный отчет на русском языке, опираясь строго на предоставленный контекст отзывов. Пиши четко."
                    "Пиши только факты, опираясь строго на предоставленные отзывы. Не придумывай лишнего.\n\n"
                    "Если вопрос не по системе Click говори пользователю задать вопрос по системе Click. "),
                    ("human", "КОНТЕКСТ С ОТЗЫВАМИ:\n{context}\n\nВОПРОС: {question}")
                ])
                
                try:
                    # 1. Собираем стандартную цепочку
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context, "question": user_question})
                    
                    # 2. Достаем сырой контент
                    raw_content = response.content if hasattr(response, "content") else response
                    
                    # 3. Если прилетел багнутый список со словарем [{'type': 'text', 'text': '...'}]
                    if isinstance(raw_content, list) and len(raw_content) > 0:
                        first_item = raw_content[0]
                        if isinstance(first_item, dict) and "text" in first_item:
                            clean_text = first_item["text"]
                        else:
                            clean_text = str(first_item)
                    else:
                        clean_text = str(raw_content)
                    
                    st.markdown(" Аналитический отчет ИИ:")
                    st.markdown(clean_text.strip())
                    
                    with st.expander("Посмотреть оригиналы найденных отзывов"):
                        for doc in relevant_docs:
                            st.write(f"**Рейтинг:** {doc.metadata['rating']}★ | **Текст:** {doc.page_content}")
                            st.write("---")

                            
                except Exception as inner_e:
                    st.error(f"Ошибка генерации через Google API: {inner_e}")

    # Вкладка 2: Полностью исправленный Автоответчик
    with tab2:
        st.subheader("Генератор официальных ответов для клиентов")
        customer_complaint = st.text_area(
            "Вставьте текст жалобы или отзыва клиента:",
            placeholder="Например: Почему при переводе на Uzcard сняли комиссию?!",
            key="click_complaint_area"
        )

        if customer_complaint:
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                btn_ru = st.button("Сгенерировать ответ на русском", key="click_btn_ru_new")
            with col_lang2:
                btn_uz = st.button("Сгенерировать ответ на узбекском", key="click_btn_uz_new")

            if btn_ru or btn_uz:
                target_lang = "русском языке" if btn_ru else "узбекском языке (O'zbek tilida, официально и уважительно)"

                with st.spinner("Формирую вежливый ответ..."):
                    # Исправленная структура промпта без конфликтов переменных
                    try:
                        reply_prompt = ChatPromptTemplate.from_messages([
                            ("system", f"Ты — специалист поддержки Click. Напиши вежливый ответ клиенту на его жалобу строго на {target_lang}. Извинись за неудобства и пообещай разобраться."
                            "Если вопрос не по системе Click говори пользователю задать вопрос по системе Click. "),
                            ("human", "Жалоба клиента: {question}")
                        ])
                        
                        reply_chain = reply_prompt | llm
                        bot_reply = reply_chain.invoke({"question": customer_complaint})
                        
                        # 2. Достаем сырой контент
                        raw_reply = bot_reply.content if hasattr(bot_reply, "content") else bot_reply
                        
                        # 3. Очищаем от скобок
                        if isinstance(raw_reply, list) and len(raw_reply) > 0:
                            first_item = raw_reply[0]
                            if isinstance(first_item, dict) and "text" in first_item:
                                clean_reply = first_item["text"]
                            else:
                                clean_reply = str(first_item)
                        else:
                            clean_reply = str(raw_reply)
                        
                        st.markdown("ответ для клиента:")
                        st.success(clean_reply.strip())
                    
                    except Exception as inner_e:
                        st.error(f"Ошибка автоответчика: {inner_e}")
show_click_analytics()
