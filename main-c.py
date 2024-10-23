import streamlit as st
from streamlit_chat import message

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import datetime
import pytz

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class MainC:
    def __init__(self):
        self.chat_model = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model_name="gpt-4o",
            temperature=0.5,
            streaming=True,
            max_tokens=1024,
        )
        self.SYSTEM_PREFIX = "あなたはAIアシスタントです。 以下はAIアシスタントとの会話です。 このアシスタントは親切で、クリエイティブで、賢く、とてもフレンドリーです。"
        self.PROMPT = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.SYSTEM_PREFIX),
                MessagesPlaceholder("history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        self.addresed_accounts = ["a", "b", "c"]
        self.addresed_groups = ["???"]

    def vertify_user_id(self, addresed_accounts):
        if "temp_user_id" not in st.session_state:
            st.session_state.temp_user_id = ""

        if st.session_state.temp_user_id in addresed_accounts:
            st.session_state.user_id = st.session_state.temp_user_id

        # ユーザーIDがない場合にのみ入力フィールドを表示
        if "temp_user_id" in st.session_state:
            if "user_id" not in st.session_state:
                st.session_state.temp_user = st.text_input(
                    "Enter your user_id", placeholder="Enter", key="temp_user_id"
                )

    def vertify_group_id(self, addresed_accounts):
        if "temp_group_id" not in st.session_state:
            st.session_state.temp_group_id = ""

        if st.session_state.temp_group_id in addresed_accounts:
            st.session_state.group_id = st.session_state.temp_group_id

        # ユーザーIDがない場合にのみ入力フィールドを表示
        if "temp_group_id" in st.session_state:
            if "group_id" not in st.session_state:
                st.session_state.temp_user = st.text_input(
                    "Enter your group_id", placeholder="Enter", key="temp_group_id"
                )

    def prepare_firestore(self):
        try:
            if not firebase_admin._apps:
                type = st.secrets["type"]
                project_id = st.secrets["project_id"]
                private_key_id = st.secrets["private_key_id"]
                private_key = st.secrets["private_key"].replace("\\n", "\n")
                client_email = st.secrets["client_email"]
                client_id = st.secrets["client_id"]
                auth_uri = st.secrets["auth_uri"]
                token_uri = st.secrets["token_uri"]
                auth_provider_x509_cert_url = st.secrets["auth_provider_x509_cert_url"]
                client_x509_cert_url = st.secrets["client_x509_cert_url"]
                universe_domain = st.secrets["universe_domain"]
                # Firebase認証情報を設定
                cred = credentials.Certificate(
                    {
                        "type": type,
                        "project_id": project_id,
                        "private_key_id": private_key_id,
                        "private_key": private_key,
                        "client_email": client_email,
                        "client_id": client_id,
                        "auth_uri": auth_uri,
                        "token_uri": token_uri,
                        "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
                        "client_x509_cert_url": client_x509_cert_url,
                        "universe_domain": universe_domain,
                    }
                )
                default_app = firebase_admin.initialize_app(cred)
            db = firestore.client()
            return db

        except:
            self.disable_chat_input()
            return None

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        # セッションIDごとの会話履歴の取得
        if "store" not in st.session_state:
            st.session_state.store = {}

        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()

        return st.session_state.store[session_id]

    def prepare_memory(self, chat_model, prompt):
        if not hasattr(st.session_state, "runnable_with_history"):
            runnable = prompt | chat_model

            # RunnableWithMessageHistoryの準備
            st.session_state.runnable_with_history = RunnableWithMessageHistory(
                runnable,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )

    def display_chat_history(self):
        # チャットのメッセージの履歴作成と表示
        if "message_history" not in st.session_state:
            st.session_state.message_history = []
            st.session_state.initge = ["はじめまして!!"]

        else:
            chat_placeholder = st.empty()
            with chat_placeholder.container():
                message(
                    st.session_state.initge[0],
                    key="init_greeting_plus",
                    avatar_style="micah",
                )
                for i in range(len(st.session_state.message_history)):
                    message(
                        st.session_state.message_history[i],
                        is_user=True,
                        key=str(i),
                        avatar_style="adventurer",
                        seed="Nala",
                    )
                    key_generated = str(i) + "keyg"
                    message(
                        st.session_state.message_history[i],
                        key=str(key_generated),
                        avatar_style="micah",
                    )

    def generate_and_store_response(self, user_input, db):
        # AIからの応答を取得
        assistant_response = st.session_state.runnable_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": str(st.session_state.user_id)}},
        ).content

        # データベースに登録
        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        doc_ref = db.collection(str(st.session_state.user_id) + "111").document(
            str(now)
        )
        doc_ref.set({"user": user_input, "asistant": assistant_response})
        return assistant_response

    def disable_chat_input(self):
        st.session_state["chat_input_disabled"] = True

    def enable_chat_input(self):
        st.session_state["chat_input_disabled"] = False

    def forward(self):
        st.title("Main C")

        if "count" not in st.session_state:
            st.session_state.count = 0

        if "user_id" not in st.session_state:
            self.vertify_user_id(self.addresed_accounts)

        if "group_id" not in st.session_state:
            self.vertify_group_id(self.addresed_groups)

        if "chat_input_disabled" not in st.session_state:
            st.session_state.chat_input_disabled = False
            st.session_state.db = self.prepare_firestore()
            self.prepare_memory(self.chat_model, self.PROMPT)

        if st.session_state.db is None:
            st.write("Firebaseの認証に失敗しました")

        self.display_chat_history()

        # チャットの開始
        if "user_id" in st.session_state and "group_id" in st.session_state:

            if st.session_state.count >= 5:
                group_url = (
                    "https://nagoyapsychology.qualtrics.com/jfe/form/SV_5cZeI9RbaCdozTU"
                )
                group_url_with_id = f"{group_url}?user_id={st.session_state.user_id}&group={st.session_state.group_id}"
                st.markdown(
                    f'これで今回の会話は終了です。こちらをクリックしてアンケートに回答してください。: <a href="{group_url_with_id}" target="_blank">リンク</a>',
                    unsafe_allow_html=True,
                )
                self.disable_chat_input()

            else:
                if user_input := st.chat_input(
                    "メッセージを送る",
                    disabled=st.session_state.chat_input_disabled,
                    on_submit=self.disable_chat_input(),
                ):
                    # チャットメッセージコンテナにユーザーメッセージを表示
                    st.chat_message("user").markdown(user_input)

                    # チャット履歴にユーザーメッセージを追加
                    st.session_state.message_history.append(
                        {"role": "user", "content": user_input}
                    )

                    with st.spinner("Wait for it..."):
                        # AIからの応答を取得、データベースに登録
                        assistant_response = self.generate_and_store_response(
                            user_input, st.session_state.db
                        )

                    # AIからの応答を表示
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)

                    # チャット履歴にAIからの応答を追加
                    st.session_state.message_history.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                    st.session_state.count += 1

                    self.enable_chat_input()
                    st.rerun()


if __name__ == "__main__":
    mainc = MainC()
    mainc.forward()
