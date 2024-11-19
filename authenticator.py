import streamlit as st
import json


class Authenticator:
    def __init__(self):
        self.GROUP_URLS = {
            "c": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
            "r": "https://main-c-ey8rrf82hhhzsur2wrhmx9.streamlit.app/",
            "b": "https://main-c-chatbot-gf6b8fhdhdenreey45xtri.streamlit.app/",
        }
        # 特別なURLを定義
        self.SPECIAL_URL = "https://www.google.com"

    def get_attendance_attributes(self):
        with open("attendance_list.json", "r") as f:
            attendance_list = json.load(f)

        addresed_accounts = list(attendance_list.keys())
        return addresed_accounts, attendance_list

    def vertify_user_id(self, addresed_accounts):
        if "temp_user_id" not in st.session_state:
            st.session_state.temp_user_id = ""

        if st.session_state.temp_user_id in addresed_accounts:
            st.session_state.user_id = st.session_state.temp_user_id
        elif st.session_state.temp_user_id != "":
            st.error("無効なIDです。もう一度お試しください。")

        if "temp_user_id" in st.session_state:
            if "user_id" not in st.session_state:
                st.session_state.temp_user = st.text_input(
                    "User IDを入力してください", placeholder="Enter", key="temp_user_id"
                )

    def make_user_url(self, addresed_accounts, attendance_list, group_urls):
        if "user_id" in st.session_state:
            if st.session_state.user_id in addresed_accounts:
                attendance_attributes = attendance_list[st.session_state.user_id]
                group_name = attendance_attributes[0]
                theme = attendance_attributes[1]
                if group_name in group_urls:
                    group_url = group_urls[group_name]
                    group_url_with_id = f"{group_url}?user_id={st.session_state.user_id}&group={group_name}&talktheme={theme}"
                    st.success(f"ようこそ、{st.session_state.user_id} さん！")
                    st.markdown(
                        f"こちらのリンクをクリックして、今日の会話を開始してください。: <a href='{group_url_with_id}' target='_blank'>リンク</a>",
                        unsafe_allow_html=True,
                    )

    def forward(self):
        st.title("Authentication")
        addresed_accounts, attendance_list = self.get_attendance_attributes()
        self.vertify_user_id(addresed_accounts)
        self.make_user_url(addresed_accounts, attendance_list, self.GROUP_URLS)


if __name__ == "__main__":
    authenticator = Authenticator()
    authenticator.forward()
