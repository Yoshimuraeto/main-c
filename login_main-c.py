import streamlit as st
import json

# TODO: 飛べるURLにする
group_urls = {
    "a": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
    "b": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
    "c": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
    "d": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
    "e": "https://main-c-hulfrzmcmnhcahoxdpc2gc.streamlit.app/",
}

# 特別なURLを定義します
SPECIAL_URL = "https://www.google.com"

st.title("Login MainC")

with open("attendance_list.json", "r") as f:
    attendance_list = json.load(f)

addresed_accounts = list(attendance_list.keys())


def vertify_user_id():
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


vertify_user_id()

if "user_id" in st.session_state:
    if st.session_state.user_id in addresed_accounts:
        group_name = attendance_list[st.session_state.user_id]
        if group_name in group_urls:
            group_url = group_urls[group_name]
            group_url_with_id = (
                f"{group_url}?user_id={st.session_state.user_id}&group={group_url}"
            )
            st.success(f"ようこそ、{st.session_state.user_id} さん！")
            st.markdown(
                f"こちらのリンクをクリックして、今日の会話を開始してください。: <a href='{group_url_with_id}' target='_blank'>リンク</a>",
                unsafe_allow_html=True,
            )
