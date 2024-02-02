import streamlit as st

st.set_page_config(
    page_title='Welcome to the Llama Index Assistant',
    layout='wide',
    page_icon=':llama:'
)


def set_page_container_style(
    max_width: int = 1100, max_width_100_percent: bool = False,
    padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f'max-width: 100%;'
    else:
        max_width_str = f'max-width: {max_width}px;'
    st.markdown(
        f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
            </style>
            ''',
        unsafe_allow_html=True,
    )


set_page_container_style(
    max_width=1100, max_width_100_percent=True,
    padding_top=0, padding_right=10, padding_left=5, padding_bottom=10
)

st.image('images/header.jpg')
st.text('Check refresh')
