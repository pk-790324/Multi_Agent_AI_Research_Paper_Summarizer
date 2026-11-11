from services.auth_client import signup_api, login_api
import gradio as gr


def signup_user(name, email, password):

    if not name or not email or not password:
        return "Please fill in all fields.", gr.update()

    success, message = signup_api(
        name,
        email,
        password
    )

    return message, gr.update(selected=1) if success else gr.update()


def login_user(email, password):

    if not email or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "Please enter email and password.",
            None
        )

    success, user, message = login_api(
        email,
        password
    )

    if not success:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            message,
            None
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Welcome, {user['name']}!",
        user
    )


def logout_user():

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None
    )