import httpx

API_URL = "http://127.0.0.1:8001/auth"


def signup_api(name, email, password):

    try:
        response = httpx.post(
            f"{API_URL}/signup",
            json={
                "name": name,
                "email": email,
                "password": password
            },
            timeout=10.0
        )

        if response.status_code == 201:
            return True, "Account created successfully! Please login."

        error = response.json().get(
            "detail",
            "Signup failed."
        )

        return False, error

    except httpx.RequestError:
        return False, "Authentication server is unavailable."


def login_api(email, password):

    try:
        response = httpx.post(
            f"{API_URL}/login",
            json={
                "email": email,
                "password": password
            },
            timeout=10.0
        )

        if response.status_code == 200:
            data = response.json()
            user = data.get("user")
            access_token = data.get("access_token")

            if not isinstance(user, dict) or not access_token:
                return False, None, "Login response did not include an authentication token."

            user["access_token"] = access_token
            return True, user, "Login successful!"

        error = response.json().get(
            "detail",
            "Invalid email or password."
        )

        return False, None, error

    except httpx.RequestError:
        return False, None, "Authentication server is unavailable."