import streamlit_authenticator as stauth

passwords = ["123"]  # List of raw passwords
hashed = stauth.Hasher(passwords).generate()
print(hashed)
