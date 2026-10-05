import customtkinter as ctk


class LoginView(ctk.CTkFrame):
    def __init__(self, parent, on_login, on_register):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_login = on_login
        self.on_register = on_register

        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Guardian Security",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#e2e8f0",
        )
        title.grid(row=0, column=0, pady=(40, 10))

        card = ctk.CTkFrame(self, width=420, height=420, corner_radius=18, fg_color="#111827")
        card.grid(row=1, column=0, padx=20, pady=10)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="Logowanie", font=ctk.CTkFont(size=24, weight="bold"), text_color="#f8fafc").grid(row=0, column=0, pady=(25, 10))

        ctk.CTkLabel(card, text="Login", text_color="#cbd5e1").grid(row=1, column=0, sticky="w", padx=30, pady=(10, 2))
        self.login_entry = ctk.CTkEntry(card, width=320, placeholder_text="wpisz login")
        self.login_entry.grid(row=2, column=0, padx=30, pady=(0, 12))

        ctk.CTkLabel(card, text="Hasło", text_color="#cbd5e1").grid(row=3, column=0, sticky="w", padx=30, pady=(8, 2))
        self.password_entry = ctk.CTkEntry(card, width=320, show="*", placeholder_text="wpisz hasło")
        self.password_entry.grid(row=4, column=0, padx=30, pady=(0, 18))
        self.password_entry.bind("<Return>", lambda event: self._handle_login())

        self.status_label = ctk.CTkLabel(card, text="", text_color="#fbbf24", anchor="w")
        self.status_label.grid(row=5, column=0, padx=30, sticky="w")

        self.login_button = ctk.CTkButton(
            card,
            text="Zaloguj",
            width=320,
            height=42,
            command=self._handle_login,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
        )
        self.login_button.grid(row=6, column=0, padx=30, pady=(18, 10))

        self.register_button = ctk.CTkButton(
            card,
            text="Zarejestruj",
            width=320,
            height=38,
            command=self.on_register,
            fg_color="#16a34a",
            hover_color="#15803d",
        )
        self.register_button.grid(row=7, column=0, padx=30, pady=(6, 25))

    def _handle_login(self):
        login = self.login_entry.get().strip()
        password = self.password_entry.get().strip()

        if not login or not password:
            self.status_label.configure(text="Wprowadź login i hasło.")
            return

        self.status_label.configure(text="Dane poprawne — mock logowania.")
        if self.on_login:
            self.on_login(login)
