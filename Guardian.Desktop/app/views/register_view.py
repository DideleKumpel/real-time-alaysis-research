import customtkinter as ctk


class RegisterView(ctk.CTkFrame):
    def __init__(self, parent, on_back, on_submit):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_back = on_back
        self.on_submit = on_submit

        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Rejestracja",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#f8fafc",
        )
        title.grid(row=0, column=0, pady=(40, 15))

        panel = ctk.CTkFrame(self, width=520, height=520, corner_radius=18, fg_color="#111827")
        panel.grid(row=1, column=0, padx=20, pady=10)
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)

        labels = [
            ("Login", 1),
            ("Email", 3),
            ("Imię", 5),
            ("Nazwisko", 7),
            ("Hasło", 9),
        ]

        self.entries = {}
        for text, row in labels:
            ctk.CTkLabel(panel, text=text, text_color="#cbd5e1").grid(row=row, column=0, sticky="w", padx=30, pady=(12, 2))
            entry = ctk.CTkEntry(panel, width=420, placeholder_text=text)
            entry.grid(row=row + 1, column=0, padx=30, pady=(0, 8))
            self.entries[text.lower()] = entry

        self.submit_button = ctk.CTkButton(
            panel,
            text="Utwórz konto",
            width=420,
            height=40,
            command=self._handle_submit,
            fg_color="#16a34a",
            hover_color="#15803d",
        )
        self.submit_button.grid(row=12, column=0, padx=30, pady=(14, 8))

        self.back_button = ctk.CTkButton(
            panel,
            text="Wróć do logowania",
            width=420,
            height=36,
            command=self.on_back,
            fg_color="#334155",
            hover_color="#475569",
        )
        self.back_button.grid(row=13, column=0, padx=30, pady=(0, 18))

        self.status_label = ctk.CTkLabel(panel, text="", text_color="#a7f3d0")
        self.status_label.grid(row=14, column=0, padx=30, sticky="w")

    def _handle_submit(self):
        required = ["login", "email", "imię", "nazwisko", "hasło"]
        values = {key: self.entries[key].get().strip() for key in required}

        if any(not value for value in values.values()):
            self.status_label.configure(text="Wypełnij wszystkie pola.", text_color="#fca5a5")
            return

        self.status_label.configure(text="Konto mock zostało utworzone.", text_color="#a7f3d0")
        if self.on_submit:
            self.on_submit(values)
