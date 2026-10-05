import customtkinter as ctk


class UserManagementView(ctk.CTkFrame):
    def __init__(self, parent, on_back):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_back = on_back

        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="#111827", corner_radius=14)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Użytkownicy",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        table = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        table.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        table.grid_columnconfigure((0, 1, 2, 3), weight=1)

        labels = ["Imię", "Nazwisko", "Rola", "Status"]
        for idx, label in enumerate(labels):
            ctk.CTkLabel(table, text=label, font=ctk.CTkFont(size=14, weight="bold"), text_color="#e2e8f0").grid(row=0, column=idx, padx=12, pady=(18, 10))

        rows = [
            ("Anna", "Kowalska", "Administrator", "Aktywny"),
            ("Jan", "Nowak", "Operator", "Aktywny"),
            ("Piotr", "Wiśniewski", "Operator", "Offline"),
            ("Karolina", "Mazur", "Administrator", "Aktywny"),
        ]

        for row_index, row in enumerate(rows, start=1):
            for col_index, value in enumerate(row):
                ctk.CTkLabel(table, text=value, text_color="#cbd5e1").grid(row=row_index, column=col_index, padx=12, pady=12, sticky="w")

        footer = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        footer.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            footer,
            text="Wróć do panelu",
            command=self.on_back,
            width=220,
            height=38,
            fg_color="#475569",
            hover_color="#334155",
        ).grid(row=0, column=0, padx=20, pady=14)
