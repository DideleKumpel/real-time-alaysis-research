import customtkinter as ctk


class EventsView(ctk.CTkFrame):
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
            text="Lista zdarzeń",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        events = [
            ("08:12", "Wykryto ruch — kamera 01", "Alarm"),
            ("08:25", "Przejście osoby w rejonie wejścia", "Info"),
            ("09:03", "Przekroczenie limitu czasu bez ruchu", "Warn"),
            ("10:44", "Wykrycie twarzy — użytkownik zapisany", "Info"),
            ("11:15", "Wykryto nieautoryzowany ruch", "Alarm"),
        ]

        listbox = ctk.CTkScrollableFrame(self, width=1200, height=620, fg_color="#0b1220")
        listbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        for idx, (time, description, status) in enumerate(events):
            row = ctk.CTkFrame(listbox, fg_color="#111827", corner_radius=12)
            row.pack(fill="x", pady=8, padx=12)
            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(row, text=time, width=80, font=ctk.CTkFont(size=14, weight="bold"), text_color="#f8fafc").grid(row=0, column=0, padx=14, pady=12, sticky="w")
            ctk.CTkLabel(row, text=description, text_color="#dbeafe").grid(row=0, column=1, sticky="w", padx=10, pady=12)
            status_btn = ctk.CTkButton(row, text=status, width=100, height=28, fg_color="#334155", hover_color="#475569")
            status_btn.grid(row=0, column=2, padx=(10, 14), pady=12, sticky="e")

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
