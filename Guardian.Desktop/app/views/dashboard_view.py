import customtkinter as ctk


class DashboardView(ctk.CTkFrame):
    def __init__(self, parent, on_open_cameras, on_open_events, on_open_users):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_open_cameras = on_open_cameras
        self.on_open_events = on_open_events
        self.on_open_users = on_open_users

        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Panel główny",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        ctk.CTkLabel(
            header,
            text="Mockup modułów dla systemu monitoringu",
            text_color="#cbd5e1",
        ).grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        cards = [
            ("Kamery", "Moduł kamer", self.on_open_cameras, "#2563eb"),
            ("Zdarzenia", "Lista zdarzeń", self.on_open_events, "#ea580c"),
            ("Użytkownicy", "Zarządzanie użytkownikami", self.on_open_users, "#16a34a"),
            ("Live View", "Podgląd w czasie rzeczywistym", lambda: None, "#7c3aed"),
            ("Alerty", "Przegląd alarmów", lambda: None, "#dc2626"),
            ("Ustawienia", "Konfiguracja systemu", lambda: None, "#0ea5e9"),
        ]

        for index, (title, subtitle, command, color) in enumerate(cards):
            row = index // 3
            col = index % 3
            tile = ctk.CTkFrame(self, width=240, height=170, corner_radius=18, fg_color="#111827")
            tile.grid(row=row + 1, column=col, padx=18, pady=15, sticky="nsew")
            tile.grid_propagate(False)

            ctk.CTkLabel(tile, text=title, font=ctk.CTkFont(size=20, weight="bold"), text_color="#f8fafc").place(x=18, y=18)
            ctk.CTkLabel(tile, text=subtitle, text_color="#cbd5e1", wraplength=200, justify="left").place(x=18, y=60)
            ctk.CTkButton(
                tile,
                text="Otwórz",
                width=180,
                height=36,
                command=command,
                fg_color=color,
                hover_color=color,
            ).place(x=18, y=118)

        self.grid_rowconfigure(3, weight=1)
