import customtkinter as ctk

from app.services.camera_service import CameraService


class CameraManagerView(ctk.CTkFrame):
    def __init__(self, parent, on_back, camera_service: CameraService):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_back = on_back
        self.camera_service = camera_service

        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Zarządzanie kamerami",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).grid(row=0, column=0, padx=18, pady=(18, 8), sticky="w")

        ctk.CTkLabel(
            header,
            text="Lista podłączonych urządzeń i możliwość dodania nowych kamer.",
            text_color="#cbd5e1",
        ).grid(row=1, column=0, padx=18, pady=(0, 18), sticky="w")

        form = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        form.grid(row=1, column=0, sticky="ew", padx=20, pady=8)
        form.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(form, text="Nazwa kamery", text_color="#e2e8f0").grid(row=0, column=0, padx=12, pady=(18, 6), sticky="w")
        self.name_entry = ctk.CTkEntry(form, placeholder_text="np. Kamera garaż")
        self.name_entry.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(form, text="Typ źródła", text_color="#e2e8f0").grid(row=0, column=1, padx=12, pady=(18, 6), sticky="w")
        self.source_type = ctk.CTkOptionMenu(
            form,
            values=["index", "url"],
            width=180,
        )
        self.source_type.set("index")
        self.source_type.grid(row=1, column=1, padx=12, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(form, text="Źródło / numer kamery", text_color="#e2e8f0").grid(row=0, column=2, padx=12, pady=(18, 6), sticky="w")
        self.source_entry = ctk.CTkEntry(form, placeholder_text="USB: 0, 1, 2  |  URL: rtsp://... ")
        self.source_entry.grid(row=1, column=2, padx=12, pady=(0, 12), sticky="ew")

        self.helper_label = ctk.CTkLabel(
            form,
            text="Dla kamery podłączonej przez USB wpisz index urządzenia, np. 0, 1, 2. Dla streamingu wpisz URL, np. rtsp://192.168.0.10:8554/stream",
            text_color="#93c5fd",
            justify="left",
            wraplength=450,
        )
        self.helper_label.grid(row=2, column=0, columnspan=3, padx=12, pady=(0, 14), sticky="w")

        actions = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        actions.grid(row=2, column=0, sticky="ew", padx=20, pady=8)
        actions.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            actions,
            text="Dodaj kamerę",
            command=self.add_camera,
            width=200,
            height=38,
            fg_color="#16a34a",
            hover_color="#15803d",
        ).grid(row=0, column=0, padx=16, pady=14, sticky="w")

        ctk.CTkButton(
            actions,
            text="Odśwież status",
            command=self.refresh_cameras,
            width=200,
            height=38,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
        ).grid(row=0, column=1, padx=16, pady=14, sticky="e")

        self.status_banner = ctk.CTkLabel(self, text="", text_color="#a7f3d0")
        self.status_banner.grid(row=3, column=0, sticky="w", padx=24, pady=(8, 4))

        self.list_container = ctk.CTkScrollableFrame(self, width=1200, height=420, fg_color="#0b1220")
        self.list_container.grid(row=4, column=0, padx=20, pady=(8, 20), sticky="nsew")

        self.footer = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        self.footer.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 18))
        self.footer.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            self.footer,
            text="Wróć do panelu",
            command=self.on_back,
            width=220,
            height=38,
            fg_color="#475569",
            hover_color="#334155",
        ).grid(row=0, column=0, padx=20, pady=14)

        self.refresh_cameras()

    def _clear_list(self):
        for child in self.list_container.winfo_children():
            child.destroy()

    def refresh_cameras(self):
        self._clear_list()
        cameras = self.camera_service.get_cameras()

        if not cameras:
            empty = ctk.CTkLabel(self.list_container, text="Brak dodanych kamer.", text_color="#cbd5e1")
            empty.pack(padx=16, pady=20, anchor="w")
            return

        for camera in cameras:
            status = camera.get("status", "disconnected")
            color = "#22c55e" if status == "connected" else "#f87171"
            row = ctk.CTkFrame(self.list_container, fg_color="#111827", corner_radius=12)
            row.pack(fill="x", padx=12, pady=8)
            row.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

            ctk.CTkLabel(row, text=camera.get("name", "Kamera"), font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").grid(row=0, column=0, padx=12, pady=12, sticky="w")
            ctk.CTkLabel(row, text=camera.get("source", "-"), text_color="#cbd5e1").grid(row=0, column=1, padx=12, pady=12, sticky="w")
            ctk.CTkLabel(row, text=camera.get("source_type", "index"), text_color="#cbd5e1").grid(row=0, column=2, padx=12, pady=12, sticky="w")
            ctk.CTkButton(
                row,
                text=status,
                width=90,
                height=28,
                fg_color=color,
                hover_color=color,
            ).grid(row=0, column=3, padx=(6, 12), pady=12, sticky="e")
            ctk.CTkButton(
                row,
                text="Usuń",
                width=90,
                height=28,
                fg_color="#ef4444",
                hover_color="#dc2626",
                command=lambda cam_id=camera.get("id"): self.remove_camera(cam_id),
            ).grid(row=0, column=4, padx=(6, 12), pady=12, sticky="e")

    def add_camera(self):
        name = self.name_entry.get().strip()
        source = self.source_entry.get().strip()
        source_type = self.source_type.get()

        if not name or not source:
            self.status_banner.configure(text="Uzupełnij nazwę i źródło kamery.", text_color="#fca5a5")
            return

        if source_type == "index":
            try:
                int(source)
            except ValueError:
                self.status_banner.configure(text="Dla kamery USB wpisz numer indeksu, np. 0, 1, 2.", text_color="#fca5a5")
                return

        try:
            camera = self.camera_service.add_camera(name, source, source_type)
            self.name_entry.delete(0, "end")
            self.source_entry.delete(0, "end")
            self.status_banner.configure(text=f"Dodano kamerę: {camera['name']} ({camera['status']})", text_color="#a7f3d0")
            self.refresh_cameras()
        except ValueError as exc:
            self.status_banner.configure(text=str(exc), text_color="#fca5a5")

    def remove_camera(self, camera_id):
        removed = self.camera_service.remove_camera(camera_id)
        if removed:
            self.status_banner.configure(text="Kamera została usunięta.", text_color="#a7f3d0")
        else:
            self.status_banner.configure(text="Nie znaleziono kamery do usunięcia.", text_color="#fca5a5")
        self.refresh_cameras()
