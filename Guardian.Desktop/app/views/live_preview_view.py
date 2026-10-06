import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from app.bootstrap import bootstrap_application
from app.services.live_preview_service import LivePreviewService


class LivePreviewView(ctk.CTkFrame):
    def __init__(self, parent, on_back, camera_service, live_service=None, ai_pipeline=None):
        super().__init__(parent)
        self.configure(fg_color="#0f172a")
        self.on_back = on_back
        self.camera_service = camera_service
        self.live_service = live_service or LivePreviewService()
        self.ai_pipeline = ai_pipeline or bootstrap_application()
        self.camera_tiles = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Podgląd live",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).grid(row=0, column=0, padx=18, pady=(18, 4), sticky="w")

        self.status_label = ctk.CTkLabel(header, text="Wczytywanie kamer…", text_color="#cbd5e1")
        self.status_label.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="w")

        self.grid_container = ctk.CTkScrollableFrame(self, fg_color="#0b1220")
        self.grid_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        self.grid_container.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_container.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_container.configure(width=1300, height=820)

        self.footer = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        self.footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 20))
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

        self._last_frame_refresh = 0.0
        self.refresh_camera_list()
        self.after(50, self.refresh_live_tiles)

    def _clear_grid(self):
        for child in self.grid_container.winfo_children():
            child.destroy()
        self.camera_tiles = []

    def refresh_camera_list(self):
        self._clear_grid()
        cameras = self.camera_service.get_cameras()

        if not cameras:
            self.status_label.configure(text="Brak dodanych kamer.", text_color="#cbd5e1")
            empty = ctk.CTkLabel(self.grid_container, text="Brak kamer do podglądu", text_color="#cbd5e1")
            empty.grid(row=0, column=0, padx=20, pady=20, sticky="w")
            return

        self.status_label.configure(text=f"Aktywny podgląd: {len(cameras)} kamer", text_color="#a7f3d0")

        for index, camera in enumerate(cameras):
            row = index // 3
            col = index % 3
            tile = ctk.CTkFrame(self.grid_container, width=420, height=360, corner_radius=16, fg_color="#111827")
            tile.grid(row=row, column=col, padx=14, pady=14, sticky="nsew")
            tile.grid_propagate(False)

            ctk.CTkLabel(tile, text=camera.get("name", "Kamera"), font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").place(x=14, y=12)

            status = camera.get("status", "disconnected")
            status_color = "#22c55e" if status == "connected" else "#f87171"
            ctk.CTkButton(
                tile,
                text=status,
                width=96,
                height=28,
                fg_color=status_color,
                hover_color=status_color,
                state="disabled",
            ).place(x=300, y=12)

            preview = ctk.CTkLabel(tile, text="offline", width=360, height=240, text_color="#cbd5e1")
            preview.place(x=20, y=58)

            status_label = ctk.CTkLabel(tile, text="NO MOTION", text_color="#4ade80", font=ctk.CTkFont(size=12, weight="bold"))
            status_label.place(x=20, y=305)

            ctk.CTkButton(
                tile,
                text="Aktywny",
                width=360,
                height=34,
                fg_color="#7c3aed",
                hover_color="#6d28d9",
                state="disabled",
            ).place(x=20, y=330)

            self.camera_tiles.append({"camera": camera, "label": preview, "status_label": status_label})

        for camera in cameras:
            self.live_service.open_camera(camera, camera.get("source"), camera.get("source_type", "index"))

    def refresh_live_tiles(self):
        if not self.camera_tiles:
            self.after(50, self.refresh_live_tiles)
            return

        now = __import__("time").monotonic()
        refresh_interval = 1.0 / 20.0

        if now - self._last_frame_refresh < refresh_interval:
            self.after(50, self.refresh_live_tiles)
            return

        self._last_frame_refresh = now

        for item in self.camera_tiles:
            camera = item["camera"]
            label = item["label"]
            status_label = item["status_label"]
            frame = self.live_service.read_frame(camera)

            if frame is None:
                label.configure(text="offline", text_color="#fca5a5")
                status_label.configure(text="OFFLINE", text_color="#fca5a5")
                continue

            try:
                camera_key = str(camera.get("id", camera.get("source", "camera")))
                result = self.ai_pipeline.process_frame(frame, camera_id=camera_key)
                output = result.get("display", frame)
                status_text = str(result.get("status_text", "NO MOTION")).upper()
                print(f"[UI] camera={camera_key} status={status_text} display_shape={getattr(output, 'shape', 'no-shape')} dtype={getattr(output, 'dtype', 'no-dtype')}")
                if status_text == "NO MOTION":
                    color = "#4ade80"
                elif status_text == "NO FACE DETECTED":
                    color = "#fbbf24"
                elif status_text == "UNKNOWN":
                    color = "#ef4444"
                else:
                    color = "#22c55e"
                status_label.configure(text=status_text, text_color=color)
            except Exception as exc:
                output = frame
                print(f"[UI] EXCEPTION in AI processing: {exc}")
                status_label.configure(text="NO MOTION", text_color="#4ade80")

            try:
                if output is None:
                    raise ValueError("output frame is None")
                output = np.asarray(output)
                if output.ndim == 2:
                    rgb = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
                elif output.ndim == 3 and output.shape[2] == 4:
                    rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
                elif output.ndim == 3 and output.shape[2] == 3:
                    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"unsupported frame shape: {output.shape}")
                image = Image.fromarray(rgb)
                image = ImageOps.pad(image, (340, 220), color=(15, 23, 42))
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text="")
                label.image = photo
            except Exception as exc:
                print(f"[UI] FRAME_CONVERSION_ERROR: {exc} shape={getattr(output, 'shape', 'unknown')} type={type(output).__name__}")
                fallback = frame.copy()
                if fallback.ndim == 2:
                    fallback = cv2.cvtColor(fallback, cv2.COLOR_GRAY2RGB)
                elif fallback.ndim == 3 and fallback.shape[2] == 4:
                    fallback = cv2.cvtColor(fallback, cv2.COLOR_BGRA2BGR)
                else:
                    fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(fallback)
                image = ImageOps.pad(image, (340, 220), color=(15, 23, 42))
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text="")
                label.image = photo

        self.after(50, self.refresh_live_tiles)
