import customtkinter as ctk

from app.services.camera_service import CameraService
from app.services.live_preview_service import LivePreviewService
from app.views.camera_manager_view import CameraManagerView
from app.views.dashboard_view import DashboardView
from app.views.events_view import EventsView
from app.views.live_preview_view import LivePreviewView
from app.views.login_view import LoginView
from app.views.register_view import RegisterView
from app.views.user_management_view import UserManagementView


class MockupApplication(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Guardian Security - Mockup")
        self.geometry("1280x820")
        self.minsize(980, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.camera_service = CameraService()
        self.live_preview_service = LivePreviewService()
        self.frames = {}

        self.login_view = LoginView(self, self.on_login, self.show_register)
        self.register_view = RegisterView(self, self.show_login, self.on_register_success)
        self.dashboard_view = DashboardView(
            self,
            on_open_cameras=self.show_camera_manager,
            on_open_events=self.show_events,
            on_open_users=self.show_users,
            on_open_live=self.show_live_preview,
        )
        self.camera_view = CameraManagerView(self, self.show_dashboard, self.camera_service)
        self.live_preview_view = LivePreviewView(self, self.show_dashboard, self.camera_service, self.live_preview_service)
        self.events_view = EventsView(self, self.show_dashboard)
        self.user_view = UserManagementView(self, self.show_dashboard)

        self.frames["login"] = self.login_view
        self.frames["register"] = self.register_view
        self.frames["dashboard"] = self.dashboard_view
        self.frames["cameras"] = self.camera_view
        self.frames["live"] = self.live_preview_view
        self.frames["events"] = self.events_view
        self.frames["users"] = self.user_view

        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_remove()

        self.show_login()

    def show_login(self):
        self._show_frame("login")

    def show_register(self):
        self._show_frame("register")

    def show_dashboard(self):
        self._show_frame("dashboard")

    def show_camera_manager(self):
        self.camera_view.refresh_cameras()
        self._show_frame("cameras")

    def show_live_preview(self):
        self.live_preview_view.refresh_camera_list()
        self._show_frame("live")

    def show_events(self):
        self._show_frame("events")

    def show_users(self):
        self._show_frame("users")

    def on_login(self, username):
        self.show_dashboard()

    def on_register_success(self, data):
        self.show_login()

    def _show_frame(self, name):
        for frame_name, frame in self.frames.items():
            if frame_name == name:
                frame.grid()
                frame.tkraise()
            else:
                frame.grid_remove()


if __name__ == "__main__":
    app = MockupApplication()
    app.mainloop()
