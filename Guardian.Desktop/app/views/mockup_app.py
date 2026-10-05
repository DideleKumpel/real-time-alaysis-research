import customtkinter as ctk

from app.views.dashboard_view import DashboardView
from app.views.events_view import EventsView
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

        self.frames = {}

        self.login_view = LoginView(self, self.on_login, self.show_register)
        self.register_view = RegisterView(self, self.show_login, self.on_register_success)
        self.dashboard_view = DashboardView(
            self,
            on_open_cameras=self.show_camera_placeholder,
            on_open_events=self.show_events,
            on_open_users=self.show_users,
        )
        self.events_view = EventsView(self, self.show_dashboard)
        self.user_view = UserManagementView(self, self.show_dashboard)

        self.frames["login"] = self.login_view
        self.frames["register"] = self.register_view
        self.frames["dashboard"] = self.dashboard_view
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

    def show_events(self):
        self._show_frame("events")

    def show_users(self):
        self._show_frame("users")

    def show_camera_placeholder(self):
        self._show_frame("dashboard")
        self.dashboard_view.focus_set()

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
