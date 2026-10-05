class User:
    def __init__(self, user_id: str, name: str, is_family_member: bool = True):
        self.user_id = user_id
        self.name = name
        self.is_family_member = is_family_member
