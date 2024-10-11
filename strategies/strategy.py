class Strategy:
    def execute(self, data):
        raise NotImplementedError("Each strategy must implement its own execute method.")
