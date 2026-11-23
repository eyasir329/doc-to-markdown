class ConfigParser:
    def __init__(self, cli):
        self.cli = cli

    def generate_config_dict(self):
        return dict(self.cli)

    def get_processors(self):
        return []

    def get_renderer(self):
        return None

    def get_llm_service(self):
        return None
