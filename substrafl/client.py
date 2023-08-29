from substra import Client as SubstraClient


class Client(SubstraClient):
    def __init__(*args, **kwargs):
        if "backend_type" in kwargs:
            if kwargs["backend_type"] == "simu":
                # We remove it not to raise Errors for unrecognized backend
                kwargs.pop("backend_type")
                # We init it with default backend which is subprocess
                super().__init__(*args, **kwargs)
                # We tag it with a mark
                self.is_simu = True
        else:
            super().__init__(*args, **kwargs)
            # We tag it with a mark
            self.is_simu = False

