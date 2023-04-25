from abc import ABC
from abc import abstractmethod
from pathlib import Path


class FileIgnore(ABC):
    def write_file(self, file: Path) -> None:
        content = self.get_content()
        file.write_text(content)

    @abstractmethod
    def get_content(self) -> str:
        pass


class FileIgnorePath(FileIgnore):
    def __init__(self, path: Path):
        self.path = path

    def get_content(self) -> str:
        return self.path.read_text()


class FileIgnoreString(FileIgnore):
    def __init__(self, content: str):
        self.content = content

    def get_content(self) -> str:
        return self.content


class FileIgnoreListString(FileIgnore):
    def __init__(self, content: list[str]):
        self.content = content

    def get_content(self) -> str:
        return "\n".join(self.content)


class FileIgnoreDefault(FileIgnoreListString):
    def __init__(self):
        return super().__init__(["*.csv", "*.xls", "*.png", "*.pyc", "jpg"])
