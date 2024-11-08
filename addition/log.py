import logging
import datetime
import os
import sys
from colorama import Fore, Style, init
from sqlalchemy.util import decorator


class CustomLogger:
    """
    Класс CustomLogger предоставляет настраиваемый логгер с поддержкой цветного вывода в консоль.

    Аргументы:
    - name: имя логгера
    - log_file_path: путь к директории, где будет сохранен файл лога
    - console_log: опция вывода в консоль (по умолчанию True)
    - file_log: опция сохранения лога в файл (по умолчанию False)

    Уровни логирования
    - Debug (10): самый низкий уровень логирования, предназначенный для отладочных сообщений, для вывода диагностической информации о приложении.
    - Info (20): этот уровень предназначен для вывода данных о фрагментах кода, работающих так, как ожидается.
    - Warning (30): этот уровень логирования предусматривает вывод предупреждений, он применяется для записи сведений о событиях, на которые программист обычно обращает внимание. Такие события вполне могут привести к проблемам при работе приложения. Если явно не задать уровень логирования — по умолчанию используется именно warning.
    - Error (40): этот уровень логирования предусматривает вывод сведений об ошибках — о том, что часть приложения работает не так как ожидается, о том, что программа не смогла правильно выполниться.
    - Critical (50): этот уровень используется для вывода сведений об очень серьёзных ошибках, наличие которых угрожает нормальному функционированию всего приложения. Если не исправить такую ошибку — это может привести к тому, что приложение прекратит работу.
    """

    def __init__(self, name, level=logging.DEBUG, log_file_path=False, console_log=True, file_log=False):
        """
        Инициализирует экземпляр класса CustomLogger.
        Пример использования:
            logger = CustomLogger(name='example_logger', log_file_path='/path/to/logs', console_log=True, file_log=True)
            logger.log_info('This is an informational message.')

        Аргументы:
            - name: имя логгера
            - log_file_path: путь к директории, где будет сохранен файл лога
            - console_log: опция вывода в консоль (по умолчанию True)
            - file_log: опция сохранения лога в файл (по умолчанию False)
        """
        # Инициализация логгера с настройками
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self.console_log = console_log
        self.file_log = file_log
        self.name = name

        # Генерация пути к файлу лога с учетом времени
        log_file_path = f'{log_file_path}/log_{datetime.datetime.now().replace(microsecond=0)}.log'

        # Форматтер
        # formatter = logging.Formatter(f"{{'time': '%(asctime)s', 'logger': '%(name)s', 'level': '%(levelname)s', "
        #                               f"'file': '%(module)s', 'string': %(lineno)d, 'message': '%(message)s'}}"
        formatter = logging.Formatter(f"{{'time': '%(asctime)s', 'logger': '%(name)s', 'level': '%(levelname)s', "
                                      f"'message': %(message)s}}")

        # Обработчик для консоли
        if self.console_log:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Обработчик для файла
        if self.file_log:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Инициализация colorama для цветного вывода в консоль
        init()

    def set_log_level(self, level):
        # Установка уровня логирования для логгера
        self.logger.setLevel(level)

    @staticmethod
    def add_info_to_msg(msg, add=False):
        if add:
            # Добавление дополнительной информации к сообщению лога
            add = {f'{k}': f'{v}' for k, v in add.items()}
            add = str(add).replace('{', '').replace('}', '') if add else ''  # .replace("'", '')
            if not msg or msg in (''):
                msg = False
            msg = f'{msg}, {add}'
        if msg[:1] != '{':
            return str({'text': msg})  # .replace("'", '')
        return msg

    def log_debug(self, message, add=False):
        self.logger.debug(f"{Fore.WHITE}{Style.BRIGHT}{self.add_info_to_msg(message, add)}{Style.RESET_ALL}")

    def log_info(self, message, add=False):
        self.logger.info(f"{Fore.GREEN}{self.add_info_to_msg(message, add)}{Style.RESET_ALL}")

    def log_warning(self, message, add=False):
        self.logger.warning(f"{Fore.YELLOW}{self.add_info_to_msg(message, add)}{Style.RESET_ALL}")

    def log_error(self, message, add=False):
        self.logger.error(f"{Fore.RED}{Style.BRIGHT}{self.add_info_to_msg(message, add)}{Style.RESET_ALL}")

    def log_critical(self, message, add=False):
        self.logger.critical(f"{Fore.LIGHTRED_EX}{Style.BRIGHT}{self.add_info_to_msg(message, add)}{Style.RESET_ALL}")


def log_decorator(logger, good='DEBUG', bad='ERROR', debug=False):
    def _choise(logger, log_level):
        if log_level == 'DEBUG':
            return logger.log_debug
        elif log_level == "INFO":
            return logger.log_info
        elif log_level == "WARNING":
            return logger.log_warning
        elif log_level == "ERROR":
            return logger.log_error
        elif log_level == "CRITICAL":
            return logger.log_critical
        else:
            logger.log_error(f'''error: Incorrect logger choise. <<{log_level}>> type selected''')
            return logger.log_debug

    good_level_logger = _choise(logger, good)
    bad_level_logger = _choise(logger, bad)
    decorator = False
    return decorator
