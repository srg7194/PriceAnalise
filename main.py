from configurator import Config

from addition.log import logging, CustomLogger, log_decorator
logger = CustomLogger(level=logging.DEBUG, name='MAIN')

### Параметры для запуска
# Базовые параметры
creds_path = 'files/secret/creds.json'
local_start = True

# Запускаемые модули
start_telegram = False
start_clean_data_loader = True
start_pa_analiser = False
start_order_tester = False
start_trader = False


if __name__ == '__main__':
    # Получение файлов конфигурации
    logger.log_info('Запуск')
    logger.log_debug('Получение файла конфигурации')
    config = Config(creds_path, local_start).get_config()
    logger.log_info('Окончание работы')


# todo - Проблема в логировании, Всегда отражаются одинаковые параметры ('file': 'log', 'string': 91)