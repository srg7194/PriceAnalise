from configurator import Loader, ConfiguratorTelegram
import pprint

# Параметры для запуска
creds_path = 'files/secret/creds.json'
local_start = False

start_telegram = False
start_clean_data_loader = True
start_pa_analiser = False
start_order_tester = False
start_trader = False


if __name__ == '__main__':
    # Получение файлов конфигурации
    config_anal = Loader(creds_path, local_start).get_config()
    pprint.pprint(config_anal, sort_dicts=False)

    # Запуск телеграм бота
    if start_telegram:
        config_telegram = ConfiguratorTelegram(creds_path, local_start).get_config()
        pass

    # Запуск модуля получения данных
    if start_clean_data_loader:
        pass

    # Запуск модуля PriceAction (паттерны, индикаторы, уровни, тренды)
    if start_pa_analiser:
        pass

    # Запуск модуля тестирования стратегий
    if start_order_tester:
        pass

    # Запуск модуля торговли
    if start_trader:
        pass
