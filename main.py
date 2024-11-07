from configurator import Config

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
    config = Config(creds_path, local_start).get_config()