# План

- **Модуль получения данных для анализа**
  - Криптовалютный рынок
  - Фондовый рынок

- **Эмулятор торговли**
  - Модуль расчета сделок с комиссиями и проскальзываниями
  - Хранилище ордеров
  - Модуль генерации цен внутри свечи
  - Файл для работы с таймфреймами Эксель

- **Модуль расчета параметров**
  - Уровни
  - Индикаторы
  - Оценка влияния трендов между таймфреймами

# Идеи
# Задачи
# Баги
# Структуры
- **creds.json** - Файл с учетными данными для подключения

{
  "telegram": {
    "token": "****"
  },
  "config": {
    "link": "https://docs.google.com/spreadsheets/d/1cf_u6SFea6HJjzkL0RnBAigvSuOfaDa5y1Il4bIxxrI/edit?gid=339803300#gid=339803300",
    "path": "files/secret/google.json"
  },
  "system": {
    "auth_type": "logopas",
    "description": "Database for system usage",
    "host": "system",
    "login": "****",
    "password": "****",
    "port": "****",
    "service": "system",
    "subtype": "sqlite",
    "type": "database"
  }
}

- **google.json** - Файл с учетными данными для подключения к GoogleApi 
(https://console.cloud.google.com/apis/credentials?project=priceanalise)

{
  "type": "service_account",
  "project_id": "*****",
  "private_key_id": "*****",
  "private_key": "*****",
  "client_email": "*****@*****.iam.gserviceaccount.com",
  "client_id": "*****",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/personal-google-api%40priceanalise.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
