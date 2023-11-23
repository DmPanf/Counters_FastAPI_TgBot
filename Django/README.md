##

## ⚛️ Алгоритм действий по разворачиванию Клиент-серверной архитектуры на своем ПК:
0. После 16 ноября все обращения к OpenAI только через VPN ‼️
1. Создаешь папку проект [mkdir Django && cd Django] и туда вытаскиваешь весь архив [unzip django.zip]:
2. Перешел в папку FastAPI [cd fastapi]:
- добавить в файл .env свой OpenAI_API_KEY [nano .env] или [code .]
- [python3 -m venv venv] создать виртуальное окружение
- [source venv/bin/activate] активировать его
- [pip install -U pip] обновить установщик бибилиотек
- [pip install -r requirements.txt] установить набор библиотек для Сервера
- [uvicorn main:app --host 0.0.0.0 --port 5000] проверяешь через Swagger работу GPT:
🌐 [http://localhost:5000/docs]
3. Перешел в папку Django [cd ../django_chatbot]:
- [python3 -m venv venv] создать виртуальное окружение
- [source venv/bin/activate] активировать его
- [pip install -U pip] обновить установщик бибилиотек
- [pip install -r requirements.txt] установить набор библиотек для Django
- [python manage.py runserver] запускаешь Django
- проверяешь работу клиента на своем браузере:
🌐 [http://localhost:8000]
