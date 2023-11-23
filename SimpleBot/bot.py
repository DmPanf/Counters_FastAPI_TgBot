from aiogram import Bot, Dispatcher, types
import logging
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json
from dotenv import load_dotenv
from io import BytesIO
import base64
import os
import csv
from datetime import datetime
from text_module import commands, help_text

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("TOKEN")
ADMIN_CHAT_ID = os.getenv("CHAT_ID")
API_URL = os.getenv("API_URL")
mdl_name = os.getenv("mdl_name")  #  получаем значение переменной mdl_name из .env файла

log_folder = os.getenv("LOG_DIR")
log_file = os.getenv("LOG_FILE")

  
if not API_TOKEN:
    raise ValueError("TG_TOKEN is not set in the environment variables or .env file!")

if not os.path.exists(log_folder):  # Создание директории для логов, если её нет
    os.makedirs(log_folder)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{log_folder}/{log_file}"),
        logging.StreamHandler()
    ]
)


bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

async def on_startup(dp):
    bot_info = await dp.bot.get_me()
    bot_name = bot_info.username
    user_id = bot_info.id
    await dp.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f'✴️ Бот @{bot_name} запущен!\n🈲 Main Server: ciet')
    print(f'✴️  Бот {bot_name} запущен!')


#async def fetch(session, url):
#    async with session.get(url) as response:  # Использование асинхронного менеджера контекста для выполнения GET-запроса
#        return await response.text()          # Возврат текстового содержимого ответа

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
                # return await response.json()
            else:
                raise Exception(f"Error: {response.status} - {await response.text()}")
    except Exception as e:
        print(f"fetch ------ Error occurred: {e}")
        raise e
    

def save_to_csv(results, user_id, user_name, phone):
    with open('./data/save_images.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), user_id, user_name, phone, *results.values()])


@dp.message_handler(commands=['start','info'])
async def send_info(message: types.Message):
    # API_URL = os.getenv("API_URL")  #  получаем значение переменной из .env файла
    # print(f' ... API_URL/info: {API_URL}')
    timeout = ClientTimeout(total=5)  # Устанавливаем общий таймаут в 5 секунд
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            info = await fetch(session, f'{API_URL}/info') 
            data = json.loads(info)["Project 2023"]
            # await message.answer(f"🔰 <b>Project Info:</b>\n<pre>{data}</pre>", parse_mode="HTML", reply_markup=keyboard)
            await message.answer(f"🔰 <b>Project Info:</b>\n<pre>{data}</pre>", parse_mode="HTML")
        except Exception as e:
            print(f"info ------ Error occurred: {e}")
            # await message.answer(f"⛔️ Сервер <b>FastAPI</b> недоступен!\n{e}", parse_mode="HTML", reply_markup=keyboard)
            await message.answer(f"⛔️ Сервер <b>FastAPI</b> недоступен!\n{e}", parse_mode="HTML")


# +++++++++++++++ PHOTO +++++++++++++++
# handler for receiving images and making POST requests to FastAPI
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def process_image(message: types.Message):
    file_id = message.photo[-1].file_id  # Получение файла изображения 
    file = await bot.get_file(file_id)  # Получение объекта файла 
    file_path = file.file_path  # Получение пути к файлу 

    image_data = await bot.download_file(file_path)  # Загрузка изображения

    url = f'{API_URL}/predict'  # Отправка изображения на сервер 

    timeout = ClientTimeout(total=15)  # Устанавливаем общий таймаут в 15 секунд
    async with aiohttp.ClientSession(timeout=timeout) as session:  # Инициализация асинхронной сессии HTTP
        form = aiohttp.FormData()  # Инициализация формы 
        form.add_field('file', image_data, filename='input_image.jpg', content_type='image/jpg')  # Добавление поля в форму 
        form.add_field('mdl_name', mdl_name)
        try:
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    response_json = await response.json()
                    img_str = response_json['image']
                    results = response_json['results'] 

                    add_info = f"\n<b>📟 Показание прибора:</b> [<code>{results['counter']}</code>]" \
                               f"\n<b>⏱ Время обработки:</b> {results['inference']}ms" \
                               f"\n<b>⚖️ Модель:</b> {results['model_name']}" \
                               f"\n<b>🔍 Распознано {results['object_count']} объектов</b>" \
                               f"\n<b>📆 Дата/Время:</b> <code>{results['current_time']}</code>" \
                               f"\n<b>📸 Изображение {results['wh_check']}</b>" \
                               f"\n<b>💾 Имя файла:</b> <code>{results['file_name']}</code>"
                    
                    # Save to CSV
                    user_id = message.from_user.id
                    user_name = message.from_user.username
                    phone = message.contact.phone_number if message.contact else 'N/A'  # Adjust as needed
                    save_to_csv(results, user_id, user_name, phone)

                    output_image_data = BytesIO(base64.b64decode(img_str))
                    output_image_data.seek(0)
                    await message.reply_photo(photo=output_image_data, caption=add_info, parse_mode="HTML")
        except KeyError as key_error:
            print(f"Key error: {key_error}")
            await message.answer(f"🛑 Ошибка обработки: {key_error}")
        except Exception as e:
            print(f"predict ------ ⛔️ Сервер недоступен: {e}")
            await message.answer(f"⛔️ Сервер <b>FastAPI [{API_URL}]</b> недоступен!\n⌛️ Timeout=<b>{timeout} s</b>\n{e}", parse_mode="HTML")

# +++++++++++++++ PHOTO +++++++++++++++


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    await message.answer(help_text, parse_mode="Markdown")


# +++++++++++++ MAIN +++++++++++++
async def main():
    await on_startup(dp)
    await dp.start_polling()  # Запуск бота 

if __name__ == '__main__':
    asyncio.run(main())
