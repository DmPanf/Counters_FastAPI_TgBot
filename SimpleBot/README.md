## 

## Dockerfile

## docker-compose.yml

##


## Pydantic обеспечивает валидацию данных и их автоматическое преобразование в JSON.

<code>
from pydantic import BaseModel
class ResponseModel(BaseModel):
    image: str
    results: dict
# ...
response = ResponseModel(image=img_str, results=image_data)
return response
</code>
