# План реализации слоя репозиториев

## Структура каталогов

```
app/
  repositories/
    __init__.py
    document_repository.py
```

## Класс DocumentRepository

### Методы

1. `find_similar(embedding, limit)` - для поиска ближайших векторов через оператор <=>
2. `get_by_id(doc_id)` - для получения документа
3. `semantic_search(query_embedding, limit)` - семантический поиск
4. `lexical_search(query_text, limit)` - лексический поиск
5. `save_document(document)` - сохранение документа
6. `delete_document(doc_id)` - удаление документа

## Интеграция с FastAPI

Использование Depends для получения сессии БД и вызова методов репозитория.

## Перенос SQL-запросов

### Из app/api/search.py
- `lexical_search` - перенести в DocumentRepository
- `semantic_search` - перенести в DocumentRepository

### Из app/services/hybrid_search.py
- `_lexical_search` - перенести в DocumentRepository
- `_semantic_search` - перенести в DocumentRepository

## Проверка безопасности

Убедиться, что все SQL-запросы используют параметры для предотвращения SQL-инъекций.