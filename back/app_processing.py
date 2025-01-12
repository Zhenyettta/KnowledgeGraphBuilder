def process_text(file_path, labels=None):
    """
    Обробляє текстовий файл та витягує дані на основі переданих labels.

    :param file_path: Шлях до текстового файлу.
    :param labels: Список labels для витягання даних (опціонально).
    :return: Словник із результатами обробки.
    """
    # Читання тексту з файлу
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Основна обробка тексту
    processed_data = {
        "word_count": len(content.split()),
        "line_count": len(content.splitlines()),
        "summary": content[:100] + "..." if len(content) > 100 else content
    }

    # Додаткова обробка з labels
    if labels:
        # Простий приклад пошуку слів, які збігаються з labels
        found_labels = {label: content.lower().count(label.lower()) for label in labels}
        processed_data["label_matches"] = found_labels

    return processed_data
