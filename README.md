# sber_test_bot
  Задача создания простого вопросно-ответного бота.
  База вопрос-ответ и тематики лежит в файле BC_base.rar и themes_base.json соответсвенно
  Изходный файл vk.rar
  w2v_cards_model.rar - файл содержащий модель w2v обученную на диалогах из vk.ru
  kmeans_clustering100.sav-файл содержайщий pickle модели кластеризации для тематик
  chat_bot_clusters.ipynb - файл демонстрации предобработки вопросов и обучения кластеризатора по темам
  Class_bot_prototype.ipynb - содержит прототип класса для обработки запросов бота: ранжирование ответов, confidence ответов, ранжирование кластеров и их confidence
  bot.py и engine3.py-файлы прототипа бота для телеграмм и исполняемого класса для функционала ранжирования ответов и выбора тем соответственно
  PyQT_interface.ipynb и gui.py - файлы демонстрирующие работу бота локально на осное PyQT интерфейса (требуется либа PyQT)
