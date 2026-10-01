# Учебный пет-проект EDA-анализ. 
# Составить анализ аппартаментов (распределение цен на квартиры, зависимость цены от жил. площади кв., тепловая карты)
# Очистить данные и вывести графики используя учебную БД apartmnets.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Блок внешнего вида графиков и настройка шифта
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)

print("--- ШАГ 1: ЗАГРУЗКА И ИНФОРМАЦИЯ ---")
df = pd.read_csv('apartments.csv')

print("\nСтруктура датасета:")
print(df.info())

print("\nПервые 5 строк таблицы:")
print(df.head())

# Блок очистки данных
print("\n--- ШАГ 2: ОЧИСТКА ДАННЫХ ---")

# Поиск и удаление дубликатов
duplicate_count = df.duplicated().sum()
print(f"Найдено полных дубликатов строк: {duplicate_count}")
df.drop_duplicates(inplace=True)
print("Дубликаты успешно удалены.")

print("\nКоличество пропусков в каждом столбце до очистки:")
print(df.isnull().sum())

# Вычисляем медиану жилой площади
median_area = df['Жилая площадь'].median()
print(f"Медианное значение жилой площади: {median_area} кв.м")

# Заполняем пропуски вычисленной медианой
df['Жилая площадь'] = df['Жилая площадь'].fillna(median_area)
print("Пропуски успешно заполнены.")


# Одномерный анализ (Распределение цен)
print("\n--- ШАГ 3: ПОСТРОЕНИЕ ГРАФИКА РАСПРЕДЕЛЕНИЯ ЦЕН ---")

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Цена', kde=True, color='royalblue', bins=20) # kde=True добавляет плавную линию плотности распределения

plt.title('Распределение цен на квартиры', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Цена (в млн рублей)', fontsize=12)
plt.ylabel('Количество квартир', fontsize=12)
plt.show() # Выводим график на экран

# двумерный анализ (Связь площади и цены)

print("\n--- ШАГ 4: ИССЛЕДОВАНИЕ ВЗАИМОСВЯЗИ ПРИЗНАКОВ ---")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Жилая площадь', y='Цена', color='darkorange', s=70, alpha=0.8) # Диаграмма рассеяния

# Находим аномальную строку в коде для визуального выделения
anomaly = df[(df['Жилая площадь'] < 40) & (df['Цена'] > 20)]
if not anomaly.empty:
    plt.scatter(anomaly['Жилая площадь'], anomaly['Цена'], color='red', s=150, facecolors='none', edgecolors='red', linewidths=2)
    plt.text(anomaly['Жилая площадь'].values[0] + 2, anomaly['Цена'].values[0], 'Выброс (Аномалия)', color='red', weight='bold')

plt.title('Зависимость цены от жилой площади квартиры', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Жилая площадь (кв.м)', fontsize=12)
plt.ylabel('Цена (млн руб)', fontsize=12)
plt.show()

# Корреляционный анализ
print("\n--- ШАГ 5: МАТРИЦА КОРРЕЛЯЦИИ ---")

plt.figure(figsize=(8, 5))
correlation_matrix = df.corr() # Строим тепловую карту корреляции Пирсона
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)

plt.title('Тепловая карта корреляции признаков', fontsize=14, fontweight='bold', pad=15)
plt.show()

print("EDA завершен успешно!")