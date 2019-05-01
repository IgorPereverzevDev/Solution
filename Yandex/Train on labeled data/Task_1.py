import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import seaborn as sns
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D

# Считаем данные по росту и весу (weights_heights.csv, приложенный в задании) в объект Pandas DataFrame:
data = pd.read_csv('weights_heights.csv', index_col='Index')

# Построим гистограмму распределения роста подростков из выборки data.
# Используем метод plot для DataFrame data c аргументами y='Height' (это тот признак, распределение которого мы строим)

data.plot(y='Height', kind='hist',
          color='red', title='Height (inch.) distribution')

# Посмотрите на первые 5 записей с помощью метода head Pandas DataFrame.
# Нарисуйте гистограмму распределения веса с помощью метода plot Pandas DataFrame.
# Сделайте гистограмму зеленой, подпишите картинку.
data.head(5)
data.plot(y='Weight', kind='hist',
          color='green', title='Weight (f.) distribution')


# Один из эффективных методов первичного анализа данных - отображение попарных зависимостей признаков.
# Создается  𝑚×𝑚  графиков (m* - число признаков), где по диагонали рисуются гистограммы распределения признаков,
# а вне диагонали - scatter plots зависимости двух признаков.
# Это можно делать с помощью метода  𝑠𝑐𝑎𝑡𝑡𝑒𝑟_𝑚𝑎𝑡𝑟𝑖𝑥  Pandas Data Frame или *pairplot библиотеки Seaborn.
#
# Чтобы проиллюстрировать этот метод, интересней добавить третий признак.
# Создадим признак Индекс массы тела (BMI).
# Для этого воспользуемся удобной связкой метода apply Pandas DataFrame и lambda-функций Python.

def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / \
           (height_inch / METER_TO_INCH) ** 2


data['BMI'] = data.apply(lambda row: make_bmi(row['Height'],
                                              row['Weight']), axis=1)

# Постройте картинку, на которой будут отображены попарные зависимости признаков ,
# 'Height', 'Weight' и 'BMI' друг от друга. Используйте метод pairplot библиотеки Seaborn.

sns.pairplot(data)


# Часто при первичном анализе данных надо исследовать зависимость какого-то количественного признака от категориального
# (скажем, зарплаты от пола сотрудника).
# В этом помогут "ящики с усами" - boxplots библиотеки Seaborn.
# Box plot - это компактный способ показать статистики вещественного признака (среднее и квартили)
# по разным значениям категориального признака.
# Также помогает отслеживать "выбросы" - наблюдения,
# в которых значение данного вещественного признака сильно отличается от других.
#
# Создайте в DataFrame data новый признак weight_category,
# который будет иметь 3 значения: 1 – если вес меньше 120 фунтов.
# (~ 54 кг.), 3 - если вес больше или равен 150 фунтов (~68 кг.), 2 – в остальных случаях.
# Постройте «ящик с усами» (boxplot), демонстрирующий зависимость роста от весовой категории.
# Используйте метод boxplot библиотеки Seaborn и метод apply Pandas DataFrame.
# Подпишите ось y* меткой «Рост», ось *x – меткой «Весовая категория».

def weight_category(weight):
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    else:
        return 2


data['weight_cat'] = data['Weight'].apply(weight_category)

sns.boxplot(data=data, x='weight_cat', y='Height')

# Постройте scatter plot зависимости роста от веса,
# используя метод plot для Pandas DataFrame с аргументом kind='scatter'. Подпишите картинку.

data.plot(y='Height', x='Weight', kind='scatter',
          color='blue', title='Height (Weight) depending')


# Задание 2. Минимизация квадратичной ошибки

# В простейшей постановке задача прогноза значения вещественного признака
# по прочим признакам (задача восстановления регрессии) решается минимизацией квадратичной функции ошибки.
#
# Напишите функцию, которая по двум параметрам  𝑤0  и  𝑤1  вычисляет квадратичную ошибку приближения зависимости роста
# 𝑦  от веса  𝑥  прямой линией  𝑦=𝑤0+𝑤1∗𝑥 :
# 𝑒𝑟𝑟𝑜𝑟(𝑤0,𝑤1)=∑𝑖=1𝑛(𝑦𝑖−(𝑤0+𝑤1∗𝑥𝑖))2
#
# Здесь  𝑛  – число наблюдений в наборе данных,  𝑦𝑖  и  𝑥𝑖  – рост и вес  𝑖 -ого человека в наборе данных.

def error(w0, w1):
    s = 0.
    x = data['Weight']
    y = data['Height']
    for i in range(1, len(data.index)):
        s += (y[i] - w0 - w1 * x[i]) ** 2
    return s


# Итак, мы решаем задачу: как через облако точек, соответсвующих наблюдениям в нашем наборе данных,
# в пространстве признаков "Рост" и "Вес" провести прямую линию так, чтобы минимизировать функционал из п. 6.
# Для начала давайте отобразим хоть какие-то прямые и убедимся, что они плохо передают зависимость роста от веса.
#
# Проведите на графике из п. 5 Задания 1 две прямые, соответствующие значениям параметров ( 𝑤0,𝑤1)=(60,0.05)
# и ( 𝑤0,𝑤1)=(50,0.16).
# Используйте метод plot из matplotlib.pyplot, а также метод linspace библиотеки NumPy. Подпишите оси и график.

x = np.array(data['Weight'])
w0, w1 = 60, 0.05
y1 = [w0 + t * w1 for t in x]

w0, w1 = 50, 0.16
y2 = [w0 + t * w1 for t in x]

data.plot(y='Height', x='Weight', kind='scatter',
          color='blue', title='Height (Weight) depending')

plt.plot(x, y1, color="red", label="line1")
plt.plot(x, y2, color="green", label="line2")
plt.grid(True)
plt.legend(loc='upper left')

# Минимизация квадратичной функции ошибки - относительная простая задача, поскольку функция выпуклая.
# Для такой задачи существует много методов оптимизации.
# Посмотрим, как функция ошибки зависит от одного параметра (наклон прямой),
# если второй параметр (свободный член) зафиксировать.
#
# Постройте график зависимости функции ошибки, посчитанной в п. 6, от параметра  𝑤1  при  𝑤0  = 50.
# Подпишите оси и график.

w0 = 50.
w = np.arange(-0.5, 0.8, 0.1)

err = [error(w0, w1) for w1 in w]
plt.title('Error')
plt.xlabel('w1 ')
plt.ylabel('error(50,w1)')

plt.plot(w, err, color="red", label="function of error")
plt.legend()


# Теперь методом оптимизации найдем "оптимальный" наклон прямой, приближающей зависимость роста от веса,
# при фиксированном коэффициенте  𝑤0=50 .
#
# С помощью метода minimize_scalar из scipy.optimize найдите минимум функции, определенной в п. 6,
# для значений параметра  𝑤1  в диапазоне [-5,5].
# Проведите на графике из п. 5 Задания 1 прямую, соответствующую
# значениям параметров ( 𝑤0 ,  𝑤1 ) = (50,  𝑤1_𝑜𝑝𝑡 ), где  𝑤1_𝑜𝑝𝑡  – найденное в
# п.8 оптимальное значение параметра  𝑤1 .


def error50(w1):
    return error(50, w1)


min = minimize_scalar(error50, bounds=(-5, 5), method='bounded')
w1_opt = min.x

x = np.array(data['Weight'])

w0, w1 = 50, w1_opt
y = [w0 + t * w1 for t in x]

data.plot(y='Height', x='Weight', kind='scatter',
          color='blue', title='Height (Weight) depending')

plt.plot(x, y, color="red", label="lineOptimum")

plt.grid(True)
plt.legend(loc='upper left')

# При анализе многомерных данных человек часто хочет получить интуитивное представление
# о природе данных с помощью визуализации.
# Увы, при числе признаков больше 3 такие картинки нарисовать невозможно.
# На практике для визуализации данных в 2D и 3D в данных выделаяют 2 или, соответственно,
# 3 главные компоненты (как именно это делается - мы увидим далее в курсе)
# и отображают данные на плоскости или в объеме.
#
# Посмотрим, как в Python рисовать 3D картинки,
# на примере отображения функции  𝑧(𝑥,𝑦)=𝑠𝑖𝑛(𝑥2+𝑦2⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯√)  для значений  𝑥  и  𝑦
# из интервала [-5,5] c шагом 0.25.

fig = plt.figure()
ax = fig.gca(projection='3d')  # get current axis

# Создаем массивы NumPy с координатами точек по осям X и У.
# Используем метод meshgrid, при котором по векторам координат
# создается матрица координат. Задаем нужную функцию Z(x, y).
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

# Наконец, используем метод *plot_surface* объекта
# типа Axes3DSubplot. Также подписываем оси.
surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Постройте 3D-график зависимости функции ошибки, посчитанной в п.6 от параметров  𝑤0  и  𝑤1 .
# Подпишите ось  𝑥  меткой «Intercept», ось  𝑦  – меткой «Slope», a ось  𝑧  – меткой «Error».

fig = plt.figure()
ax = fig.gca(projection='3d')
w0 = np.arange(-5, 5, 0.25)
w1 = np.arange(-5, 5, 0.25)
W0, W1 = np.meshgrid(w0, w1)
E = error(W0, W1)


# С помощью метода minimize из scipy.optimize найдите минимум функции,
# определенной в п. 6, для значений параметра  𝑤0  в диапазоне [-100,100] и  𝑤1  - в диапазоне [-5, 5].
# Начальная точка – ( 𝑤0 ,  𝑤1 ) = (0, 0).
# Используйте метод оптимизации L-BFGS-B (аргумент method метода minimize).
# Проведите на графике из п. 5 Задания 1 прямую, соответствующую найденным оптимальным значениям параметров  𝑤0  и  𝑤1.
# Подпишите оси и график.

def error1(w):
    s = 0.
    x = data['Weight']
    y = data['Height']
    for i in range(1, len(data.index)):
        s += (y[i] - w[0] - w[1] * x[i]) ** 2
    return s


min = optimize.minimize(error1, np.array([0, 0]), method='L-BFGS-B', bounds=((-100, 100), (-5, 5)))
print(min.x, min.fun)

x = np.array(data['Weight'])

w0, w1 = min.x
y = [w0 + t * w1 for t in x]

data.plot(y='Height', x='Weight', kind='scatter',
          color='blue', title='Height (Weight) depending')

plt.plot(x, y, color="red", label="lineOptimum")

plt.grid(True)
plt.legend(loc='upper left')
