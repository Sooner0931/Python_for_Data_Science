
# ***matplotlib***
# ***matplotlib***
# ***matplotlib***

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

print('all libs for data viz has been imported.')
print(f'\n')


x = np.linspace(0, 10)
print(x)
print(type(x))
print(f'\n')


y = np.cos(x)
print(y)
print(type(y))
print(f'\n')


plt.plot(x, y)
plt.show()
print(f'\n')


z = np.sin(x)
plt.plot(x,y, x,z)
plt.show()
print(f'\n')

plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Trignometry')
plt.legend(['y', 'z'], loc = 1)
plt.plot(x,y, x,z)
plt.show()
print(f'\n')


plt.subplot(1, 2, 1)
plt.plot(x, y, 'r-x')
plt.subplot(1, 2, 2)
plt.plot(x, z, 'g-o')
plt.tight_layout()
plt.show()



x = np.arange(30)
print(x)
print(f'\n')

y = x ** 2
z = 2 * x**2 + 3 * x
plt.subplot(2,1,1)
plt.plot(x, y, 'r-o')
plt.subplot(2,1,2)
plt.plot(x, z, 'g-o')
plt.show()
print(f'\n')


plt.plot(x,y,'r-o', x,z,'g-o')
plt.show()



# ***seaborn***
# ***seaborn***
# ***seaborn***

# import seaborn as sns
df = sns.load_dataset('tips')
print(df.head())
print(df.describe())
print(type(df))
print(df.index)
print(df.columns)
print(df.shape)
print(f'\n')



sns.distplot(df['total_bill'])
plt.show()
sns.distplot(df['total_bill'], bins = 30, kde = False)
plt.show()



sns.relplot(x = 'total_bill', y = 'tip', data = df)
plt.show()

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'sex')
plt.show()

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker')
plt.show()

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'day')
plt.show()

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'time')
plt.show()

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'size')
plt.show()




sns.pairplot(df, hue = 'sex')
plt.show()

sns.pairplot(df, hue = 'time', palette = 'coolwarm')
plt.show()




sns.countplot(x = 'sex', data = df)
plt.show()
print(df.groupby('sex').count())
print(f'\n')

sns.countplot(x = 'day', data = df)
plt.show()





# ***pandas***
# ***pandas***
# ***pandas***

d = {'A':np.random.rand(5),
     'B':np.random.rand(5),
     'C':np.random.rand(5),
     'D':np.random.rand(5),
    }
df = pd.DataFrame(d)

df.hist()
plt.show()

df['C'].hist()
plt.show()





df.plot.area()
plt.show()

df.plot.area(alpha = 0.2)
plt.show()




df.plot.bar()
plt.show()

df.plot.bar(stacked = True)
plt.show()




df.plot.line()
plt.show()

df.plot.line(y = ["C"])
plt.show()

df.plot.line(y = ['B', 'C'])
plt.show()












