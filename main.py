import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


trips_data = pandas.read_excel("trips_data.xlsx")



#print(trips_data.head(10))

# trips_data.salary.hist()
# plt.show() #отображение гистрограммы
#
#
# print(trips_data.age.describe())
# trips_data.age.hist()
# plt.show()

trips_data.vacation_preference.value_counts().plot(kind = 'bar')
#plt.show()

#print(trips_data[(trips_data.salary < 40000) & (trips_data.age.between(45,60))])
#print(trips_data[trips_data.age.between(45,60)])
X = trips_data.drop("target", axis=1)
Y = trips_data.target

X_dummies = pandas.get_dummies(X, columns=['city', 'vacation_preference', 'transport_preference'])
rfc = RandomForestClassifier()
rfc.fit(X_dummies, Y)
print(rfc.score(X_dummies, Y))

tst = {'salary': [130000], 'age': [30], 'family_members': [1], 'city_Екатеринбург': [0], 'city_Киев': [0], 'city_Краснодар': [1], 'city_Минск': [0], 'city_Москва': [0], 'city_Новосибирск': [0], 'city_Омск': [0], 'city_Петербург': [0], 'city_Томск': [0], 'city_Хабаровск': [0], 'city_Ярославль': [0], 'vacation_preference_Архитектура': [0], 'vacation_preference_Ночные клубы': [0], 'vacation_preference_Пляжный отдых': [0], 'vacation_preference_Шоппинг': [1], 'transport_preference_Автомобиль': [0], 'transport_preference_Космический корабль': [0], 'transport_preference_Морской транспорт': [0], 'transport_preference_Поезд': [0], 'transport_preference_Самолет': [1]}

example_df = pandas.DataFrame(data=tst, columns=X_dummies.columns)
#print(example_df)
print(rfc.predict(example_df))
print(rfc.predict_proba(example_df))