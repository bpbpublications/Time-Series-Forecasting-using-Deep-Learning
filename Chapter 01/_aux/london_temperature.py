import matplotlib.pyplot as plt

temp = [
    6.7,
    6.4,
    6.8,
    10.3,
    12.6,
    15.1,
    15.7,
    17.2,
    14,
    10.5,
    8.7,
    6
]

plt.plot(temp)
plt.title('London Average Temperature in 2020')
plt.ylabel('Temperature')
plt.xlabel('Month')
plt.show()
