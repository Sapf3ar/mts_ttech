p = [0.2956,0.2975, 0.3042, 0.3097, 0.311, 0.3097, 0.3069, 0.3116 ]
m2 = [i/100 for i in range(80, 120, 5)]
print(m2)
for j in range(len(p)):
    print(p[j]**2/(2*m2[j]))