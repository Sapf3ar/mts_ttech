p = [0.2956,0.2975, 0.3042, 0.3097, 0.311, 0.3097, 0.3069, 0.3116 ]
m2 = [i/100 for i in range(80, 120, 5)]
# print(m2)
# for j in range(len(p)):
    # print(p[j]**2/(2*m2[j]))


p1 = [0.2344, 0.2336, 0.236,0.2368,  0.2368, 0.236, 0.2332, 0.2372]
ek = [p1[j]**2/(2*0.4) + p[j]**2/(2*m2[j]) for j in range(len(p))]
print(sum(ek)/len(ek))

ek1 = [p1[j]**2/(2*0.4)  for j in range(len(p))]
print(sum(ek1)/len(ek1))

ek2 = [ p[j]**2/(2*m2[j]) for j in range(len(p))]

print(sum(ek2)/len(ek2))
