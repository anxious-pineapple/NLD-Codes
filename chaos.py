import matplotlib.pyplot as plt

A=2
xo=0.4

delta_A=0.001
A_dict=[]

values=[]
while A<=4:
    x=xo*(1-xo)*A
    for i in range(1000):                     #for transients
        x=x*(1-x)*A
    for i in range(100):
        values.append(x)
        A_dict.append(A)
        x=x*(1-x)*A
    A+=delta_A

fig = plt.figure(figsize=(50, 50))
plt.plot(A_dict,values,'o')
plt.title('Bifurcation plot')
plt.savefig('bifurcation_plt3.png')
plt.show()
plt.close()