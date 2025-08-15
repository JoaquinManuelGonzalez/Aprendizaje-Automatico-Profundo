
class GradientDescent:
    def __init__(self, f, fp) -> None:
        self.f = f
        self.fp = fp

    def fit(self,eta,epsilon,x):
        str = "i".ljust(5) + "x".ljust(10) + "error".ljust(10) + "\n"
        error = 1
        i = 1
        MAX_ITE=100
        while i<=MAX_ITE and error > epsilon:
            xold = x
            x = xold - eta * self.fp(xold)
            error = abs(self.f(x) - self.f(xold))
            str += f"{i}".ljust(5) + f"{x:.4f}".ljust(10) + f"{error:.4f}".ljust(10) + "\n"
            i += 1
        return str

grd=GradientDescent(lambda x: -2/(x**2+1), lambda x: 4*x/((x**4+2*x**2+1)))
print("a")
print(grd.fit(0.2,10e-06,1.65))
print("b")
print(grd.fit(0.3,10e-06,-3.5))
print("c")
print(grd.fit(0.1,10e-06,1.5))

