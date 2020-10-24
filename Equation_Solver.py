
def solve(term, y="ns", step=1, modus="reell"):
    term = term.replace(" ", "")
    z = 0
    equation = []

    for character in term:
        if character == "i":
            equation.append("1j")
        else:
            equation.append(character)

    if y == "ns":
        temp_equation = equation.copy()

        while True:
            for i in range(len(equation)):
                if equation[i] == "x":
                    temp_equation[i] = equation[i].replace("x", str(z))

            solution = str(temp_equation).strip("[]()").replace(",", "")
            if modus == "reell":
                if round(eval(eval(solution)), 3) == 0:
                    break
            else:
                if eval(eval(solution)) == 0j:
                    break

            z = z + step

        return z, -z
    else:
        pass


print(solve("x*x + 3*x + 40*i**2", modus="komplex"))












