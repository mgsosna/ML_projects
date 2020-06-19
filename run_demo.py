from demos import run_linear_regression_demo

model = int(input("Do you want to run linear (1) or logistic (2) demo?"))

if model not in [1, 2]:
    print("Come on!")
    quit()

if model == 1:
    run_linear_regression_demo()