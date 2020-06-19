from python import DataGenerator, LinearRegressor


def get_int_from_user(message: str) -> int:
    """
    | Get number of observations from the user.
    """
    user_input = None
    n_iter = 0

    while not isinstance(user_input, int) and n_iter < 3:
        user_input = input(message)

        try:
            user_input = int(user_input)
        except ValueError:
            print(f"{user_input} is not an integer; please type in an integer.")

        n_iter += 1

    if not isinstance(user_input, int):
        print("Alright, let's try this again some other day.")
        exit()

    return user_input

####################################################################
def run_linear_regression_demo():
    dg = DataGenerator()
    lr = LinearRegressor()

    print("Welcome to the linear regression demo!")

    n_obs = get_int_from_user(message="Number of observations: ")
    n_feat = get_int_from_user(message="Number of features: ")

    print("Alright, see ya")
