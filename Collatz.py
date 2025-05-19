# libraries
# import math

# global variables
N_range: int = 12
phi_range: int = 4
# main function


def main() -> None:
    # create a numbers list
    number_list: list = []

    # print the first numbers
    # for n in range(0, N_range):
    #     number_list.append(2**n)

    # print the first odds
    for n in range(1, N_range):
        next_number: int = 1
        for i in range(1, n):
            next_number += 4**(n - i)
        number_list.append(next_number)
    #     number_list.append(next_number * 2)
    #     number_list.append(next_number * 4)

    # second odds
    # even powers of 2
    # for phi in range(1, 3):
    #     for n in range(1, N_range):
    #         next_number = 1
    #         for i in range(1, n + 2 * phi):
    #             next_number += i * 4**(n - i + 2 * phi)
    #         number_list.append(next_number)

    # odd powers of 2
    # for phi in range(0, 3):
    #     for n_prime in range(1, N_range):
    #         next_number = 1
    #         for i in range(1, 2 * n_prime + phi):
    #             next_number += i * 4**(2 * n_prime - i + phi)
    #         for i in range(1, phi):
    #             next_number += 4**(phi - i)
    #         number_list.append(next_number)

    # third odds
    # even - even 
    # for phi in range(1, 4):
    #     for n in range(3, N_range):
    #         next_number = 1
    #         for i in range(1, n):
                
    #         number_list.append(next_number)

    # sort numbers and print
    number_list.sort()
    print(number_list)
    return

# special classes


# special functions


# execute if main

if __name__ == "__main__":
    main()
