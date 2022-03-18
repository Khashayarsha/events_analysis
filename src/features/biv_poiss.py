import math 


def prob_mass_func(x,y, l1, l2, l3):
    #print(f"x,y before int-ing: {x},{y} of types {type(x), type(y)}")
    x = int(x)
    y = int(y)

    #print(f"x,y after int-ing: {x},{y} of types {type(x), type(y)}")


    if x < 0 or y < 0:
        print("ERROR, NEGATIVE VALUE FOR GOALS DETECTED IN BIV_POISSON.PMF")

    result = 0
    product1 = math.exp(-(l1+l2+l3)) * ((l1**x) /
                                        math.factorial(x)) * ((l2**y)/math.factorial(y))

    for k in range(0, min(x, y)+1):

        product2 = math.comb(x, k) * math.comb(y, k) * \
            math.factorial(k) * ((l3 / (l1 * l2))**k)
        result = result + (product1 * product2)

    if result < 0:
        print(f"ERROR IN EVALUATION OF PROBABILITY MASS FUNCTION: \n prob = {result} using parameters x={x},y={y}, l1,l2,l3 = {l1,l1,l3}")
    return max(0.001,result)


def link_function(ai, aj, bi, bj, delta):  #

    try:
        l1 = min(50, math.exp(delta + ai - bj))
        l1 = max(0.1, l1)
    except OverflowError:
        l1 = 50

    try:
        l2 = min(50, math.exp(aj - bi))
        l2 = max(0.1, l2)
    except OverflowError:
        l2 = 50

    return l1, l2


def S(q, x, y, l1, l2, l3):

    limit = min(x, y)
    res = 0

    for k in range(0, limit+1):

        temp = math.comb(x, k) * math.comb(y, k) * \
            math.factorial(k) * (k**q) * ((l3/(l1*l2))**k)
        res = res + temp

    return res

# Calculation of the score vector


def score(x, y, l1, l2, l3):

    U = S(1, x, y, l1, l2, l3) / S(0, x, y, l1, l2, l3)
    res = [x-l1-U, y-l2-U, l2-y+U, l1-x+U]

    return res


def link_function_ext1(ai, aj, bi, bj, delta, eta, gamma_i, nu_j, ):  #

    try:
        l1 = min(50, math.exp(delta + ai - bj +eta*(gamma_i - nu_j)))
        l1 = max(0.1, l1)
    except OverflowError:
        l1 = 50

    try:
        l2 = min(50, math.exp(aj - bi))
        l2 = max(0.1, l2)
    except OverflowError:
        l2 = 50

    return l1, l2


def link_function_ext2(ai, aj, bi, bj, delta, eta1, eta2, gamma_i, gamma_j, nu_i, nu_j):
    try:
        l1 = min(50, math.exp(delta + ai - bj + eta1*(gamma_i - nu_j)))
        l1 = max(0.1, l1)
    except OverflowError:
        l1 = 50

    try:
        l2 = min(50, math.exp(aj - bi + eta2*(gamma_j - nu_i)))
        l2 = max(0.1, l2)
    except OverflowError:
        l2 = 50

    return l1, l2
