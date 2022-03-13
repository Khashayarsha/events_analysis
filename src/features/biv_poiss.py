import math 


def pmf(x,y, l1, l2, l3):
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

    return result
