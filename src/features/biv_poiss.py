import math 
import numpy as np
thresh = 30
eps = 1e-12

# def bivariate_poisson_like(a, b,  l1, l2, l3, log = False):
#     #eps = 1e-3

#     l1 = max(l1, eps)
#     l2 = max(l2, eps)
#     l3 = max(l3, eps)
#     x = min(a, b)
#     y = max(a, b)
#     t_0 = l3
#     if a < b:
#         t_1 = l1
#         t_2 = l2
#     else:
#         t_2 = l1
#         t_1 = l2

#     p_km_km = np.exp(-t_1-t_2-t_0)
#     if y == 0:
#         if log == False:
#             return p_km_km
#     else:
#         return np.log(p_km_km)
#     for k in range(1, y-x+1):
#         p_km_km *= t_2/k

#     if x == 0:
#         if log == False:
#             return p_km_km
#         else:
#             return np.log(p_km_km)

#     p_km_k = p_km_km * t_2 / (y-x+1)

#     for k in range(1, x):
#         p_k_k = t_1/k*p_km_k + t_0/k*p_km_km
#         p_k_kp = t_2/(y-x+k+1)*p_k_k + t_0/(y-x+k+1)*p_km_k
#         p_km_km = p_k_k
#         p_km_k = p_k_kp

#     if log == False:
#         return t_1/x*p_km_k+t_0/x*p_km_km
#     else:
#         return np.log(t_1/x*p_km_k+t_0/x*p_km_km)










def prob_mass_func(x,y, l1, l2, l3):
    #print(f"x,y before int-ing: {x},{y} of types {type(x), type(y)}")
    x = int(x)
    y = int(y)

    # l1 = max(l1, eps)
    # l2 = max(l2, eps)
    # l3 = max(l3, eps)

    l1 = np.clip(l1, a_min = eps, a_max = thresh)
    l2 = np.clip(l2, a_min = eps, a_max = thresh)
    l3 = np.clip(l3, a_min=eps, a_max=thresh)


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
    return max(eps,result)


def link_function(ai, aj, bi, bj, delta):  #
    #exponent1= np.clip(delta + ai - bj, a_min =-5, a_max = 20 )  # bestond eerst niet 
    #exponent2 = np.clip(aj - bi, a_min= -5, a_max = 20)
    exponent1 = delta + ai - bj 
    try:
        l1 = min(thresh, math.exp(exponent1))       #delta + ai - bj))
        l1 = max(0.01, l1)
    except OverflowError:
        l1 = thresh #was eerst 50, daarvoor 10 

    try:
        l2 = min(thresh, math.exp(aj - bi))
        l2 = max(0.01, l2)
    except OverflowError:
        l2 = thresh

    return l1, l2


def S(q, x, y, l1, l2, l3):

    if x < 0 or y < 0:
        print("ERROR, NEGATIVE VALUE FOR GOALS DETECTED IN BIV_POISSON.SCORE         (.S). INT-ing it now")

    mini = min(x, y)
    x = int(x)
    y = int(y)
    res = 0

    # l1 = max(l1, eps)
    # l2 = max(l2, eps)
    # l3 = max(l3, eps)
    
    l1 = np.clip(l1, a_min = eps, a_max = thresh)
    l2 = np.clip(l2, a_min = eps, a_max = thresh)
    l3 = np.clip(l3, a_min=eps, a_max=thresh)

    for k in range(0, mini+1):

        #fraction = np.clip( ((l3/(l1*l2))**k),   a_min = 0.01, a_max= 100 )

        temp = math.comb(x, k) * math.comb(y, k) * \
            math.factorial(k) * (k**q) * ((l3/(l1*l2))**k) #fraction #  
        res = res + temp

    return res

# Calculation of the score vector


def score(x, y, l1, l2, l3):

    U = S(1, x, y, l1, l2, l3) / S(0, x, y, l1, l2, l3)
    res = [x-l1-U, y-l2-U, l2-y+U, l1-x+U]

    return res


def score_extended(x,y,l1,l2,l3):
    U = 0 
    res = 0


def link_function_attempts(ai, aj, bi, bj, delta,eta):

    try:
        l1 = min(thresh, math.exp(delta + eta* ( ai - bj)))
        l1 = max(eps, l1) # was eerst max(0.1, l1)
    except OverflowError:
        l1 = thresh

    try:
        l2 = min(thresh, math.exp(eta*(aj - bj)))
        l2 = max(eps, l2)
    except OverflowError:
        l2 = thresh



def link_function_ext1(ai, aj, bi, bj, delta, eta, gamma_i, nu_j, ):  #

    try:
        l1 = min(thresh, math.exp(delta + ai - bj +eta*(gamma_i - nu_j)))
        l1 = max(eps, l1)
    except OverflowError:
        l1 = thresh

    try:
        l2 = min(thresh, math.exp(aj - bi))
        l2 = max(eps, l2)
    except OverflowError:
        print(f"Overflow l2: aj = {aj} bi = {bi}")
        l2 = thresh

    return l1, l2


def link_function_ext2(ai, aj, bi, bj, delta, eta1, eta2, gamma_i, gamma_j, nu_i, nu_j):
    try:        
            
        l1 = min(thresh, math.exp(delta + ai - bj + eta1*(gamma_i - nu_j)))
        l1 = max(eps, l1)
    except OverflowError:
        l1 = thresh
        print(f"Overflow l2: delta = {delta} aj = {ai} bi = {bj}  eta1 (gamma_i - nu_j)= {eta1}*({gamma_i} - {nu_j})")

    try:
        l2 = min(thresh, math.exp(aj - bi + eta2*(gamma_j - nu_i)))
        l2 = max(eps, l2)
    except OverflowError:
        print(f"Overflow l2: aj = {aj} bi = {bi}  eta2 (gamma_j - nu_i)= {eta2}*({gamma_j} - {nu_i})")
        l2 = thresh
        #l2 = max()

    return l1, l2


# def link_function_ext_simultaneous1(alpha_i, alpha_j, beta_i, beta_j, delta1, eta1, gamma_i, nu_j):
#     try:
#         l1 = min(50, math.exp(delta1 + alpha_i - beta_j + eta1*(gamma_i - nu_j)))
#         l1 = max(0.1, l1)
#     except OverflowError:
#         l1 = 50

#     try:
#         l2 = min(50, math.exp(alpha_j - beta_i + eta2*(gamma_j - nu_i)))
#         l2 = max(0.1, l2)
#     except OverflowError:
#         l2 = 50

#     return l1, l2


