def gcd(a, b):
    """
    辗转相除法求最大公约数
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """
    最小公倍数等于两数积除以最大公约数
    """
    return a * b // gcd(a, b)

def smallest_multiple(n):
    """
    求1到n的最小公倍数
    """
    res = 1
    for i in range(1, n + 1):
        res = lcm(res, i)
    return res

# res = smallest_multiple(4)
# print(res)