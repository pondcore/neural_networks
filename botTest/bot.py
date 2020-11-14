# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.microsoft import EdgeChromiumDriverManager

# driver = webdriver.Edge(EdgeChromiumDriverManager().install())
# driver = webdriver.Edge(executable_path='C:\\Users\\Ninja\\AppData\\Local\\Programs\\Python\\Python38-32\\msedgedriver.exe')
# def myDistance(u, a, t):
#     s = (u*t) + 0.5*a*(t**2)
#     return s
# def myVelocity(u,a,s):
#     v = (u**2) + 2*a*s
#     return v**0.5
# s = myDistance(20,3,5)
# v = myVelocity(20,3,s)
# print("{:.2f} {:.2f}".format(s,v))
def getPricePlusVAT(country, price):
    if country == 'TH':
        vat = price*1.07
    elif country == 'UK':
        vat = price*1.2
    elif country == 'FI':
        vat = price*1.24
    elif country == 'SE':
        vat = price*1.25
    else: return -1
    return vat