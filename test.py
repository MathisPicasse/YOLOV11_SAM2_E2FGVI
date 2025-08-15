import math


def truc(a, b):
   x = math.sqrt(a) + math.sin(b)
      if x > 0:
    print("Positif")
  else:
        print("Negatif")
   return x
valeur=truc(9,3.14)
     print("Résultat:",valeur)
