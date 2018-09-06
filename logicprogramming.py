from kanren import *
from kanren.core import success, fail, goaleval, condeseq, eq, var, lall
from sympy.ntheory.generate import prime, isprime
import itertools as it
import time

def prime_check(x):
    if isvar(x):
       return condeseq([(eq,x,p)]
       for p in map(prime, it.count(1)))
    else:
       return success if isprime(x) else fail

x = var()
print((set(run(0,x,(membero,x,(12,14,15,19,20,21,22,23,29,30,41,44,52,62,65,85)),(prime_check,x)))))
print((run(10,x,prime_check(x))))

def left(q, p, list):
   return membero((q,p), zip(list, list[1:]))
def next(q, p, list):
   return conde([left(q, p, list)], [left(p, q, list)])

houses = var()

rules_zebraproblem = lall(
   (eq, (var(), var(), var(), var(), var()), houses),
   (membero,('Englishman', var(), var(), var(), 'red'), houses),
   (membero,('Swede', var(), var(), 'dog', var()), houses),
   (membero,('Dane', var(), 'tea', var(), var()), houses),
   (left,(var(), var(), var(), var(), 'green'),
   (var(), var(), var(), var(), 'white'), houses),
   (membero,(var(), var(), 'coffee', var(), 'green'), houses),
   (membero,(var(), 'Pall Mall', var(), 'birds', var()), houses),
   (membero,(var(), 'Dunhill', var(), var(), 'yellow'), houses),
   (eq,(var(), var(), (var(), var(), 'milk', var(), var()), var(), var()), houses),
   (eq,(('Norwegian', var(), var(), var(), var()), var(), var(), var(), var()), houses),
   (next,(var(), 'Blend', var(), var(), var()),
   (var(), var(), var(), 'cats', var()), houses),
   (next,(var(), 'Dunhill', var(), var(), var()),
   (var(), var(), var(), 'horse', var()), houses),
   (membero,(var(), 'Blue Master', 'beer', var(), var()), houses),
   (membero,('German', 'Prince', var(), var(), var()), houses),
   (next,('Norwegian', var(), var(), var(), var()),
   (var(), var(), var(), var(), 'blue'), houses),
   (next,(var(), 'Blend', var(), var(), var()),
   (var(), var(), 'water', var(), var()), houses),
   (membero,(var(), var(), var(), 'zebra', var()), houses)
)

solutions = run(0, houses, rules_zebraproblem)

output_zebra = [house for house in solutions[0] if 'zebra' in house][0][0]

print ('\n'+ output_zebra + ' owns zebra.')
